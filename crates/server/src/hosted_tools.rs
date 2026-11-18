use serde::{Deserialize, Serialize};
use std::{future::Future, pin::Pin, sync::Arc, time::Duration};
use thiserror::Error;
use url::Url;

pub struct WebSearchRequest {
    pub query: String,
    pub allowed_domains: Vec<String>,
    pub max_results: usize,
    pub control: Arc<crate::RequestControl>,
    pub deadline: std::time::Instant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Error)]
pub enum HostedToolError {
    #[error("hosted web search is disabled")]
    Disabled,
    #[error("invalid hosted-tool request: {0}")]
    InvalidRequest(String),
    #[error("hosted web search failed: {0}")]
    Provider(String),
}

pub trait HostedToolExecutor: Send + Sync {
    fn web_search<'a>(
        &'a self,
        request: &'a WebSearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<WebSearchResult>, HostedToolError>> + Send + 'a>>;
}

#[derive(Default)]
pub struct DisabledHostedTools;

impl HostedToolExecutor for DisabledHostedTools {
    fn web_search<'a>(
        &'a self,
        _request: &'a WebSearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<WebSearchResult>, HostedToolError>> + Send + 'a>> {
        Box::pin(async { Err(HostedToolError::Disabled) })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebSearchPolicy {
    pub endpoint: String,
    pub timeout: Duration,
    pub max_results: usize,
    pub max_response_bytes: usize,
    pub allowed_domains: Vec<String>,
}

impl Default for WebSearchPolicy {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost/".into(),
            timeout: Duration::from_secs(10),
            max_results: 5,
            max_response_bytes: 1024 * 1024,
            allowed_domains: Vec::new(),
        }
    }
}

fn normalize_domain(domain: &str) -> Result<String, HostedToolError> {
    let normalized = domain.trim().trim_end_matches('.').to_ascii_lowercase();
    if normalized.is_empty()
        || !normalized
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    {
        return Err(HostedToolError::InvalidRequest(format!(
            "invalid allowed domain {domain:?}"
        )));
    }
    Ok(normalized)
}

fn validate_requested_domain(
    domain: &str,
    policy: &WebSearchPolicy,
) -> Result<String, HostedToolError> {
    let domain = normalize_domain(domain)?;
    if !policy.allowed_domains.is_empty()
        && !policy.allowed_domains.iter().any(|allowed| {
            normalize_domain(allowed)
                .is_ok_and(|allowed| domain == allowed || domain.ends_with(&format!(".{allowed}")))
        })
    {
        return Err(HostedToolError::InvalidRequest(
            "request allowed_domains exceeds operator policy".into(),
        ));
    }
    Ok(domain)
}

fn validate_url(url: &str, domains: &[String]) -> Result<Url, HostedToolError> {
    let parsed = Url::parse(url)
        .map_err(|error| HostedToolError::InvalidRequest(format!("invalid result URL: {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(HostedToolError::InvalidRequest(
            "result URL must use HTTP(S)".into(),
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| HostedToolError::InvalidRequest("result URL has no host".into()))?
        .trim_matches(['[', ']'])
        .to_ascii_lowercase();
    if host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| match address {
                std::net::IpAddr::V4(address) => {
                    address.is_private()
                        || address.is_loopback()
                        || address.is_link_local()
                        || address.is_unspecified()
                }
                std::net::IpAddr::V6(address) => {
                    address.is_loopback() || address.is_unspecified() || address.is_unique_local()
                }
            })
    {
        return Err(HostedToolError::InvalidRequest(
            "result URL targets a private network".into(),
        ));
    }
    if !domains.is_empty()
        && !domains
            .iter()
            .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
    {
        return Err(HostedToolError::InvalidRequest(
            "result URL exceeds requested domains".into(),
        ));
    }
    Ok(parsed)
}

pub struct SearxngHostedTools {
    client: reqwest::Client,
    policy: WebSearchPolicy,
}

impl SearxngHostedTools {
    pub fn new(policy: WebSearchPolicy) -> Result<Self, HostedToolError> {
        let endpoint = Url::parse(&policy.endpoint).map_err(|error| {
            HostedToolError::InvalidRequest(format!("invalid SearXNG endpoint: {error}"))
        })?;
        if !matches!(endpoint.scheme(), "http" | "https") || endpoint.host_str().is_none() {
            return Err(HostedToolError::InvalidRequest(
                "SearXNG endpoint must be an absolute HTTP(S) URL".into(),
            ));
        }
        if policy.max_results == 0 || policy.max_response_bytes == 0 {
            return Err(HostedToolError::InvalidRequest(
                "web-search limits must be nonzero".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| HostedToolError::Provider(error.to_string()))?;
        Ok(Self { client, policy })
    }

    fn effective_domains(&self, requested: &[String]) -> Result<Vec<String>, HostedToolError> {
        let requested = requested
            .iter()
            .map(|domain| validate_requested_domain(domain, &self.policy))
            .collect::<Result<Vec<_>, _>>()?;
        if self.policy.allowed_domains.is_empty() || !requested.is_empty() {
            return Ok(requested);
        }
        self.policy
            .allowed_domains
            .iter()
            .map(|domain| normalize_domain(domain))
            .collect()
    }
}

#[derive(Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Deserialize)]
struct SearxngResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default, alias = "content")]
    snippet: String,
}

fn bounded_request_timeout(
    policy_timeout: Duration,
    request_timeout: Duration,
) -> Result<Duration, HostedToolError> {
    if request_timeout.is_zero() {
        return Err(HostedToolError::InvalidRequest(
            "web-search timeout must be nonzero".into(),
        ));
    }
    Ok(policy_timeout.min(request_timeout))
}

impl HostedToolExecutor for SearxngHostedTools {
    fn web_search<'a>(
        &'a self,
        request: &'a WebSearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<WebSearchResult>, HostedToolError>> + Send + 'a>> {
        Box::pin(async move {
            let query = request.query.trim();
            if query.is_empty() {
                return Err(HostedToolError::InvalidRequest(
                    "web-search query must not be empty".into(),
                ));
            }
            let domains = self.effective_domains(&request.allowed_domains)?;
            let query = if domains.is_empty() {
                query.to_owned()
            } else {
                format!(
                    "{} {}",
                    query,
                    domains
                        .iter()
                        .map(|domain| format!("site:{domain}"))
                        .collect::<Vec<_>>()
                        .join(" OR ")
                )
            };
            let limit = request.max_results.min(self.policy.max_results);
            if limit == 0 {
                return Err(HostedToolError::InvalidRequest(
                    "web-search max_results must be nonzero".into(),
                ));
            }
            let remaining = request
                .control
                .remaining_until(request.deadline)
                .map_err(|error| HostedToolError::Provider(error.to_string()))?;
            let timeout = bounded_request_timeout(self.policy.timeout, remaining)?;
            let mut response = self
                .client
                .get(&self.policy.endpoint)
                .query(&[("q", query.as_str()), ("format", "json")])
                .timeout(timeout)
                .send()
                .await
                .map_err(|error| HostedToolError::Provider(error.to_string()))?
                .error_for_status()
                .map_err(|error| HostedToolError::Provider(error.to_string()))?;
            let mut body = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|error| HostedToolError::Provider(error.to_string()))?
            {
                if body.len().saturating_add(chunk.len()) > self.policy.max_response_bytes {
                    return Err(HostedToolError::Provider(
                        "web-search response exceeds configured size limit".into(),
                    ));
                }
                body.extend_from_slice(&chunk);
            }
            let decoded: SearxngResponse = serde_json::from_slice(&body)
                .map_err(|error| HostedToolError::Provider(error.to_string()))?;
            Ok(decoded
                .results
                .into_iter()
                .filter_map(|result| {
                    let parsed = validate_url(&result.url, &domains).ok()?;
                    Some(WebSearchResult {
                        title: result.title,
                        url: parsed.into(),
                        snippet: result.snippet,
                    })
                })
                .take(limit)
                .collect())
        })
    }
}

pub fn disabled_hosted_tools() -> Arc<dyn HostedToolExecutor> {
    Arc::new(DisabledHostedTools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_domains_are_normalized_and_bounded_by_policy() {
        let policy = WebSearchPolicy {
            allowed_domains: vec!["example.com".into()],
            ..WebSearchPolicy::default()
        };

        assert_eq!(
            validate_requested_domain("Docs.Example.com", &policy).unwrap(),
            "docs.example.com"
        );
        assert!(validate_requested_domain("example.net", &policy).is_err());
        assert!(validate_requested_domain("https://example.com/path", &policy).is_err());
        assert!(validate_requested_domain("*.example.com", &policy).is_err());
    }

    #[test]
    fn result_urls_exclude_private_network_targets() {
        for url in [
            "http://127.0.0.1/admin",
            "http://10.0.0.1/",
            "http://[::1]/",
            "file:///etc/passwd",
        ] {
            assert!(validate_url(url, &[]).is_err(), "{url} must be rejected");
        }
        assert!(validate_url("https://docs.example.com/page", &[]).is_ok());
    }

    #[test]
    fn request_limits_are_clamped_to_operator_policy() {
        let policy = WebSearchPolicy {
            max_results: 3,
            ..WebSearchPolicy::default()
        };
        let executor = SearxngHostedTools::new(policy).unwrap();
        let request = WebSearchRequest {
            query: "test".into(),
            allowed_domains: vec![],
            max_results: usize::MAX,
            control: Arc::new(crate::RequestControl::new()),
            deadline: std::time::Instant::now() + Duration::from_secs(2),
        };

        assert_eq!(request.max_results.min(executor.policy.max_results), 3);
        assert_eq!(
            bounded_request_timeout(executor.policy.timeout, Duration::from_secs(2)).unwrap(),
            Duration::from_secs(2)
        );
        assert!(bounded_request_timeout(executor.policy.timeout, Duration::ZERO).is_err());
        assert_eq!(
            bounded_request_timeout(executor.policy.timeout, Duration::from_secs(20)).unwrap(),
            executor.policy.timeout
        );
    }
    #[tokio::test]
    async fn async_provider_accepts_responses_slower_than_short_poll_intervals() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body = br#"{"results":[]}"#;
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            std::thread::sleep(Duration::from_millis(200));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });
        let executor = SearxngHostedTools::new(WebSearchPolicy {
            endpoint: format!("http://{address}/search"),
            timeout: Duration::from_secs(2),
            ..WebSearchPolicy::default()
        })
        .unwrap();
        let request = WebSearchRequest {
            query: "slow provider".into(),
            allowed_domains: vec![],
            max_results: 5,
            control: Arc::new(crate::RequestControl::new()),
            deadline: std::time::Instant::now() + Duration::from_secs(3),
        };
        let started = std::time::Instant::now();
        let results = executor.web_search(&request).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(150));
        assert!(results.is_empty());
        server.join().unwrap();
    }
}
