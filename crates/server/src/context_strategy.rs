use serde::{Deserialize, Serialize};

const BUILTIN_STRATEGY_VERSION: u32 = 1;
pub const WINDOW_TAIL_STRATEGY_ID: &str = "window_tail";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionNoMatchFallback {
    #[default]
    NoCompact,
    Reject,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTrigger {
    #[default]
    Request,
    None,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompactionRequest {
    #[serde(default)]
    pub declaration_id: Option<String>,
    #[serde(default)]
    pub strategy_preferences: Vec<String>,
    #[serde(default)]
    pub fallback_when_no_match: CompactionNoMatchFallback,
    #[serde(default)]
    pub trigger: CompactionTrigger,
    #[serde(default)]
    pub target_tokens: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompactionDeclaration {
    pub declaration_id: String,
    pub context_id: String,
    pub strategy_preferences: Vec<String>,
    pub target_tokens: Option<usize>,
    pub expires_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CreateCompactionDeclaration {
    pub context_id: String,
    #[serde(default)]
    pub strategy_preferences: Vec<String>,
    #[serde(default)]
    pub target_tokens: Option<usize>,
    pub expires_in_ms: u64,
}

impl Default for CompactionRequest {
    fn default() -> Self {
        Self {
            declaration_id: None,
            strategy_preferences: Vec::new(),
            fallback_when_no_match: CompactionNoMatchFallback::NoCompact,
            trigger: CompactionTrigger::Request,
            target_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CompactionStrategyInfo {
    pub strategy_id: String,
    pub version: u32,
    pub compact_mode: String,
    pub default_target_tokens: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompactionStrategyCatalog {
    pub version: u32,
    pub strategies: Vec<CompactionStrategyInfo>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionResultReason {
    RequestSatisfied,
    AnchorsOnly,
    BudgetTrimmed,
    None,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompactionResult {
    pub compact_mode: String,
    pub compact_reason: CompactionResultReason,
    pub requested_strategy_ids: Vec<String>,
    pub selected_strategy_id: Option<String>,
    pub requested_strategy_budget: usize,
    pub resulting_budget: usize,
    pub retained_indices: Vec<usize>,
    pub retained_message_count: usize,
    pub source_message_count: usize,
    pub retained_anchor_count: usize,
    pub success: bool,
    pub fallback: bool,
    pub resulting_context_epoch: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompactionProposal {
    pub resulting_tokens: Vec<String>,
    pub result: CompactionResult,
}

pub fn strategy_catalog() -> CompactionStrategyCatalog {
    CompactionStrategyCatalog {
        version: BUILTIN_STRATEGY_VERSION,
        strategies: vec![CompactionStrategyInfo {
            strategy_id: WINDOW_TAIL_STRATEGY_ID.into(),
            version: BUILTIN_STRATEGY_VERSION,
            compact_mode: "window_tail".into(),
            default_target_tokens: 3072,
            reason: "Keep newest turns while preserving anchor boundaries".into(),
        }],
    }
}

pub fn canonical_strategy_ids() -> Vec<String> {
    strategy_catalog()
        .strategies
        .into_iter()
        .map(|strategy| strategy.strategy_id)
        .collect()
}

pub fn select_strategy(preferences: &[String]) -> Option<String> {
    if preferences.is_empty() {
        return Some(WINDOW_TAIL_STRATEGY_ID.into());
    }
    for requested in preferences {
        if strategy_catalog()
            .strategies
            .iter()
            .any(|strategy| &strategy.strategy_id == requested)
        {
            return Some(requested.clone());
        }
    }
    None
}

pub fn deterministic_strategy_id(strategy: &str) -> String {
    format!("{strategy}:v{BUILTIN_STRATEGY_VERSION}")
}

pub fn apply_window_tail_strategy(
    source_tokens: &[String],
    target: usize,
    requested: &[String],
) -> Result<CompactionProposal, String> {
    let requested_budget = target.max(1);
    let source_message_count = source_messages(source_tokens);
    let anchors = anchor_indices(source_tokens);
    let turns = split_turns(source_tokens);

    if source_tokens.is_empty() {
        return Ok(CompactionProposal {
            resulting_tokens: Vec::new(),
            result: CompactionResult {
                compact_mode: "window_tail".into(),
                compact_reason: CompactionResultReason::None,
                requested_strategy_ids: requested.to_vec(),
                selected_strategy_id: Some(WINDOW_TAIL_STRATEGY_ID.into()),
                requested_strategy_budget: requested_budget,
                resulting_budget: 0,
                retained_indices: Vec::new(),
                retained_message_count: 0,
                source_message_count: 0,
                retained_anchor_count: 0,
                success: true,
                fallback: false,
                resulting_context_epoch: 1,
            },
        });
    }

    if source_tokens.len() <= requested_budget {
        let retained: Vec<usize> = (0..source_tokens.len()).collect();
        return Ok(CompactionProposal {
            resulting_tokens: source_tokens.to_vec(),
            result: CompactionResult {
                compact_mode: "window_tail".into(),
                compact_reason: CompactionResultReason::RequestSatisfied,
                requested_strategy_ids: requested.to_vec(),
                selected_strategy_id: Some(WINDOW_TAIL_STRATEGY_ID.into()),
                requested_strategy_budget: requested_budget,
                resulting_budget: source_tokens.len(),
                retained_indices: retained.clone(),
                retained_message_count: source_messages(source_tokens),
                source_message_count,
                retained_anchor_count: anchors.len(),
                success: true,
                fallback: false,
                resulting_context_epoch: 1,
            },
        });
    }

    let mut retained_turns = Vec::new();
    let mut used_tokens = 0usize;

    for turn in turns.iter().rev() {
        if used_tokens + turn.len() > requested_budget {
            if includes_policy_anchor(turn) {
                continue;
            }
            break;
        }
        retained_turns.push(turn);
        used_tokens += turn.len();
        if used_tokens >= requested_budget {
            break;
        }
    }

    if retained_turns.is_empty() {
        let start = anchors.first().copied().unwrap_or(0);
        let fallback_count = source_tokens.len().saturating_sub(start);
        let retained_count = fallback_count.min(requested_budget);
        let resulting_tokens = source_tokens[source_tokens.len() - retained_count..].to_vec();
        let retained_indices: Vec<usize> = (source_tokens.len() - retained_count..source_tokens.len()).collect();
        return Ok(CompactionProposal {
            resulting_tokens,
            result: CompactionResult {
                compact_mode: "window_tail".into(),
                compact_reason: CompactionResultReason::AnchorsOnly,
                requested_strategy_ids: requested.to_vec(),
                selected_strategy_id: Some(WINDOW_TAIL_STRATEGY_ID.into()),
                requested_strategy_budget: requested_budget,
                resulting_budget: retained_indices.len(),
                retained_indices,
                retained_message_count: 1,
                source_message_count,
                retained_anchor_count: anchors.len().min(1),
                success: true,
                fallback: true,
                resulting_context_epoch: 1,
            },
        });
    }

    retained_turns.reverse();
    let resulting_tokens = retained_turns
        .into_iter()
        .flat_map(|turn| turn.iter().cloned())
        .collect::<Vec<_>>();
    let retained_indices = resulting_tokens
        .iter()
        .map(|token| source_tokens.iter().position(|source| source == token).unwrap_or(0))
        .collect::<Vec<_>>();

    Ok(CompactionProposal {
        result: CompactionResult {
            compact_mode: "window_tail".into(),
            compact_reason: CompactionResultReason::BudgetTrimmed,
            requested_strategy_ids: requested.to_vec(),
            selected_strategy_id: Some(WINDOW_TAIL_STRATEGY_ID.into()),
            requested_strategy_budget: requested_budget,
            resulting_budget: used_tokens,
            retained_indices: retained_indices.clone(),
            retained_message_count: split_turns(&resulting_tokens).len(),
            source_message_count,
            retained_anchor_count: retained_indices
                .iter()
                .filter(|index| anchors.contains(index))
                .count(),
            success: true,
            fallback: false,
            resulting_context_epoch: 1,
        },
        resulting_tokens,
    })
}

fn split_turns(tokens: &[String]) -> Vec<Vec<String>> {
    let mut turns = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for token in tokens {
        if is_role_boundary(token) && !current.is_empty() {
            turns.push(std::mem::take(&mut current));
        }
        current.push(token.clone());
    }
    if !current.is_empty() {
        turns.push(current);
    }
    if turns.is_empty() {
        vec![tokens.to_vec()]
    } else {
        turns
    }
}

fn is_role_boundary(token: &str) -> bool {
    token.contains("<start_of_turn") || token == "</s>"
}

fn anchor_indices(tokens: &[String]) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            if token.starts_with("<start_of_turn>system")
                || token.starts_with("<start_of_turn>tool")
                || token.starts_with("<system")
                || token.starts_with("<|system|>")
            {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn includes_policy_anchor(tokens: &[String]) -> bool {
    tokens
        .iter()
        .any(|token| token.starts_with("<start_of_turn>system") || token.starts_with("<start_of_turn>tool"))
}

fn source_messages(tokens: &[String]) -> usize {
    if tokens.is_empty() {
        0
    } else {
        split_turns(tokens).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_ids_are_versioned() {
        let catalog = strategy_catalog();
        assert_eq!(catalog.version, 1);
        assert_eq!(catalog.strategies[0].strategy_id, WINDOW_TAIL_STRATEGY_ID);
        assert_eq!(catalog.strategies[0].version, 1);
    }

    #[test]
    fn select_prefers_requested_strategy_if_available() {
        let selected = select_strategy(&[
            "unknown".into(),
            WINDOW_TAIL_STRATEGY_ID.into(),
        ]);
        assert_eq!(selected.as_deref(), Some(WINDOW_TAIL_STRATEGY_ID));
    }

    #[test]
    fn window_tail_keeps_anchor_when_possible() {
        let tokens: Vec<String> = vec![
            "sys".into(),
            "<start_of_turn>user".into(),
            "hello".into(),
            "world".into(),
            "<start_of_turn>assistant".into(),
            "reply".into(),
            "chain".into(),
        ];
        let proposal = apply_window_tail_strategy(&tokens, 6, &[WINDOW_TAIL_STRATEGY_ID.into()]).unwrap();
        let expected: Vec<String> = vec![
            "<start_of_turn>user".into(),
            "hello".into(),
            "world".into(),
            "<start_of_turn>assistant".into(),
            "reply".into(),
            "chain".into(),
        ];
        assert_eq!(proposal.resulting_tokens, expected);
        assert_eq!(proposal.result.retained_anchor_count, 0);
    }

    #[test]
    fn strategy_output_is_idempotent_for_same_context() {
        let tokens = vec![
            "<start_of_turn>system".into(),
            "rules".into(),
            "<start_of_turn>user".into(),
            "hello".into(),
            "<start_of_turn>assistant".into(),
            "ok".into(),
        ];
        let first =
            apply_window_tail_strategy(&tokens, 4, &[WINDOW_TAIL_STRATEGY_ID.into()]).unwrap();
        let second =
            apply_window_tail_strategy(&tokens, 4, &[WINDOW_TAIL_STRATEGY_ID.into()]).unwrap();
        assert_eq!(first.resulting_tokens, second.resulting_tokens);
        assert_eq!(first.result, second.result);
    }
}
