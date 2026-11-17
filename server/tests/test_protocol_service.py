#!/usr/bin/env python3
"""Real-HTTP regression suite for the production Source controllers.

Build/run without a model or GPU:
  python3 server/tests/test_protocol_service.py [terrane-build-artifact] -v
Without an artifact, this stages the fixture and the actual production controllers
under `server/.trn/protocol-fixture`, then builds and runs it. The separate
`tools/test-hybrid-service.py` remains the production-engine startup smoke test.
"""
import http.client
import inspect
import json
import os
import re
import shutil
import selectors
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from openai import OpenAI

PROJECT = Path(__file__).resolve().parents[2]
SERVICE = PROJECT / "server"


def fixture_artifact():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return Path(sys.argv.pop(1)).resolve()
    fixture = Path(__file__).resolve().parent / "protocol-fixture"
    staged = SERVICE / ".trn" / "protocol-fixture"
    staged.mkdir(parents=True, exist_ok=True)
    for source, destination, ignore in (
        (fixture / "src", staged / "src", None),
        (SERVICE / "src", staged / "service", shutil.ignore_patterns("main.trn")),
        (SERVICE / "templates", staged / "templates", None),
    ):
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=ignore)
    shutil.copy2(fixture / "fixture.rs", staged / "fixture.rs")
    manifest = (fixture / "package.toml").read_text()
    manifest = re.sub(
        r'(?m)^(path\s*=\s*)"([^"]+)"',
        lambda match: match[1] + json.dumps(os.path.relpath(fixture / match[2], staged)),
        manifest,
    )
    (staged / "package.toml").write_text(manifest)
    (staged / ".cargo").mkdir(exist_ok=True)
    (staged / ".cargo" / "config.toml").write_text(
        "[patch.crates-io]\ncusco-server = { path = "
        + json.dumps(str(PROJECT / "crates" / "server")) + " }\n"
    )
    result = subprocess.run(
        ["terrane", "build", str(staged)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    # `terrane build` prints the resulting executable path as its final line.
    return Path(result.stdout.strip().splitlines()[-1]).resolve()


BINARY = fixture_artifact()
artifact_file = os.environ.get("CUSCO_PROTOCOL_ARTIFACT_FILE")
if artifact_file:
    Path(artifact_file).write_text(str(BINARY) + "\n", encoding="utf-8")


class ProtocolServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cusco-source-http-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("db", "models", "spill", "local"):
            (self.root / name).mkdir()
        (self.root / "user.yaml").write_text("models: []\n")
        self.config = {
            "version": 1,
            "listen": "127.0.0.1:0",
            "paths": {
                "database": str(self.root / "db/catalog.sqlite"),
                "models": str(self.root / "models"),
                "spill": str(self.root / "spill"),
                "user_models": str(self.root / "local"),
                "user_config": str(self.root / "user.yaml"),
            },
            "openapi": {"docs_ui": {"enabled": False}},
            "execution": {
                "device_capacity": "64 MiB", "host_capacity": "64 MiB",
                "storage_capacity": "128 MiB", "context_reserve": "1 MiB",
                "context_tokens": 128, "gpu_layers": 0,
            },
            "hosted_tools": {"web_search": {
                "enabled": True, "endpoint": "http://fixture.invalid", "timeout_ms": 10000,
                "max_results": 8, "max_response_bytes": 1048576,
                "allowed_domains": [],
            }},
            "server": {
                "active_requests": 2, "queue_count": 4, "queue_bytes": 1048576,
                "request_bytes": 65536, "pre_queue_concurrency": 4,
                "default_output_tokens": 4, "header_bytes": 32768,
                "body_timeout_ms": 10000, "wall_time_ms": 30000,
                "active_time_ms": 20000, "stream_buffer": 1,
                "shutdown_grace_ms": 1000, "compaction_enabled": True,
                "compaction_workers": 1, "compaction_declarations": 16,
                "compaction_declaration_rate_per_minute": 8,
                "compaction_declaration_lifetime_ms": 60000,
            },
            "scheduler": {
                "version": 1, "interactive_weight": 4, "standard_weight": 2,
                "batch_weight": 1, "deficit_refill": 1, "prefill_tokens": 32,
                "promotion_rounds": 64, "diagnostics_enabled": False,
                "diagnostic_capacity": 128,
            },
            "vision": {
                "max_images": 8, "max_encoded_bytes": 16777216,
                "max_decoded_bytes": 12582912, "max_dimension": 8192,
                "max_total_pixels": 32000000, "retention_capacity": "256 MiB",
                "metadata_allowlist": [],
            },
        }
        self.config_path = self.root / "config.yaml"
        self.environment = os.environ.copy()
        self.environment.pop("CUSCO_HTTP_DEBUG", None)
        self.environment["CUSCO_BEARER_TOKEN"] = "source-fixture-token"
        self.process = None
        self.port = None
        self.stderr = None

    def openai(self):
        return OpenAI(
            base_url=f"http://{self.host}:{self.port}/openai/v1",
            api_key="source-fixture-token",
            max_retries=0,
        )

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def start(self, *args):
        self.write_config()
        self.stderr = (self.root / "stderr").open("w+")
        self.addCleanup(self.stderr.close)
        self.process = subprocess.Popen(
            [str(BINARY), str(self.config_path), *args],
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
        )
        self.addCleanup(self.stop)
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(15), "Source fixture did not bind its listener")
        line = self.process.stdout.readline().strip()
        prefix = "Cusco protocol fixture listening on "
        if not line.startswith(prefix):
            self.stderr.seek(0)
            self.fail(f"fixture failed to start: {line}\n{self.stderr.read()}")
        host, port = line.removeprefix(prefix).rsplit(":", 1)
        self.host, self.port = host, int(port)
        self.assertGreater(self.port, 0)

    def stop(self):
        process = self.process
        if process is None:
            return
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        self.process = None

    def request(self, method, path, body=None, authenticated=True, headers=None):
        request_headers = dict(headers or {})
        if authenticated:
            request_headers["Authorization"] = "Bearer source-fixture-token"
        payload = None if body is None else json.dumps(body).encode()
        if payload is not None:
            request_headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
        finally:
            connection.close()

    def sdk_body(self, operation, body):
        parameters = inspect.signature(operation).parameters
        accepted = {
            name for name, parameter in parameters.items()
            if parameter.kind != inspect.Parameter.VAR_KEYWORD
            and name not in {"extra_body", "extra_headers", "extra_query", "timeout"}
        }
        arguments = {}
        extra = {}
        for key, value in body.items():
            (arguments if key in accepted else extra)[key] = value
        if extra:
            arguments["extra_body"] = extra
        return arguments

    def json_request(self, method, path, body=None, **kwargs):
        if not path.startswith("/openai/v1/") or path == "/openai/v1/openapi.json":
            status, headers, payload = self.request(method, path, body, **kwargs)
            parsed = json.loads(payload) if payload else None
            return status, headers, parsed
        client = self.openai()
        try:
            if path == "/openai/v1/models" and method == "GET":
                raw = client.models.with_raw_response.list()
            elif path == "/openai/v1/completions" and method == "POST":
                operation = client.completions.create
                raw = client.completions.with_raw_response.create(**self.sdk_body(operation, body))
            elif path == "/openai/v1/chat/completions" and method == "POST":
                operation = client.chat.completions.create
                raw = client.chat.completions.with_raw_response.create(**self.sdk_body(operation, body))
            elif path == "/openai/v1/responses" and method == "POST":
                operation = client.responses.create
                raw = client.responses.with_raw_response.create(**self.sdk_body(operation, body))
            elif path.startswith("/openai/v1/responses/") and method == "GET":
                raw = client.responses.with_raw_response.retrieve(path.rsplit("/", 1)[1])
            elif path.startswith("/openai/v1/responses/") and method == "DELETE":
                raw = client.responses.with_raw_response.delete(path.rsplit("/", 1)[1])
                # SDK 2.11 declares deletion as returning None; retain the
                # server's deletion document through its raw-response wrapper.
                return raw.status_code, {key.lower(): value for key, value in raw.headers.items()}, raw.http_response.json()
            else:
                raise AssertionError(f"unsupported OpenAI request: {method} {path}")
            parsed = raw.parse()
            return raw.status_code, {key.lower(): value for key, value in raw.headers.items()}, parsed.model_dump(mode="json", exclude_unset=True)
        except Exception as error:
            from openai import APIStatusError
            if not isinstance(error, APIStatusError):
                raise
            return error.status_code, {key.lower(): value for key, value in error.response.headers.items()}, error.response.json()

    def diagnostic_text(self):
        self.stderr.flush()
        self.stderr.seek(0)
        return self.stderr.read()

    def test_safe_transport_diagnostics_redact_credentials_and_content(self):
        secret = "fixture-private-body-value"
        self.start("safe")
        status, _, _ = self.json_request(
            "POST", "/openai/v1/completions",
            {"model": "m", "prompt": secret, "max_tokens": 1},
        )
        self.assertEqual(status, 200)
        safe = self.diagnostic_text()
        self.assertIn('"level":"safe"', safe)
        self.assertNotIn(secret, safe)
        self.assertNotIn("source-fixture-token", safe)
        self.assertIn("[REDACTED]", safe)

    def test_full_transport_diagnostics_capture_credentials_and_content(self):
        secret = "fixture-private-body-value"
        self.start("full")
        status, _, _ = self.json_request(
            "POST", "/openai/v1/completions",
            {"model": "m", "prompt": secret, "max_tokens": 1},
        )
        self.assertEqual(status, 200)
        full = self.diagnostic_text()
        self.assertIn('"level":"full"', full)
        self.assertIn(secret, full)
        self.assertIn("source-fixture-token", full)

    def test_shutdown_drains_an_inflight_stream(self):
        self.start()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        connection.request(
            "POST", "/openai/v1/chat/completions",
            json.dumps({"model": "m", "messages": [{"role": "user", "content": "[[gate:shutdown]] held"}],
                        "max_completion_tokens": 1, "stream": True}),
            {"Authorization": "Bearer source-fixture-token", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.wait_for_file(self.gate_path("shutdown", "started"))
        self.process.stdin.close()
        self.gate_path("shutdown", "release").write_text("release")
        body = response.read().decode()
        self.assertIn("data: [DONE]", body)
        connection.close()
        self.assertEqual(self.process.wait(timeout=10), 0)

    def test_restart_discards_runtime_contexts_and_admission_state(self):
        self.start()
        status, _, context = self.json_request("POST", "/cusco/v1/contexts")
        self.assertEqual(status, 200)
        context_id = context["id"]
        self.stop()
        self.start()
        self.assertEqual(self.request("GET", f"/cusco/v1/contexts/{context_id}")[0], 404)
        status, _, metrics = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["admission"]["active"], 0)
        self.assertEqual(metrics["admission"]["queued"], 0)

    def gate_path(self, gate_id, suffix):
        return self.root / "db" / f"catalog.fixture-gates/{gate_id}.{suffix}"

    def wait_for_file(self, path, timeout=5):
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if path.exists():
                return
            time.sleep(0.005)
        self.fail(f"fixture readiness marker did not appear: {path.name}")

    def test_admission_overload_and_retry_after_are_http_contracts(self):
        self.config["server"]["active_requests"] = 1
        self.config["server"]["queue_count"] = 1
        self.start()
        import threading
        results = {}

        def submit(name):
            results[name] = self.json_request(
                "POST", "/openai/v1/completions",
                {"model": "m", "prompt": f"[[gate:{name}]] hold", "max_tokens": 1},
            )

        active = threading.Thread(target=submit, args=("active",), daemon=True)
        active.start()
        self.wait_for_file(self.gate_path("active", "started"))
        queued = threading.Thread(target=submit, args=("queued",), daemon=True)
        queued.start()
        # Status is a real authenticated route; waiting on its queue metric makes
        # overload deterministic without scheduling guesses or sleeps.
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            _, _, status = self.json_request("GET", "/cusco/v1/status")
            if status["admission"]["queued"] == 1:
                break
            time.sleep(0.005)
        else:
            self.fail("second request never reached the native admission queue")
        missing_status, _, missing = self.json_request(
            "POST", "/openai/v1/completions",
            {"model": "absent", "prompt": "not admitted", "max_tokens": 1},
        )
        self.assertEqual(missing_status, 404)
        self.assertIn("error", missing)
        overloaded_status, overload_headers, overload = self.json_request(
            "POST", "/openai/v1/completions",
            {"model": "m", "prompt": "overflow", "max_tokens": 1},
        )
        self.assertEqual(overloaded_status, 429)
        self.assertEqual(overload_headers.get("retry-after"), "1")
        self.assertEqual(overload["error"]["code"], "queue_overloaded")
        self.gate_path("active", "release").write_text("release")
        active.join(5)
        self.assertFalse(active.is_alive())
        self.wait_for_file(self.gate_path("queued", "started"))
        self.gate_path("queued", "release").write_text("release")
        queued.join(5)
        self.assertFalse(queued.is_alive())
        self.assertEqual(results["active"][0], 200)
        self.assertEqual(results["queued"][0], 200)
        _, _, metrics = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(metrics["admission"]["active"], 0)
        self.assertEqual(metrics["admission"]["queued"], 0)

    def test_auth_and_bounded_body_timeout_happen_before_body_decode(self):
        self.config["server"]["body_timeout_ms"] = 200
        self.start()
        # Unauthorized raw requests must be rejected before awaiting an absent
        # request body; no JSON bytes are sent despite the declared length.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.putrequest("POST", "/openai/v1/completions")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "4096")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
        finally:
            connection.close()

        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.putrequest("POST", "/openai/v1/completions")
            connection.putheader("Authorization", "Bearer source-fixture-token")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "4096")
            connection.endheaders(b'{"model":')
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, 408)
            self.assertEqual(json.loads(body)["error"]["code"], "body_timeout")
        finally:
            connection.close()

    def test_wall_deadline_cancels_gated_work(self):
        self.config["server"]["wall_time_ms"] = 400
        self.start()
        import threading
        outcome = {}

        def run_deadline():
            outcome["deadline"] = self.json_request(
                "POST", "/openai/v1/completions",
                {"model": "m", "prompt": "[[gate:deadline]] held", "max_tokens": 1},
            )

        worker = threading.Thread(target=run_deadline, daemon=True)
        worker.start()
        self.wait_for_file(self.gate_path("deadline", "started"))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["deadline"][0], 408)
        self.assertTrue(outcome["deadline"][2]["error"])
        self.assertEqual(outcome["deadline"][2]["error"]["code"], "deadline_exceeded")

    def test_stream_disconnect_cancels_before_deadline(self):
        self.config["server"]["wall_time_ms"] = 30000
        self.start()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        connection.request(
            "POST", "/openai/v1/chat/completions",
            json.dumps({"model": "m", "messages": [{"role": "user", "content": "[[gate:disconnect]] held"}],
                        "max_completion_tokens": 1, "stream": True}),
            {"Authorization": "Bearer source-fixture-token", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.wait_for_file(self.gate_path("disconnect", "started"))
        connection.close()
        self.wait_for_file(self.gate_path("disconnect", "cancelled"))
        self.assertFalse(self.gate_path("disconnect", "release").exists())
    def test_active_and_request_deadline_bounds_gated_work(self):
        self.config["server"]["wall_time_ms"] = 5000
        self.config["server"]["active_time_ms"] = 400
        self.start()
        import threading
        outcomes = {}

        def run(name, deadline=None):
            body = {"model": "m", "prompt": f"[[gate:{name}]] held", "max_tokens": 1}
            if deadline is not None:
                body["deadline_ms"] = deadline
            outcomes[name] = self.json_request("POST", "/openai/v1/completions", body)

        request_deadline = threading.Thread(target=run, args=("request-deadline", 100), daemon=True)
        request_deadline.start()
        self.wait_for_file(self.gate_path("request-deadline", "started"))
        request_deadline.join(5)
        self.assertFalse(request_deadline.is_alive())
        self.assertEqual(outcomes["request-deadline"][0], 408)

        active_deadline = threading.Thread(target=run, args=("active-deadline",), daemon=True)
        active_deadline.start()
        self.wait_for_file(self.gate_path("active-deadline", "started"))
        active_deadline.join(5)
        self.assertFalse(active_deadline.is_alive())
        self.assertEqual(outcomes["active-deadline"][0], 408)


    def test_auth_error_mapping_headers_and_management_routes(self):
        self.start()
        unauth_status, _, unauth_body = self.request(
            "GET", "/openai/v1/models", authenticated=False
        )
        self.assertEqual(unauth_status, 401)
        self.assertEqual(json.loads(unauth_body)["error"]["code"], "unauthorized")
        status, headers, models = self.json_request("GET", "/openai/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(models["data"][0]["id"], "m")
        status, _, missing = self.json_request(
            "POST", "/openai/v1/completions", {"model": "absent", "prompt": "x"}
        )
        self.assertEqual(status, 404)
        self.assertEqual(missing["error"]["code"], "model_not_found")
        status, _, context = self.json_request("POST", "/cusco/v1/contexts")
        self.assertEqual(status, 200)
        self.assertTrue(context["id"])
        status, _, fetched = self.json_request("GET", "/cusco/v1/contexts/" + context["id"])
        self.assertEqual(status, 200)
        self.assertEqual(fetched["id"], context["id"])
        status, _, listed = self.json_request("GET", "/cusco/v1/contexts")
        self.assertEqual(status, 200)
        self.assertTrue(any(item["id"] == context["id"] for item in listed))
        status, _, imported = self.json_request(
            "POST", "/cusco/v1/contexts/import", {"tokens": ["imported-token"]}
        )
        self.assertEqual(status, 200)
        self.assertTrue(imported["id"])
        status, _, branch = self.json_request(
            "POST", "/cusco/v1/contexts/" + context["id"] + "/branches"
        )
        self.assertEqual(status, 200)
        self.assertNotEqual(branch["id"], context["id"])
        self.assertEqual(self.request("DELETE", "/cusco/v1/contexts/" + branch["id"])[0], 204)
        self.assertEqual(self.request("DELETE", "/cusco/v1/contexts/" + imported["id"])[0], 204)

    def test_disabled_hosted_search_is_rejected_before_selection(self):
        del self.config["hosted_tools"]
        self.start()
        from openai import BadRequestError
        with self.assertRaises(BadRequestError) as caught:
            self.openai().responses.create(
                model="m", input="[[gate:disabled-search]] find evidence",
                tools=[{"type": "web_search"}], tool_choice="required",
            )
        self.assertEqual(caught.exception.code, "invalid_request")
        self.assertIn("disabled", caught.exception.message)
        self.assertFalse(self.gate_path("disabled-search", "started").exists())
        _, _, status = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(status["admission"]["active"], 0)
        self.assertEqual(status["admission"]["queued"], 0)

    def test_hosted_selection_has_its_own_active_deadline(self):
        self.config["server"]["active_time_ms"] = 100
        self.config["server"]["wall_time_ms"] = 10000
        self.start()
        from openai import APIStatusError
        with self.assertRaises(APIStatusError) as caught:
            self.openai().with_options(timeout=3).responses.create(
                model="m", input="[[gate:selection-active]] find evidence",
                tools=[{"type": "web_search"}], tool_choice="required",
            )
        self.assertEqual(caught.exception.status_code, 408)
        self.assertEqual(caught.exception.code, "deadline_exceeded")
        self.assertTrue(self.gate_path("selection-active", "started").exists())
        self.assertTrue(self.gate_path("selection-active", "cancelled").exists())
        self.assertFalse(self.gate_path("selection-active", "release").exists())

    def test_unknown_fields_name_the_rejected_field(self):
        self.start()
        from openai import BadRequestError
        client = self.openai()
        for name in ("prompt_cache_key", "parallel_tool_calls"):
            with self.subTest(field=name), self.assertRaises(BadRequestError) as caught:
                client.responses.create(
                    model="m", input="hello", extra_body={name: True},
                )
            self.assertEqual(caught.exception.code, "invalid_request")
            self.assertIn("unknown field " + name, caught.exception.message)
        status, _, payload = self.request("POST", "/openai/v1/responses", [])
        failure = json.loads(payload)
        self.assertEqual(status, 400)
        self.assertEqual(failure["error"]["code"], "invalid_request")
        self.assertIn("expected an object", failure["error"]["message"])

    def test_completion_chat_defaults_validation_and_service_tier(self):
        self.start()
        client = self.openai()
        completion_raw = client.completions.with_raw_response.create(
            model="m", prompt="alpha beta", max_tokens=1,
        )
        completion = completion_raw.parse()
        self.assertEqual(completion.object, "text_completion")
        self.assertEqual(completion.choices[0].text.strip(), "beta")
        self.assertTrue(completion_raw.headers.get("x-request-id"))
        chat = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "alpha beta"}],
            max_completion_tokens=1, service_tier="priority",
        )
        self.assertEqual(chat.choices[0].message.role, "assistant")
        self.assertEqual(chat.service_tier, "default")
        for invalid in (
            {"model": "m", "prompt": "x", "unknown": True},
            {"model": "m", "prompt": "x", "max_tokens": -1},
        ):
            status, _, failure = self.json_request("POST", "/openai/v1/completions", invalid)
            self.assertEqual(status, 400)
            self.assertIn("error", failure)
        status, _, failure = self.json_request(
            "POST", "/openai/v1/chat/completions",
            {"model": "m", "messages": [], "max_tokens": 1, "max_completion_tokens": 2},
        )
        self.assertEqual(status, 400)
        self.assertEqual(failure["error"]["code"], "invalid_request")

    def test_chat_text_fragments_join_without_separators(self):
        self.start()
        status, _, chat = self.json_request(
            "POST", "/openai/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "café Hosted web "},
                {"type": "text", "text": ""},
                {"type": "text", "text": "search completed"},
            ]}], "max_completion_tokens": 16},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            chat["choices"][0]["message"]["content"],
            "Answer https://example.com/result",
        )

    def test_chat_sampling_ranges_are_rejected(self):
        self.start()
        for options in ({"temperature": -0.1}, {"temperature": 2.1}, {"top_p": 0}, {"top_p": 1.1}):
            with self.subTest(options=options):
                status, _, error = self.json_request(
                    "POST", "/openai/v1/chat/completions",
                    {"model": "m", "messages": [{"role": "user", "content": "hello"}], **options},
                )
                self.assertEqual(status, 400)
                self.assertIn("error", error)

    def test_chat_tools_are_validation_only_and_keep_text_output(self):
        self.start()
        chat = self.openai().chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "fixture:inspect-chat-tool-prompt"}],
            tools=[{"type": "function", "function": {
                "name": "lookup", "description": "Lookup fixture data",
                "parameters": {"type": "object", "properties": {}},
            }}],
            max_completion_tokens=8,
        )
        message = chat.choices[0].message
        self.assertIsNone(message.tool_calls)
        self.assertIn("fixture:inspect-chat-tool-prompt", message.content)
        self.assertNotIn("lookup", message.content)
        status, _, failure = self.json_request(
            "POST", "/openai/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hello"}],
             "tools": [{"type": "web_search"}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", failure)


    def test_function_parameter_schemas_are_validated_for_chat_and_responses(self):
        self.start()
        invalid_schema = {
            "type": "object",
            "properties": {"value": {"type": "not-a-json-schema-type"}},
        }
        valid_schema = {
            "type": "object",
            "properties": {"value": {"type": ["string", "null"]}},
        }
        for path, body in (
            ("/openai/v1/responses", {"model": "m", "input": "Call broken."}),
            ("/openai/v1/chat/completions", {
                "model": "m", "messages": [{"role": "user", "content": "Call broken."}],
            }),
        ):
            with self.subTest(path=path):
                invalid = dict(body)
                invalid["tools"] = [{
                    "type": "function", "name": "broken", "strict": True,
                    "parameters": invalid_schema,
                }]
                status, _, failure = self.json_request("POST", path, invalid)
                self.assertEqual(status, 400, failure)
                self.assertEqual(failure["error"]["code"], "invalid_request")

                accepted = dict(body)
                accepted["tools"] = [{
                    "type": "function", "name": "union", "strict": True,
                    "parameters": valid_schema,
                }]
                status, _, result = self.json_request("POST", path, accepted)
                self.assertEqual(status, 200, result)

    def test_openai_stream_event_order_usage_and_terminal_done(self):
        self.start()
        client = self.openai()
        chunks = list(client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "one two three"}],
            max_completion_tokens=2, stream=True,
            stream_options={"include_usage": True},
        ))
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")
        self.assertGreater(chunks[-1].usage.total_tokens, 0)
        completion_chunks = list(client.completions.create(
            model="m", prompt="plain stream", max_tokens=1, stream=True,
        ))
        self.assertTrue(completion_chunks)
        self.assertEqual(completion_chunks[-1].choices[0].finish_reason, "length")
        self.assertTrue(all("usage" not in chunk.model_fields_set for chunk in completion_chunks))
        self.assertTrue(all(chunk.usage is None for chunk in completion_chunks))
        chat_chunks = list(client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "plain chat stream"}],
            max_completion_tokens=1, stream=True,
        ))
        self.assertTrue(chat_chunks)
        self.assertTrue(all("usage" not in chunk.model_fields_set for chunk in chat_chunks))
        self.assertTrue(all(chunk.usage is None for chunk in chat_chunks))

    def test_responses_token_exhaustion_is_incomplete_and_survives_retrieval(self):
        self.start()
        client = self.openai()
        response = client.responses.create(
            model="m", input="a long story", max_output_tokens=1,
        )
        self.assertEqual(response.status, "incomplete")
        self.assertEqual(response.incomplete_details.reason, "max_output_tokens")
        stored = client.responses.retrieve(response.id)
        self.assertEqual(stored.status, "incomplete")
        self.assertEqual(stored.incomplete_details.reason, "max_output_tokens")
        events = list(client.responses.create(
            model="m", input="another long story", max_output_tokens=1, stream=True,
        ))
        self.assertEqual(events[-1].type, "response.incomplete")
        self.assertEqual(events[-1].response.status, "incomplete")
        self.assertEqual(events[-1].response.incomplete_details.reason, "max_output_tokens")
        self.assertNotIn("response.completed", [event.type for event in events])
        with client.responses.stream(
            model="m", input="short answer", max_output_tokens=16,
        ) as stream:
            completed_events = []
            for event in stream:
                if event.type == "response.created":
                    self.assertEqual(event.response.output, [])
                completed_events.append(event)
            completed = stream.get_final_response()
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(completed.output), 1)
        self.assertEqual(len(completed.output[0].content), 1)
        self.assertEqual(
            completed.output_text,
            "".join(event.delta for event in completed_events if event.type == "response.output_text.delta"),
        )

    def test_responses_function_stream_accumulates_arguments(self):
        self.start()
        client = self.openai()
        with client.responses.stream(
            model="m", input="fixture:stream-function", max_output_tokens=64,
            tools=[{
                "type": "function", "name": "lookup",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
            }],
            tool_choice={"type": "function", "name": "lookup"},
        ) as stream:
            events = []
            for event in stream:
                if event.type == "response.output_item.added":
                    self.assertEqual(event.item.arguments, "")
                    self.assertEqual(event.item.status, "in_progress")
                events.append(event)
            final = stream.get_final_response()
        added = next(event for event in events if event.type == "response.output_item.added")
        delta = next(event for event in events if event.type == "response.function_call_arguments.delta")
        done = next(event for event in events if event.type == "response.function_call_arguments.done")
        self.assertEqual(delta.item_id, added.item.id)
        self.assertEqual(done.item_id, added.item.id)
        self.assertEqual(done.name, "lookup")
        self.assertEqual(json.loads(delta.delta), {"key": "value"})
        self.assertEqual(done.arguments, final.output[0].arguments)
        self.assertEqual(json.loads(final.output[0].arguments), {"key": "value"})
        self.assertEqual(events[-1].type, "response.completed")

    def test_responses_schema_store_lineage_and_context_update(self):
        self.start()
        status, _, failure = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "hello", "tool_choice": "required", "tools": []},
        )
        self.assertEqual(status, 400)
        self.assertEqual(failure["error"]["code"], "invalid_request")
        status, headers, base = self.json_request(
            "POST", "/openai/v1/responses", {"model": "m", "input": "hello", "max_output_tokens": 1}
        )
        self.assertEqual(status, 200)
        self.assertTrue(base["id"])
        ratio = base["cusco"]["usage"]["prefill"]["cached_ratio"]
        self.assertIsInstance(ratio, (int, float))
        self.assertEqual(ratio, 0.0)
        response_id = base["id"]
        self.assertEqual(self.json_request("GET", f"/openai/v1/responses/{response_id}")[0], 200)
        empty_base_status, _, empty_base = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "empty branch", "max_output_tokens": 1},
        )
        self.assertEqual(empty_base_status, 200)
        empty_continuation_status, _, empty_continuation = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": [], "max_output_tokens": 1,
             "previous_response_id": empty_base["id"]},
        )
        self.assertEqual(empty_continuation_status, 200)
        self.assertEqual(empty_continuation["previous_response_id"], empty_base["id"])
        status, _, continued = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "continue", "max_output_tokens": 1,
             "previous_response_id": response_id},
        )
        self.assertEqual(status, 200)
        self.assertEqual(continued["previous_response_id"], response_id)
        competing_status, _, competing = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "competing", "max_output_tokens": 1,
             "previous_response_id": response_id},
        )
        self.assertEqual(competing_status, 409)
        self.assertIn("error", competing)
        transient_status, _, transient = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "transient", "store": False, "max_output_tokens": 1},
        )
        self.assertEqual(transient_status, 200)
        transient_id = transient["id"]
        transient_get_status, _, retrieved_transient = self.json_request("GET", f"/openai/v1/responses/{transient_id}")
        self.assertEqual(transient_get_status, 200)
        self.assertEqual(retrieved_transient["id"], transient_id)
        self.assertFalse(retrieved_transient["store"])
        transient_next_status, _, transient_next = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "continue", "previous_response_id": transient_id,
             "max_output_tokens": 1},
        )
        self.assertEqual(transient_next_status, 400)
        self.assertEqual(transient_next["error"]["code"], "invalid_request")
        self.assertIn("requires a stored response", transient_next["error"]["message"])
        fold_base_status, _, fold_base = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "fold base", "max_output_tokens": 1},
        )
        self.assertEqual(fold_base_status, 200)
        update_request = {
            "base_response_id": fold_base["id"], "expected_revision": 0,
            "operation_id": "source-update-1",
            "operation": {"type": "fold", "items": [{"type": "message", "role": "user", "text": "summary"}]},
        }
        status, update_headers, update = self.json_request(
            "POST", "/cusco/v1/context-updates", update_request,
        )
        self.assertEqual(status, 200)
        self.assertEqual(update["revision"], 1)
        self.assertEqual(update_headers.get("x-request-id"), "source-update-1")
        self.assertEqual(update["response"]["cusco"]["context_update"]["operation"], "fold")
        replay_status, _, replay = self.json_request(
            "POST", "/cusco/v1/context-updates", update_request,
        )
        self.assertEqual(replay_status, 409)
        self.assertIn("error", replay)
        self.assertEqual(self.json_request("GET", "/cusco/v1/capabilities")[0], 200)
        self.assertEqual(self.json_request("DELETE", f"/openai/v1/responses/{response_id}")[0], 409)
        self.assertEqual(self.json_request("DELETE", f"/openai/v1/responses/{continued['id']}")[0], 200)
        self.assertEqual(self.json_request("DELETE", f"/openai/v1/responses/{response_id}")[0], 200)
        self.assertEqual(self.json_request("GET", f"/openai/v1/responses/{response_id}")[0], 404)

    def test_responses_portable_function_call_and_tool_output_lineage(self):
        self.start()
        status, _, result = self.json_request(
            "POST", "/openai/v1/responses",
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "Look up the key"},
                    {"type": "function_call", "call_id": "call_fixture_1", "name": "lookup",
                     "arguments": "{\"key\":\"value\"}"},
                    {"type": "function_call_output", "call_id": "call_fixture_1", "output": "result"},
                ],
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                "tool_choice": "auto", "max_output_tokens": 1, "store": True,
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(result["id"])
        self.assertEqual(result["object"], "response")
        response_id = result["id"]
        self.assertEqual(self.json_request("GET", f"/openai/v1/responses/{response_id}")[0], 200)
        self.assertEqual(self.json_request("DELETE", f"/openai/v1/responses/{response_id}")[0], 200)
        self.assertEqual(self.json_request("GET", f"/openai/v1/responses/{response_id}")[0], 404)

    def test_hosted_search_results_and_citations_survive_retrieval(self):
        self.start()
        status, _, response = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "Explain Rust ownership", "max_output_tokens": 1,
             "tool_choice": "required", "tools": [{"type": "web_search"}]},
        )
        self.assertEqual(status, 200, response)
        search, message = response["output"]
        self.assertEqual(search["type"], "web_search_call")
        self.assertEqual(search["status"], "completed")
        self.assertEqual(search["action"]["query"], "rust ownership")
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["content"][0]["annotations"][0]["url"], "https://example.com/result")
        status, _, stored = self.json_request("GET", f"/openai/v1/responses/{response['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(stored["output"], response["output"])

    def test_hosted_selection_obeys_parent_deadline(self):
        self.config["server"]["wall_time_ms"] = 400
        self.start()
        status, _, failure = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "[[gate:search-deadline]] held", "max_output_tokens": 1,
             "tool_choice": "required", "tools": [{"type": "web_search"}]},
        )
        self.assertEqual(status, 408, failure)
        self.assertEqual(failure["error"]["code"], "deadline_exceeded")
        self.wait_for_file(self.gate_path("search-deadline", "cancelled"))
        _, _, metrics = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(metrics["admission"]["active"], 0)

    def test_disconnect_cancels_hosted_selection_before_parent_deadline(self):
        self.config["server"]["wall_time_ms"] = 30000
        self.start()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        self.addCleanup(connection.close)
        connection.request(
            "POST", "/openai/v1/responses",
            json.dumps({"model": "m", "input": "[[gate:search-disconnect]] held",
                        "max_output_tokens": 1, "tool_choice": "required",
                        "tools": [{"type": "web_search"}]}),
            {"Authorization": "Bearer source-fixture-token", "Content-Type": "application/json"},
        )
        self.wait_for_file(self.gate_path("search-disconnect", "started"))
        connection.close()
        self.wait_for_file(self.gate_path("search-disconnect", "cancelled"))
        self.assertFalse(self.gate_path("search-disconnect", "release").exists())

    def test_responses_tool_and_structured_wire_validation(self):
        self.start()
        status, _, structured = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": [{"role": "user", "content": "hello"}],
             "max_output_tokens": 1, "text": {"format": {"type": "json_schema",
             "name": "answer", "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(structured["object"], "response")
        self.assertEqual(json.loads(structured["output"][0]["content"][0]["text"]), {})
        status, _, failure = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "hello", "tool_choice": {"type": "function", "name": "missing"},
             "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]},
        )
        self.assertEqual(status, 400)
        self.assertIn("undeclared function", failure["error"]["message"])
        for effort in ("low", "medium", "high", "max"):
            with self.subTest(chat_reasoning_effort=effort):
                status, _, failure = self.json_request(
                    "POST", "/openai/v1/chat/completions",
                    {"model": "m", "messages": [{"role": "user", "content": "hello"}],
                     "reasoning_effort": effort},
                )
                self.assertEqual(status, 400)
                self.assertEqual(failure["error"]["code"], "invalid_request")
                self.assertIn("does not advertise reasoning levels", failure["error"]["message"])
        status, _, maximum_effort = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "hello", "max_output_tokens": 1, "reasoning_effort": "max"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(maximum_effort["object"], "response")
        status, _, failure = self.json_request(
            "POST", "/openai/v1/responses",
            {"model": "m", "input": "hello",
             "response_format": {"type": "json_object"},
             "text": {"format": {"type": "json_object"}}},
        )
        self.assertEqual(status, 400)
        self.assertEqual(failure["error"]["code"], "invalid_request")
        self.assertIn("cannot both be supplied", failure["error"]["message"])

    def test_ollama_models_residency_methods_and_ndjson_error(self):
        self.start()
        status, _, version = self.json_request("GET", "/cusco/v1/api/version")
        self.assertEqual(status, 200)
        self.assertTrue(version["version"])
        status, _, tags = self.json_request("GET", "/cusco/v1/api/tags")
        self.assertEqual(status, 200)
        self.assertTrue(any(model["id"].startswith("m") for model in tags["models"]))
        status, _, shown = self.json_request("POST", "/cusco/v1/api/show", {"name": "m"})
        self.assertEqual(status, 200)
        self.assertEqual(shown["id"], "m")
        status, _, copied = self.json_request(
            "POST", "/cusco/v1/api/copy", {"source": "m", "destination": "m-copy"}
        )
        self.assertEqual(status, 200)
        status, _, copied_model = self.json_request(
            "POST", "/cusco/v1/api/show", {"model": "m-copy"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(copied_model["id"], "m")
        status, _, residency = self.json_request("GET", "/cusco/v1/api/ps")
        self.assertEqual(status, 200)
        self.assertIsInstance(residency["models"], list)
        status, headers, body = self.request(
            "POST", "/cusco/v1/api/pull", {"name": "not-a-model", "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertIn("ndjson", headers["content-type"])
        lines = [json.loads(line) for line in body.splitlines()]
        self.assertTrue(lines)
        self.assertIn("error", lines[-1])
        status, _, failure = self.json_request(
            "POST", "/cusco/v1/api/show", {"model": "m", "name": "different"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", failure)
        self.assertEqual(self.request("DELETE", "/cusco/v1/api/delete", {"model": "m"})[0], 200)


    def test_request_header_limit_maps_to_431_before_admission(self):
        self.config["server"]["header_bytes"] = 1024
        self.start()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            connection.request(
                "GET", "/openai/v1/models",
                headers={"Authorization": "Bearer source-fixture-token", "X-Oversized": "x" * 2048},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 431)
        finally:
            connection.close()
        status, _, metrics = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(metrics["admission"]["active"], 0)
        self.assertEqual(metrics["admission"]["queued"], 0)

    def test_transport_body_bound_and_response_header_contracts(self):
        self.start()
        oversized = json.dumps({"model": "m", "prompt": "x" * 70000}).encode()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            connection.request("POST", "/openai/v1/completions", oversized,
                              {"Authorization": "Bearer source-fixture-token",
                               "Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read()
            self.assertEqual(response.status, 413)
            self.assertTrue(payload)
        finally:
            connection.close()
        _, _, metrics = self.json_request("GET", "/cusco/v1/status")
        self.assertEqual(metrics["admission"]["active"], 0)
        self.assertEqual(metrics["admission"]["queued"], 0)
        status, headers, spec = self.json_request("GET", "/openai/v1/openapi.json")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertIn("/openai/v1/responses", spec["paths"])
        self.assertIn("/cusco/v1/context-updates", spec["paths"])
        self.assertEqual(self.request("GET", "/openapi/ui")[0], 404)
        self.assertEqual(self.request("GET", "/not-found")[0], 404)

    def test_interactive_docs_route_is_config_gated(self):
        self.config["openapi"]["docs_ui"]["enabled"] = True
        self.start()
        status, headers, payload = self.request("GET", "/openapi/ui")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"SwaggerUIBundle", payload)


if __name__ == "__main__":
    unittest.main()
