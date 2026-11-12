#!/usr/bin/env python3
"""Run model-free boundary checks against an already-built real Cusco service.

Pass the Terrane build artifact as the first argument. Native executor libraries
must be available to the process (LD_LIBRARY_PATH on Linux). No model is loaded.
"""
import http.client
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1)).resolve()


class HybridServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cusco-hybrid-")
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
            "execution": {
                "device_capacity": "64 MiB", "host_capacity": "64 MiB",
                "storage_capacity": "128 MiB", "context_reserve": "1 MiB",
                "context_tokens": 128, "gpu_layers": 0,
            },
            # Match the acceptance config's explicit limits: omitting a nested
            # object does not apply its per-field Serde defaults.
            "hosted_tools": {"web_search": {
                "enabled": False, "endpoint": None, "timeout_ms": 10000,
                "max_results": 8, "max_response_bytes": 1048576,
                "allowed_domains": [],
            }},
            "server": {
                "active_requests": 4, "queue_count": 32, "queue_bytes": 16777216,
                "request_bytes": 1048576, "pre_queue_concurrency": 16,
                "default_output_tokens": 1024, "header_bytes": 32768,
                "body_timeout_ms": 10000, "wall_time_ms": 300000,
                "active_time_ms": 240000, "stream_buffer": 8,
                "shutdown_grace_ms": 30000, "compaction_enabled": True,
                "compaction_workers": 1, "compaction_declarations": 128,
                "compaction_declaration_rate_per_minute": 32,
                "compaction_declaration_lifetime_ms": 3600000,
            },
            "scheduler": {
                "version": 1, "interactive_weight": 4, "standard_weight": 2,
                "batch_weight": 1, "deficit_refill": 1, "prefill_tokens": 32,
                "promotion_rounds": 64, "diagnostics_enabled": False,
                "diagnostic_capacity": 1024,
            },
            "vision": {
                "max_images": 8, "max_encoded_bytes": 16777216,
                "max_decoded_bytes": 12582912, "max_dimension": 8192,
                "max_total_pixels": 32000000, "retention_capacity": "256 MiB",
                "metadata_allowlist": [],
            },
        }
        self.path = self.root / "config.yaml"
        self.environment = os.environ.copy()
        self.environment.pop("CUSCO_HTTP_DEBUG", None)
        self.environment["CUSCO_BEARER_TOKEN"] = "hybrid-test-token"

    def write_config(self):
        self.path.write_text(json.dumps(self.config))

    def launch(self, *arguments):
        self.write_config()
        self.errors = (self.root / "stderr").open("w+")
        self.addCleanup(self.errors.close)
        process = subprocess.Popen([str(BINARY), "--config", str(self.path), *arguments], env=self.environment,
                                   stdout=subprocess.PIPE, stderr=self.errors, text=True)
        self.addCleanup(self.stop, process)
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(ready.select(15), "service did not report its bound address")
        line = process.stdout.readline().strip()
        if not line.startswith("Cusco listening on "):
            self.errors.seek(0)
            self.fail(f"service did not start: {line}\n{self.errors.read()}")
        self.host, port = line.removeprefix("Cusco listening on ").rsplit(":", 1)
        self.port = int(port)
        self.assertGreater(self.port, 0)
        return process

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()

    def request(self, method, path, body=None, authenticated=False):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        headers = {}
        if authenticated:
            headers["Authorization"] = "Bearer hybrid-test-token"
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_terrane_frontend_and_authenticated_rust_api_share_listener(self):
        process = self.launch()
        status, headers, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"<title>Cusco</title>", page)
        self.assertIn(b"/openai/v1/openapi.json", page)
        self.assertEqual(self.request("HEAD", "/")[2], b"")
        self.assertEqual(self.request("GET", "/health")[::2], (200, b"ready\n"))
        self.assertEqual(self.request("HEAD", "/health")[2], b"")
        self.assertEqual(self.request("GET", "/cusco/v1/status")[0], 401)
        status, _, payload = self.request("GET", "/openai/v1/models", authenticated=True)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload), {"object": "list", "data": []})
        status, _, payload = self.request("POST", "/cusco/v1/contexts", authenticated=True)
        self.assertEqual(status, 200)
        context = json.loads(payload)
        status, _, payload = self.request("GET", "/cusco/v1/contexts/" + context["id"], authenticated=True)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["id"], context["id"])
        self.assertEqual(self.request("GET", "/cusco/v1/capabilities", authenticated=True)[0], 200)
        self.assertEqual(self.request("GET", "/openai/v1/openapi.json")[0], 200)
        self.assertEqual(self.request("GET", "/does-not-exist")[0], 404)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertTrue((self.root / "db/catalog.sqlite").is_file())

    def test_explicit_debug_override_wins_over_invalid_environment(self):
        self.environment["CUSCO_HTTP_DEBUG"] = "not-a-level"
        self.launch("--http-debug", "off")
        self.assertEqual(self.request("GET", "/health")[0], 200)

    def test_public_anonymous_binding_is_rejected_before_startup(self):
        self.environment.pop("CUSCO_BEARER_TOKEN")
        self.config["listen"] = "0.0.0.0:0"
        self.write_config()
        result = subprocess.run([str(BINARY), "--config", str(self.path)], env=self.environment,
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe unauthenticated listener", result.stderr)
        self.assertNotIn("Cusco listening", result.stdout)


if __name__ == "__main__":
    unittest.main()
