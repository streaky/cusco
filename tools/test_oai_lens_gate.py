import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "tools/oai-lens-gate.py"
REVISION = "1d145321ded86a1cfc7c2852667854957c57c5fb"
PROBES = {
    "chat.core.messages": "pass",
    "responses.core.simple_text": "fail",
}


class OaiLensGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.report = self.directory / "report.json"
        self.gate_report = self.directory / "gate.json"
        self.expectations = self.directory / "expectations.json"
        self.revision = self.directory / "revision.txt"
        self.expectations.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "runner_revision": REVISION,
                    "profile": "openai_api",
                    "probes": PROBES,
                }
            ),
            encoding="utf-8",
        )
        self.revision.write_text(f"{REVISION}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, runner: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(WRAPPER),
                "--report",
                str(self.report),
                "--gate-report",
                str(self.gate_report),
                "--expectations",
                str(self.expectations),
                "--revision-file",
                str(self.revision),
                "--",
                *runner,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_probe_failures_are_advisory_when_report_is_valid(self) -> None:
        report = {
            "version": 1,
            "profile": "openai_api",
            "results": [
                {"name": name, "status": status} for name, status in PROBES.items()
            ],
        }
        producer = (
            "import json,sys;"
            f"json.dump({report!r}, open(sys.argv[1], 'w'));"
            "raise SystemExit(1)"
        )

        completed = self.invoke([sys.executable, "-c", producer, str(self.report)])

        self.assertEqual(completed.returncode, 0, completed.stderr)
        gate = json.loads(self.gate_report.read_text(encoding="utf-8"))
        self.assertEqual(gate["classification"], "runner_completed_with_probe_failures")
        self.assertEqual(gate["runner_exit_code"], 1)
        self.assertEqual(gate["deltas"]["unchanged_failing"], ["responses.core.simple_text"])
        self.assertEqual(gate["deltas"]["newly_failing"], [])
        self.assertIn("upstream_report_sha256", gate)

    def test_missing_report_is_a_harness_failure(self) -> None:
        completed = self.invoke([sys.executable, "-c", "raise SystemExit(1)"])

        self.assertEqual(completed.returncode, 2)
        gate = json.loads(self.gate_report.read_text(encoding="utf-8"))
        self.assertEqual(gate["classification"], "harness_failed")
        self.assertEqual(gate["runner_exit_code"], 1)
        self.assertIn("did not produce its report", gate["error"])


if __name__ == "__main__":
    unittest.main()
