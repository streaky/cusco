import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("acceptance_report", ROOT / "tools/acceptance-report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class AcceptanceReportTests(unittest.TestCase):
    def invoke(self, listed, executed="test result: ok. 1 passed; 0 failed; 0 ignored;"):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=listed if command[-1] == "--list" else executed, stderr="")

        with patch.object(REPORT.subprocess, "run", side_effect=run), patch.object(
            REPORT, "write_report", side_effect=lambda mode, manifest, gates: gates
        ):
            rows = REPORT.model_free({"model_free": [{"crate": "cusco-server", "test": "gone", "contract": "x"}]})
        return rows, calls

    def test_deleted_rust_filter_is_a_failure_and_does_not_execute(self):
        rows, calls = self.invoke("\n")
        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["matched_tests"], 0)
        self.assertEqual(rows[0]["error"], "Rust test filter matched zero tests")
        self.assertEqual(len(calls), 1)

    def test_existing_rust_filter_executes_after_listing(self):
        rows, calls = self.invoke("present_test: test\n")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["matched_tests"], 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--list", calls[1])

    def test_ignored_only_rust_filter_is_not_a_passing_gate(self):
        rows, _ = self.invoke("ignored_test: test\n", "test result: ok. 0 passed; 0 failed; 1 ignored;")
        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["executed_tests"], 0)
        self.assertEqual(rows[0]["error"], "Rust test filter executed zero tests")

    def test_execution_totals_include_all_binaries_and_doctests(self):
        rows, _ = self.invoke(
            "unit_case: test\nintegration_case: test\ndoc_case: test\n",
            "running 0 tests\n"
            "test result: ok. 0 passed; 0 failed; 2 ignored;\n"
            "test result: ok. 2 passed; 0 failed; 1 ignored;\n"
            "   Doc-tests example\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored;\n",
        )
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(rows[0]["matched_tests"], 3)
        self.assertEqual(rows[0]["executed_tests"], 3)

    def test_failed_results_still_count_executed_tests_across_binaries(self):
        listed = "unit_case: test\nintegration_case: test\n"
        output = (
            "test result: ok. 1 passed; 0 failed; 0 ignored;\n"
            "test result: FAILED. 2 passed; 1 failed; 3 ignored;\n"
        )

        def run(command, **kwargs):
            listing = command[-1] == "--list"
            return subprocess.CompletedProcess(
                command, 0 if listing else 101,
                stdout=listed if listing else output, stderr="",
            )

        with patch.object(REPORT.subprocess, "run", side_effect=run), patch.object(
            REPORT, "write_report", side_effect=lambda mode, manifest, gates: gates
        ):
            rows = REPORT.model_free({"model_free": [
                {"crate": "example", "test": "case", "contract": "aggregate"}
            ]})
        self.assertFalse(rows[0]["passed"])
        self.assertEqual(rows[0]["executed_tests"], 4)

    def test_protocol_suite_is_invoked_as_an_executable_gate(self):
        calls = []
        command_result = subprocess.CompletedProcess(
            ["python3", "server/tests/test_protocol_service.py"], 0, stdout="", stderr=""
        )
        with patch.object(REPORT.subprocess, "run", side_effect=lambda command, **kwargs: calls.append(command) or command_result), patch.object(
            REPORT, "write_report", side_effect=lambda mode, manifest, gates: gates
        ):
            rows = REPORT.model_free({"model_free": [
                {"protocol_suite": "server/tests/test_protocol_service.py", "contract": "protocol"}
            ]})
        self.assertEqual(calls, [["python3", "server/tests/test_protocol_service.py"]])
        self.assertTrue(rows[0]["passed"])


if __name__ == "__main__":
    unittest.main()
