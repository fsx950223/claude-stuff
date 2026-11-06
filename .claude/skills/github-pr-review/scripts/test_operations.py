"""Offline configuration, installation and CLI integration tests."""

import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import install as installer
import review_prs as runner


def config(**changes):
    return {
        "github_login": "reviewer",
        "review_requested": True,
        "assigned": True,
        "execution_authorized": True,
        "publication_authorized": True,
        "publish_reviews": True,
        "review_backend": "codex",
        "daily_time": "09:00",
        "timezone": "Asia/Shanghai",
        "pr_timeout_seconds": 30,
        "test_environments": {},
        **changes,
    }


class ConfigurationTests(unittest.TestCase):
    def test_publication_requires_recorded_authorization(self):
        with self.assertRaisesRegex(ValueError, "publication_authorized"):
            runner.validate_config(config(publication_authorized=False))

    def test_invalid_verdict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid review verdict"):
            runner.choose_event({"verdict": "UNKNOWN", "findings": []})

    def test_draft_mode_does_not_require_publication_authorization(self):
        runner.validate_config(
            config(publish_reviews=False, publication_authorized=False)
        )

    def test_invalid_timeout_and_scope_are_rejected(self):
        for changes in [
            {"pr_timeout_seconds": 0},
            {"pr_timeout_seconds": True},
            {"review_requested": False, "assigned": False},
            {"repositories": "ROCm/aiter"},
            {"repositories": ["ROCm/aiter", "../unexpected"]},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_config(config(**changes))

    def test_timezone_and_local_time_preserved(self):
        self.assertEqual(installer.calendar(config()), "*-*-* 09:00:00 Asia/Shanghai")
        with self.assertRaises(ValueError):
            installer.calendar(config(daily_time="24:00"))

    def test_backend_commands_preserve_model_and_permission_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            codex = runner.agent_command(Path(directory), config())
            claude = runner.agent_command(
                Path(directory), config(review_backend="claude")
            )
        self.assertNotIn("--model", codex + claude)
        self.assertNotIn("--dangerously-skip-permissions", claude)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", codex)
        self.assertIn("--sandbox=workspace-write", codex)
        self.assertIn("dontAsk", claude)

    def test_claude_error_or_unstructured_response_is_rejected(self):
        for envelope in [
            {"is_error": True, "structured_output": {}},
            {"result": "{}"},
            [],
        ]:
            with self.subTest(envelope=envelope), self.assertRaises(ValueError):
                runner.claude_result(envelope)

    def test_preparation_does_not_activate_or_contact_remote_tools(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            installer, "run"
        ) as remote:
            target = Path(directory) / "runtime"
            result = installer.prepare(target, config())
            remote.assert_not_called()
            self.assertFalse(result["activated"])
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertEqual((target / "config.json").stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "Runtime already exists"):
                installer.prepare(target, config())

    def test_existing_timer_blocks_activation_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            units = Path(directory) / "units"
            units.mkdir()
            existing = units / "github-pr-review.timer"
            existing.write_text("existing timer")
            target = Path(directory) / "runtime"
            with patch.object(installer, "verify_activation") as verify:
                with self.assertRaisesRegex(
                    ValueError, "Destination units already exist"
                ):
                    installer.prepare(
                        target, config(), activate=True, unit_directory=units
                    )
                verify.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(existing.read_text(), "existing timer")

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_rendered_user_units_accept_spaces_and_percent_in_runtime_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "review runtime % name"
            record = installer.prepare(target, config())
            result = subprocess.run(
                ["systemd-analyze", "--user", "verify", *record["units"]],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.api_log = self.root / "api.jsonl"
        self.agent_log = self.root / "agent.jsonl"
        self.gh = self.root / "fake-gh"
        self.agent = self.root / "fake-agent"
        self.gh.write_text("""#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(os.environ["FAKE_API_LOG"],"a") as f: f.write(json.dumps(args)+"\\n")
initial={"state":"open","draft":False,"head":{"sha":"a"*40},"base":{"sha":"b"*40},
         "user":{"login":"contributor"},"title":"GPU fix","body":"",
         "html_url":"https://github.com/org/repo/pull/1"}
if args[0]=="search":
 result=[{"number":1,"title":"GPU fix","url":initial["html_url"],
          "repository":{"nameWithOwner":"org/repo"},"isDraft":False,
          "updatedAt":"2026-11-06T00:00:00Z"}]
else:
 endpoint=args[3]
 if endpoint=="user": result={"login":"reviewer"}
 elif "reviews?" in endpoint: result=[[]]
 elif endpoint.endswith("/reviews/42"): result={"state":"APPROVED"}
 elif endpoint.endswith("/reviews") and "POST" in args:
  result={"id":42,"html_url":initial["html_url"]+"#review"}
 else: result=initial
print(json.dumps(result))
""")
        self.agent.write_text("""#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
prompt=sys.stdin.read()
with open(os.environ["FAKE_AGENT_LOG"],"a") as f: f.write(json.dumps(args)+"\\n")
assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in prompt
result={"verdict":"APPROVE","summary":"No blocking defect.",
        "validation_sufficient":True,"findings":[],"tests":[],"limitations":[]}
if os.environ.get("FAKE_INVALID_RESULT"): result={"verdict":"APPROVE"}
if "--output-last-message" in args:
 pathlib.Path(args[args.index("--output-last-message")+1]).write_text(json.dumps(result))
 print(json.dumps({"type":"completed"}))
else:
 print(json.dumps({"is_error":False,"structured_output":result}))
print("fixture diagnostic",file=sys.stderr)
""")
        self.gh.chmod(0o700)
        self.agent.chmod(0o700)
        self.environment = {
            **os.environ,
            "FAKE_API_LOG": str(self.api_log),
            "FAKE_AGENT_LOG": str(self.agent_log),
        }

    def prepare(self, backend="codex", **changes):
        directory = self.root / ("runtime-" + backend)
        installer.prepare(
            directory,
            config(
                gh=str(self.gh),
                codex=str(self.agent),
                claude=str(self.agent),
                review_backend=backend,
                **changes
            ),
        )
        return directory

    def run_cli(self, directory, *args, environment=None):
        return subprocess.run(
            [
                sys.executable,
                str(directory / "review_prs.py"),
                "--directory",
                str(directory),
                *args,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment or self.environment,
            timeout=30,
        )

    def calls(self, path):
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def test_dry_run_does_not_review_publish_or_write_state(self):
        directory = self.prepare(execution_authorized=False)
        result = self.run_cli(directory, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["prs"]), 1)
        self.assertFalse(self.agent_log.exists())
        self.assertFalse((directory / "state.json").exists())
        self.assertFalse(any("POST" in args for args in self.calls(self.api_log)))

    def test_both_cli_backends_publish_exact_commit_once(self):
        for backend in ["codex", "claude"]:
            with self.subTest(backend=backend):
                directory = self.prepare(backend)
                first = self.run_cli(directory)
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                summary = json.loads((directory / "latest.json").read_text())
                task_dir = Path(summary["run"]) / "org--repo--1"
                payload = json.loads((task_dir / "review-payload.json").read_text())
                self.assertEqual(payload["event"], "APPROVE")
                self.assertEqual(payload["commit_id"], "a" * 40)
                self.assertIn(
                    "fixture diagnostic", (task_dir / "agent.stderr.log").read_text()
                )
                before = len(self.calls(self.agent_log))
                second = self.run_cli(directory)
                self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
                self.assertEqual(len(self.calls(self.agent_log)), before)

    def test_invalid_agent_json_cannot_publish(self):
        directory = self.prepare()
        result = self.run_cli(
            directory, environment={**self.environment, "FAKE_INVALID_RESULT": "1"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("POST" in args for args in self.calls(self.api_log)))
        self.assertFalse((directory / "state.json").exists())

    def test_execution_without_authorization_cannot_start_agent(self):
        directory = self.prepare(execution_authorized=False)
        result = self.run_cli(directory)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.agent_log.exists())
        self.assertFalse(any("POST" in args for args in self.calls(self.api_log)))


if __name__ == "__main__":
    unittest.main()
