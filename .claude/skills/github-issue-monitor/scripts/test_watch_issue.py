#!/usr/bin/env python3
"""Offline invariants and process lifecycle tests; no live GitHub mutations."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("watch_issue.py")
SPEC = importlib.util.spec_from_file_location("watch_issue", SCRIPT)
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


def comment(comment_id, body="@agent do work", author="owner"):
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": author},
        "html_url": f"https://github.com/owner/repo/issues/7#issuecomment-{comment_id}",
        "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
    }


class FakeGitHub:
    def __init__(self):
        self.listed = []
        self.live = {}
        self.posts = 0
        self.identity_name = "reporter"
        self.fail_list = False
        self.fail_live = False
        self.alter_post = False

    def comments(self):
        if self.fail_list:
            raise w.GitHubError(503)
        return self.listed.copy()

    def comment(self, comment_id):
        if self.fail_live:
            raise w.GitHubError(403)
        return self.live.get(int(comment_id))

    def identity(self):
        return self.identity_name

    def post(self, body_file):
        self.posts += 1
        body = body_file.read_text()
        item = comment(
            900 + self.posts, body + (" altered" if self.alter_post else ""), "reporter"
        )
        self.listed.append(item)
        self.live[item["id"]] = item
        return item["id"]

    def add(self, item):
        self.listed.append(item)
        self.live[item["id"]] = item


class PollTests(unittest.TestCase):
    def setUp(self):
        w.STOP.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config = {
            "repository": "owner/repo",
            "issue": 7,
            "hostname": "github.com",
            "author": "owner",
            "mention": "@agent",
            "result_authors": ["reporter"],
            "authorize_execution": True,
            "authorize_results": True,
            "interval_seconds": 900,
            "cwd": str(self.directory),
            "scope": "test work",
            "gh": "gh",
            "delivery_command": ["adapter", "{message}"],
        }
        w.atomic_json(self.directory / "config.json", self.config)
        w.atomic_json(
            self.directory / "state.json",
            {"enabled": True, "baseline_id": 10, "high_water_id": 10, "deliveries": {}},
        )
        self.gh = FakeGitHub()
        self.sent = []

        def sender(item, config, directory):
            state = w.read_json(directory / "state.json")
            self.assertEqual(
                state["deliveries"][str(item["id"])]["status"], "delivering"
            )
            self.sent.append(item)
            return subprocess.CompletedProcess([], 0, stdout="Accepted", stderr="")

        self.watcher = w.Watcher(self.directory, github=self.gh, sender=sender)

    def tearDown(self):
        w.STOP.clear()
        self.temporary.cleanup()

    def state(self):
        return w.read_json(self.directory / "state.json")

    def result(self, request_id, result_id=50, author="reporter"):
        self.gh.add(
            comment(
                result_id,
                "Verified result\n\n" + w.marker(self.config, request_id),
                author,
            )
        )

    def test_exact_author_and_mention(self):
        for text in (
            "@agent-extra x",
            "@@agent x",
            "mail@agent",
            "@Agent x",
            "@agents",
        ):
            self.assertFalse(w.eligible(comment(11, text), self.config), text)
        self.assertTrue(
            w.eligible(comment(11, "hello @agent, please test", "OWNER"), self.config)
        )
        self.assertFalse(w.eligible(comment(11, author="outsider"), self.config))

    def test_baseline_dryrun_and_sequential_delivery(self):
        for item in (
            comment(9),
            comment(11, author="outsider"),
            comment(12),
            comment(13),
        ):
            self.gh.add(item)
        before = (self.directory / "state.json").read_bytes()
        self.assertEqual(self.watcher.poll(dry_run=True)["new_eligible_ids"], [12, 13])
        self.assertEqual(before, (self.directory / "state.json").read_bytes())
        self.watcher.poll()
        self.watcher.poll()
        self.assertEqual([c["id"] for c in self.sent], [12])
        self.assertEqual(self.state()["high_water_id"], 13)
        self.result(12)
        self.watcher.poll()
        self.assertEqual([c["id"] for c in self.sent], [12, 13])
        self.assertEqual(self.state()["deliveries"]["12"]["status"], "completed")

    def test_live_recheck_skips_deleted_and_uses_latest_body(self):
        self.gh.add(comment(11))
        self.gh.add(comment(12))
        self.gh.live[11] = None
        self.gh.live[12] = comment(12, "@agent edited task")
        self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "skipped")
        self.assertEqual(self.sent[0]["body"], "@agent edited task")

    def test_markers_require_exact_author_issue_and_live_readback(self):
        self.gh.add(comment(11))
        self.watcher.poll()
        self.result(11, 50, "outsider")
        self.result(11, 51)
        self.gh.live[51] = comment(51, "removed marker", "reporter")
        self.result(11, 52)
        self.gh.live[52][
            "issue_url"
        ] = "https://api.github.com/repos/owner/repo/issues/8"
        self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "queued")
        self.result(11, 53)
        self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "completed")

    def test_quoted_marker_does_not_suppress_a_request(self):
        self.gh.add(comment(11))
        self.gh.add(
            comment(
                12,
                "@agent investigate this quoted result:\n" + w.marker(self.config, 11),
            )
        )
        self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "queued")
        self.assertEqual(self.state()["deliveries"]["12"]["status"], "pending")

    def test_fetch_errors_do_not_advance_cursor(self):
        before = (self.directory / "state.json").read_bytes()
        self.gh.fail_list = True
        with self.assertRaises(w.GitHubError):
            self.watcher.poll()
        self.assertEqual(before, (self.directory / "state.json").read_bytes())
        self.gh.fail_list = False
        self.gh.add(comment(11))
        self.gh.fail_live = True
        with self.assertRaises(w.GitHubError):
            self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "pending")

    def test_ambiguous_delivery_does_not_replay_or_pass_later_task(self):
        self.gh.add(comment(11))
        self.gh.add(comment(12))
        attempts = []

        def timeout(item, config, directory):
            attempts.append(item["id"])
            raise subprocess.TimeoutExpired("adapter", 45)

        self.watcher.sender = timeout
        self.watcher.poll()
        self.watcher.poll()
        self.assertEqual(attempts, [11])
        self.assertEqual(
            self.state()["deliveries"]["11"]["status"], "delivery_uncertain"
        )
        self.result(11)
        self.watcher.poll()
        self.assertEqual(attempts, [11, 12])

    def test_stop_cancels_pending_and_preserves_queued(self):
        self.gh.add(comment(11))
        self.gh.add(comment(12))
        self.watcher.poll()
        self.assertTrue(w.stop(self.directory, wait_seconds=1)["stopped"])
        self.watcher.poll()
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(self.state()["enabled"])
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "queued")
        self.assertEqual(
            self.state()["deliveries"]["12"]["status"], "cancelled_on_stop"
        )

    def test_failed_or_crashed_delivery_requires_recovery(self):
        self.gh.add(comment(11))
        self.gh.add(comment(12))
        self.watcher.sender = lambda *_: subprocess.CompletedProcess(
            [], 1, stdout="", stderr="failed"
        )
        self.watcher.poll()
        self.assertEqual(self.state()["deliveries"]["11"]["status"], "delivery_failed")
        self.assertTrue(self.watcher.poll()["waiting_for_result_or_recovery"])
        with self.watcher.state() as state:
            state["deliveries"]["11"]["status"] = "delivering"
        self.assertTrue(self.watcher.poll()["waiting_for_result_or_recovery"])
        self.assertEqual(self.state()["deliveries"]["12"]["status"], "pending")

    def test_post_result_deduplication_and_readback(self):
        self.gh.add(comment(11))
        body = self.directory / "result.md"
        body.write_text("Handled the request.\n\n" + w.marker(self.config, 11) + "\n")
        result = self.watcher.post_result(11, body)
        self.assertTrue(result["verified"])
        self.assertTrue(self.watcher.post_result(11, body)["already_handled"])
        self.assertEqual(self.gh.posts, 1)
        self.gh.identity_name = "outsider"
        with self.assertRaises(RuntimeError):
            self.watcher.post_result(11, body)

    def test_post_refuses_without_authorization_and_detects_changed_body(self):
        body = self.directory / "result.md"
        body.write_text("Blocked request.\n\n" + w.marker(self.config, 11) + "\n")
        self.watcher.config["authorize_results"] = False
        with self.assertRaises(RuntimeError):
            self.watcher.post_result(11, body)
        self.watcher.config["authorize_results"] = True
        self.gh.alter_post = True
        with self.assertRaises(RuntimeError):
            self.watcher.post_result(11, body)

    def test_comment_is_passed_as_data_to_adapter(self):
        capture = self.directory / "capture.json"
        adapter = self.directory / "adapter.py"
        adapter.write_text(
            "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))"
        )
        sentinel = self.directory / "MUST_NOT_EXIST"
        body = f"@agent $(touch {sentinel}) `touch {sentinel}`\nUnicode 中文"
        config = dict(
            self.config,
            delivery_command=[sys.executable, str(adapter), str(capture), "{message}"],
        )
        result = w.deliver(comment(11, body), config, self.directory)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(sentinel.exists())
        payload = json.loads(capture.read_text())
        self.assertEqual(len(payload), 1)
        self.assertIn("MUST_NOT_EXIST", payload[0])
        self.assertIn("Unicode 中文", payload[0])


class ProcessTests(unittest.TestCase):
    def test_initialize_start_stop_resume_and_private_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = directory / "github.json"
            calls = directory / "api-calls.jsonl"
            delivery = directory / "delivered.jsonl"
            fixture.write_text(json.dumps([comment(10)]))
            fake = directory / "gh"
            fake.write_text(f"#!{sys.executable}\n" + """import json,sys
from pathlib import Path
base=Path(__file__).parent
args=sys.argv[1:]
with (base/'api-calls.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')
endpoint=args[1]
comments=json.loads((base/'github.json').read_text())
if endpoint=='user':result={'login':'reporter'}
elif '/comments?' in endpoint:result=[comments[:1],comments[1:]]
else:
 result=next((c for c in comments if str(c['id'])==endpoint.rsplit('/',1)[-1]),None)
 if result is None:sys.stderr.write('HTTP 404');sys.exit(1)
print(json.dumps(result))
""")
            fake.chmod(0o700)
            adapter = directory / "adapter.py"
            adapter.write_text(
                "import json,sys; from pathlib import Path\nwith Path(sys.argv[1]).open('a') as f:f.write(json.dumps(sys.argv[2])+'\\n')\nprint('accepted')\n"
            )
            command = directory / "adapter.json"
            command.write_text(
                json.dumps([sys.executable, str(adapter), str(delivery), "{message}"])
            )
            runtime = directory / "runtime"

            def cli(*args):
                response = subprocess.run(
                    [sys.executable, str(SCRIPT), "--directory", str(runtime), *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(response.returncode, 0, response.stderr)
                return json.loads(response.stdout)

            def wait_delivery(count):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if (
                        delivery.exists()
                        and len(delivery.read_text().splitlines()) >= count
                    ):
                        return
                    time.sleep(0.05)
                self.fail("Adapter delivery was not observed")

            cli(
                "init",
                "--repository",
                "owner/repo",
                "--issue",
                "7",
                "--author",
                "owner",
                "--mention",
                "@agent",
                "--interval-seconds",
                "60",
                "--cwd",
                str(directory),
                "--scope",
                "test authorized tasks",
                "--gh",
                str(fake),
                "--delivery-command-file",
                str(command),
                "--authorize-execution",
                "--authorize-results",
            )
            self.assertEqual(w.read_json(runtime / "state.json")["baseline_id"], 10)
            self.assertEqual((runtime / "config.json").stat().st_mode & 0o777, 0o600)
            fixture.write_text(json.dumps([comment(10), comment(11)]))
            self.assertEqual(cli("once", "--dry-run")["new_eligible_ids"], [11])
            try:
                self.assertTrue(cli("start")["started"])
                wait_delivery(1)
                current = cli("status")
                self.assertTrue(current["running"])
                self.assertNotIn("delivery_command", current)
                self.assertNotIn("comment", current)
                self.assertNotIn("scope", current)
                self.assertTrue(cli("stop")["stopped"])
                self.assertFalse((runtime / "watch.pid").exists())
                before = calls.read_text()
                time.sleep(0.1)
                self.assertEqual(calls.read_text(), before)
                self.assertTrue(w.daemon_lock_free(runtime))
                result = comment(
                    20,
                    "done\n" + w.marker(w.read_json(runtime / "config.json"), 11),
                    "reporter",
                )
                fixture.write_text(
                    json.dumps([comment(10), comment(11), comment(12), result])
                )
                self.assertTrue(cli("start")["started"])
                self.assertTrue(cli("stop")["stopped"])
                self.assertEqual(len(delivery.read_text().splitlines()), 1)
                self.assertEqual(
                    w.read_json(runtime / "state.json")["high_water_id"], 20
                )
                # Explicit backlog inclusion revives undelivered work from a pause.
                with w.Watcher(runtime).state() as state:
                    state["deliveries"]["12"] = {"status": "cancelled_on_stop"}
                fixture.write_text(
                    json.dumps([comment(10), comment(11), comment(12), result])
                )
                self.assertTrue(cli("start", "--include-paused")["started"])
                wait_delivery(2)
                self.assertTrue(cli("stop")["stopped"])
            finally:
                if w.live_pid(runtime):
                    cli("stop")


if __name__ == "__main__":
    unittest.main()
