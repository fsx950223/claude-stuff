import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_prs as run


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "state": "open",
            "draft": False,
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40},
            "user": {"login": "contributor"},
        }
        self.result = {
            "verdict": "APPROVE",
            "summary": "No blocking issues found.",
            "validation_sufficient": True,
            "findings": [],
            "tests": [],
            "limitations": [],
        }
        self.config = {
            "github_login": "reviewer",
            "publish_reviews": True,
            "publication_authorized": True,
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def publish(self, responses):
        with patch.object(run, "api", side_effect=responses) as remote:
            outcome = run.publish(
                "org/repo", 1, self.initial, self.result, self.config, self.directory
            )
        return outcome, remote

    def successful_responses(self):
        return [
            self.initial,
            [[]],
            {"login": "reviewer"},
            {"id": 42, "html_url": "https://github.com/org/repo/pull/1#review"},
        ]

    def payload(self):
        return json.loads((self.directory / "review-payload.json").read_text())

    def test_approval_is_bound_to_reviewed_commit(self):
        outcome, remote = self.publish(self.successful_responses())
        self.assertEqual(outcome["status"], "submitted")
        self.assertEqual(self.payload()["commit_id"], "a" * 40)
        self.assertEqual(self.payload()["event"], "APPROVE")
        self.assertIn("POST", remote.call_args.args)

    def test_head_update_prevents_any_publication(self):
        changed = copy.deepcopy(self.initial)
        changed["head"]["sha"] = "c" * 40
        outcome, remote = self.publish([changed])
        self.assertEqual(outcome["status"], "stale")
        self.assertEqual(remote.call_count, 1)
        self.assertFalse((self.directory / "review-payload.json").exists())

    def test_base_update_prevents_any_publication(self):
        changed = copy.deepcopy(self.initial)
        changed["base"]["sha"] = "c" * 40
        outcome, remote = self.publish([changed])
        self.assertEqual(outcome["status"], "stale")
        self.assertEqual(remote.call_count, 1)

    def test_insufficient_validation_cannot_approve(self):
        self.result["validation_sufficient"] = False
        self.result["limitations"] = ["Required GPU host is unavailable."]
        self.publish(self.successful_responses())
        self.assertEqual(self.payload()["event"], "COMMENT")

    def test_blocking_finding_cannot_approve(self):
        self.result["findings"] = [
            {
                "priority": "P1",
                "title": "Out-of-bounds access",
                "path": "kernel.py",
                "line": 10,
                "explanation": "The final partial tile accesses elements past N.",
            }
        ]
        with patch.object(run, "api", return_value=self.initial) as remote:
            with self.assertRaisesRegex(ValueError, "blocking finding"):
                run.publish(
                    "org/repo",
                    1,
                    self.initial,
                    self.result,
                    self.config,
                    self.directory,
                )
        self.assertEqual(remote.call_count, 1)

    def test_recovery_after_post_does_not_duplicate_review(self):
        self.publish(self.successful_responses())
        existing = {
            "id": 42,
            "state": "APPROVED",
            "user": {"login": "reviewer"},
            "body": self.payload()["body"],
            "html_url": "https://github.com/org/repo/pull/1#review",
        }
        outcome, remote = self.publish([self.initial, [[existing]]])
        self.assertEqual(outcome["status"], "already_submitted")
        self.assertEqual(remote.call_count, 2)

    def test_dismissed_review_does_not_suppress_new_review(self):
        self.publish(self.successful_responses())
        dismissed = {
            "id": 42,
            "state": "DISMISSED",
            "user": {"login": "reviewer"},
            "body": self.payload()["body"],
            "html_url": "https://github.com/org/repo/pull/1#review",
        }
        responses = self.successful_responses()
        responses[1] = [[dismissed]]
        outcome, remote = self.publish(responses)
        self.assertEqual(outcome["status"], "submitted")
        self.assertEqual(remote.call_count, 4)

    def test_request_changes_preserves_evidenced_finding(self):
        self.result["verdict"] = "REQUEST_CHANGES"
        self.result["findings"] = [
            {
                "priority": "P2",
                "title": "Tail tile reads beyond N",
                "path": "kernel.py",
                "line": 10,
                "explanation": "For N=65 the second tile reads beyond the allocation.\nAdd the tail mask.",
            }
        ]
        self.publish(self.successful_responses())
        self.assertEqual(self.payload()["event"], "REQUEST_CHANGES")
        self.assertIn("N=65", self.payload()["body"])
        self.assertIn("\nAdd the tail mask.", self.payload()["body"])

    def test_own_pr_uses_comment(self):
        self.initial["user"]["login"] = "reviewer"
        self.publish(self.successful_responses())
        self.assertEqual(self.payload()["event"], "COMMENT")

    def test_account_switch_prevents_post(self):
        with patch.object(
            run, "api", side_effect=[self.initial, [[]], {"login": "another-account"}]
        ) as remote:
            with self.assertRaisesRegex(RuntimeError, "account changed"):
                run.publish(
                    "org/repo",
                    1,
                    self.initial,
                    self.result,
                    self.config,
                    self.directory,
                )
        self.assertEqual(remote.call_count, 3)

    def test_incomplete_review_remains_in_queue_after_request_is_fulfilled(self):
        state = self.directory / "state.json"
        state.write_text(
            json.dumps(
                {
                    "org/repo#1": {
                        "event": "COMMENT",
                        "pr": "https://github.com/org/repo/pull/1",
                    }
                }
            )
        )
        current = {
            **self.initial,
            "number": 1,
            "title": "GPU fix",
            "html_url": "https://github.com/org/repo/pull/1",
            "updated_at": "2026-11-06T00:00:00Z",
        }
        config = {**self.config, "review_requested": True, "assigned": True}
        with patch.object(run, "STATE", state), patch.object(
            run, "command_json", return_value=[]
        ), patch.object(run, "api", return_value=current):
            discovered = run.discover(config)
        self.assertEqual([pr["number"] for pr in discovered], [1])

    def test_closed_incomplete_review_is_not_retried(self):
        state = self.directory / "state.json"
        state.write_text(
            json.dumps(
                {
                    "org/repo#1": {
                        "event": "COMMENT",
                        "pr": "https://github.com/org/repo/pull/1",
                    }
                }
            )
        )
        config = {**self.config, "review_requested": True, "assigned": True}
        with patch.object(run, "STATE", state), patch.object(
            run, "command_json", return_value=[]
        ), patch.object(run, "api", return_value={"state": "closed"}):
            self.assertEqual(run.discover(config), [])


class RepositoryScopeTests(unittest.TestCase):
    def test_discovery_filters_both_requests_and_assignments(self):
        config = {
            "github_login": "reviewer",
            "review_requested": True,
            "assigned": True,
            "repositories": ["ROCm/aiter"],
        }

        def candidate(repo, number):
            return {
                "repository": {"nameWithOwner": repo},
                "number": number,
                "url": "https://github.com/{}/pull/{}".format(repo, number),
                "updatedAt": "2026-11-06T00:00:00Z",
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                run, "STATE", Path(directory) / "state.json"
            ), patch.object(
                run,
                "command_json",
                return_value=[
                    candidate("ROCm/aiter", 1),
                    candidate("ROCm/AITemplate", 78),
                ],
            ) as query:
                result = run.discover(config)
        self.assertEqual(
            [pr["repository"]["nameWithOwner"] for pr in result], ["ROCm/aiter"]
        )
        self.assertEqual(query.call_count, 2)
        for call in query.call_args_list:
            self.assertIn("--repo", call.args[0])
            self.assertIn("ROCm/aiter", call.args[0])

    def test_out_of_scope_incomplete_review_does_not_reenter_queue(self):
        config = {
            "github_login": "reviewer",
            "review_requested": True,
            "assigned": True,
            "repositories": ["ROCm/aiter"],
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "ROCm/AITemplate#78": {
                            "event": "COMMENT",
                            "pr": "https://github.com/ROCm/AITemplate/pull/78",
                        }
                    }
                )
            )
            with patch.object(run, "STATE", state), patch.object(
                run, "command_json", return_value=[]
            ), patch.object(run, "api") as remote:
                result = run.discover(config)
                remote.assert_not_called()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
