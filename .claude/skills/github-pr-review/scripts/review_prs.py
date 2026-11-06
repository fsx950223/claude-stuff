#!/usr/bin/env python3
"""Review assigned/requested GitHub PRs and publish only against an unchanged SHA."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state.json"
GH = "gh"
CODEX = "codex"
CLAUDE = "claude"
HOST = "github.com"
SKILL_ROOT = Path(__file__).resolve().parent.parent


def resource(name):
    runtime_file = ROOT / name
    return runtime_file if runtime_file.is_file() else SKILL_ROOT / "assets" / name


def validate_config(config):
    login = config.get("github_login")
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]+", login):
        raise ValueError("github_login must be the verified GitHub login")
    for key in (
        "review_requested",
        "assigned",
        "publish_reviews",
        "execution_authorized",
        "publication_authorized",
    ):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(key + " must be a boolean")
    if not (config.get("review_requested") or config.get("assigned")):
        raise ValueError("Enable review_requested, assigned, or both")
    if config.get("publish_reviews") and not config.get("publication_authorized"):
        raise ValueError("Publishing requires recorded publication_authorized")
    if config.get("review_backend", "codex") not in {"codex", "claude"}:
        raise ValueError("review_backend must be codex or claude")
    timeout = config.get("pr_timeout_seconds", 2700)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("pr_timeout_seconds must be a positive integer")
    if not isinstance(config.get("test_environments", {}), dict):
        raise ValueError("test_environments must be an object")
    repositories = config.get("repositories", [])
    if not isinstance(repositories, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", name)
        or name.rsplit("/", 1)[1] in {".", ".."}
        for name in repositories
    ):
        raise ValueError("repositories must be a list of OWNER/REPO names")
    host = config.get("github_host", "github.com")
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
        raise ValueError("github_host must be a hostname")


def configure(directory, config):
    global ROOT, STATE, GH, CODEX, CLAUDE, HOST
    ROOT = Path(directory).expanduser().resolve()
    STATE = ROOT / "state.json"
    GH = config.get("gh") or shutil.which("gh") or "gh"
    CODEX = config.get("codex") or shutil.which("codex") or "codex"
    CLAUDE = config.get("claude") or shutil.which("claude") or "claude"
    HOST = config.get("github_host", "github.com")


def allowed_repository(repo, config):
    selected = {name.casefold() for name in config.get("repositories", [])}
    return not selected or repo.casefold() in selected


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command_json(args, timeout=90):
    environment = os.environ.copy()
    environment["GH_HOST"] = HOST
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Command failed: " + args[0])
    return json.loads(result.stdout)


def api(endpoint, *options):
    return command_json([GH, "api", "--hostname", HOST, endpoint, *options])


def discover(config):
    candidates = {}
    for enabled, flag in [
        (config["review_requested"], "--review-requested"),
        (config["assigned"], "--assignee"),
    ]:
        if not enabled:
            continue
        repository_args = [
            value
            for repo in config.get("repositories", [])
            for value in ("--repo", repo)
        ]
        found = command_json(
            [
                GH,
                "search",
                "prs",
                flag,
                config["github_login"],
                *repository_args,
                "--state=open",
                "--limit=1000",
                "--json",
                "number,title,url,repository,isDraft,updatedAt",
            ]
        )
        if len(found) >= 1000:
            raise RuntimeError(
                "GitHub search reached its 1000-result limit; narrow the scope."
            )
        for pr in found:
            if not allowed_repository(pr["repository"]["nameWithOwner"], config):
                continue
            candidates[pr["url"]] = pr
    # A COMMENT can fulfill GitHub's review request before hardware validation
    # is complete. Keep those PRs in our queue until a terminal review or closure.
    tracked = json.loads(STATE.read_text()) if STATE.exists() else {}
    for key, previous in tracked.items():
        if previous.get("event") != "COMMENT" or previous.get("pr") in candidates:
            continue
        repo, number = key.rsplit("#", 1)
        if not allowed_repository(repo, config):
            continue
        current = api("repos/{}/pulls/{}".format(repo, int(number)))
        if current["state"] == "open" and not current["draft"]:
            candidates[current["html_url"]] = {
                "number": current["number"],
                "title": current["title"],
                "url": current["html_url"],
                "repository": {"nameWithOwner": repo},
                "isDraft": current["draft"],
                "updatedAt": current["updated_at"],
            }
    return sorted(candidates.values(), key=lambda x: x["updatedAt"], reverse=True)


def selector(pr):
    return pr["repository"]["nameWithOwner"], pr["number"]


def fingerprint(pr):
    return pr["head"]["sha"] + ":" + pr["base"]["sha"]


def choose_event(result, own_pr=False):
    event = result["verdict"]
    if event not in {"APPROVE", "REQUEST_CHANGES", "COMMENT"}:
        raise ValueError("Invalid review verdict")
    blocking = any(f["priority"] in {"P0", "P1", "P2"} for f in result["findings"])
    if event == "APPROVE" and blocking:
        raise ValueError("Refusing approval with a blocking finding")
    if event == "REQUEST_CHANGES" and not blocking:
        raise ValueError("Request changes needs an evidenced blocking finding")
    if own_pr or (event == "APPROVE" and not result["validation_sufficient"]):
        event = "COMMENT"
    return event


def render(result, repo, head, event, token):
    parts = [result["summary"].strip()]
    for finding in result["findings"]:
        path = finding["path"]
        if path.startswith("/") or ".." in Path(path).parts or finding["line"] < 1:
            raise ValueError("Invalid finding location")
        url = "https://{}/{}/blob/{}/{}#L{}".format(
            HOST, repo, head, quote(path, safe="/"), finding["line"]
        )
        parts.append(
            "**[{}] {}** — [{}:{}]({})\n\n{}".format(
                finding["priority"],
                finding["title"],
                path,
                finding["line"],
                url,
                finding["explanation"],
            )
        )
    if result["tests"]:
        rows = ["Validation:", ""]
        for test in result["tests"]:
            rows.append(
                "- **{}** on `{}`: `{}`\n  {}".format(
                    test["result"], test["host"], test["command"], test["evidence"]
                )
            )
        parts.append("\n".join(rows))
    if result["limitations"]:
        parts.append(
            "Remaining validation:\n\n"
            + "\n".join("- " + value for value in result["limitations"])
        )
    parts.append("Automated PR review of commit `{}`.\n\n{}".format(head[:12], token))
    body = "\n\n".join(parts)
    if not result["summary"].strip() or len(body) > 60000:
        raise ValueError("Review body is empty or too large")
    return body


def agent_command(task_dir, config):
    if config.get("review_backend", "codex") == "claude":
        return [
            CLAUDE,
            "--print",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Bash,Read,Write,Edit,Glob,Grep",
            "--json-schema",
            resource("review-schema.json").read_text(),
        ]
    return [
        CODEX,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--color=never",
        "--sandbox=workspace-write",
        "-c",
        'approval_policy="never"',
        "-c",
        "sandbox_workspace_write.network_access=true",
        "--cd",
        str(task_dir),
        "--output-schema",
        str(resource("review-schema.json")),
        "--output-last-message",
        str(task_dir / "result.json"),
        "-",
    ]


def claude_result(envelope):
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        raise ValueError("Claude returned an error envelope")
    result = envelope.get("structured_output")
    if not isinstance(result, dict):
        raise ValueError("Claude did not return structured_output")
    return result


def run_model(task_dir, task, config):
    atomic_json(task_dir / "task.json", task)
    prompt = resource("review-prompt.md").read_text()
    prompt += "\n\nTask JSON:\n" + json.dumps(task, ensure_ascii=False, indent=2)
    prompt += "\n\nConfigured test environments:\n" + json.dumps(
        config.get("test_environments", {}), indent=2
    )
    prompt += "\n\nLive runtime config path: " + str(ROOT / "config.json")
    output = task_dir / "result.json"
    args = agent_command(task_dir, config)
    with (task_dir / "events.jsonl").open("w") as events, (
        task_dir / "agent.stderr.log"
    ).open("w") as errors:
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=errors,
            text=True,
            start_new_session=True,
            cwd=task_dir,
        )
        try:
            process.communicate(prompt, timeout=config.get("pr_timeout_seconds", 2700))
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError("Agent review failed; inspect " + str(task_dir))
    if config.get("review_backend", "codex") == "claude":
        envelope = json.loads((task_dir / "events.jsonl").read_text())
        atomic_json(output, claude_result(envelope))
    if not output.is_file():
        raise RuntimeError("Agent produced no review result; inspect " + str(task_dir))
    result = json.loads(output.read_text())
    # Validate the full response before permitting publication.
    import jsonschema

    jsonschema.validate(result, json.loads(resource("review-schema.json").read_text()))
    return result


def cleanup_checkouts(task_dir):
    # These directories were created inside this run's isolated workspace.
    # Keep structured results and logs, rather than accumulating repository copies.
    for child in task_dir.iterdir():
        if child.is_dir() and not child.is_symlink() and (child / ".git").exists():
            shutil.rmtree(child)


def publish(repo, number, initial, result, config, task_dir):
    endpoint = "repos/{}/pulls/{}".format(repo, number)
    current = api(endpoint)
    if current["state"] != "open" or current["draft"]:
        return {"status": "skipped", "reason": "PR closed or became a draft"}
    if fingerprint(current) != fingerprint(initial):
        return {"status": "stale", "reason": "Head or base changed; retry next run"}
    event = choose_event(
        result, own_pr=current["user"]["login"] == config["github_login"]
    )
    identity = hashlib.sha256(initial["head"]["sha"].encode()).hexdigest()[:20]
    token = "<!-- codex-daily-pr-review:{}:{} -->".format(identity, event)
    body = render(result, repo, initial["head"]["sha"], event, token)
    pages = api(endpoint + "/reviews?per_page=100", "--paginate", "--slurp")
    expected_state = {
        "APPROVE": "APPROVED",
        "REQUEST_CHANGES": "CHANGES_REQUESTED",
        "COMMENT": "COMMENTED",
    }[event]
    for page in pages:
        for review in page:
            if (
                review["user"]["login"] == config["github_login"]
                and review.get("state") == expected_state
                and (
                    token in (review.get("body") or "")
                    or (
                        review.get("commit_id") == initial["head"]["sha"]
                        and "<!-- codex-daily-pr-review:" in (review.get("body") or "")
                    )
                )
            ):
                return {
                    "status": "already_submitted",
                    "event": event,
                    "url": review["html_url"],
                    "review_id": review["id"],
                    "fingerprint": fingerprint(initial),
                }
    payload = {"commit_id": initial["head"]["sha"], "event": event, "body": body}
    payload_path = task_dir / "review-payload.json"
    atomic_json(payload_path, payload)
    (task_dir / "review.md").write_text(body + "\n")
    if not config["publish_reviews"]:
        return {"status": "draft", "event": event}
    if not config.get("publication_authorized"):
        raise RuntimeError("Review publication is not authorized")
    if api("user")["login"] != config["github_login"]:
        raise RuntimeError("GitHub active account changed before publication")
    # This is the only mutation of GitHub performed by the runner.
    submitted = api(
        endpoint + "/reviews", "--method", "POST", "--input", str(payload_path)
    )
    return {
        "status": "submitted",
        "event": event,
        "url": submitted["html_url"],
        "review_id": submitted["id"],
        "fingerprint": fingerprint(initial),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(
            os.environ.get(
                "GITHUB_PR_REVIEW_HOME",
                str(Path.home() / ".local/share/github-pr-review"),
            )
        ),
        help="Private runtime directory containing config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover PRs without reviewing or publishing",
    )
    parser.add_argument(
        "--pr", help="Review only this currently assigned/requested PR URL"
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    config = json.loads((directory / "config.json").read_text())
    validate_config(config)
    configure(directory, config)
    login = api("user")["login"]
    if login != config["github_login"]:
        raise RuntimeError(
            "GitHub active account changed; refusing to review as " + login
        )
    candidates = discover(config)
    if args.pr:
        candidates = [pr for pr in candidates if pr["url"] == args.pr]
        if not candidates:
            raise RuntimeError("Selected PR is not currently assigned/requested")
    if args.dry_run:
        print(
            json.dumps(
                {"account": login, "prs": candidates}, ensure_ascii=False, indent=2
            )
        )
        return
    if not config.get("execution_authorized"):
        raise RuntimeError("Review execution is not authorized")
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another review run is active; skipping", flush=True)
            return
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = ROOT / "reports" / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        outcomes = []
        for candidate in candidates:
            repo, number = selector(candidate)
            key = "{}#{}".format(repo, number)
            try:
                initial = api("repos/{}/pulls/{}".format(repo, number))
                if initial["state"] != "open" or initial["draft"]:
                    outcomes.append({"pr": candidate["url"], "status": "skipped"})
                    continue
                previous = state.get(key, {})
                if (
                    previous.get("fingerprint", "").split(":")[0]
                    == initial["head"]["sha"]
                    and previous.get("review_id")
                    and previous.get("event") in {"APPROVE", "REQUEST_CHANGES"}
                ):
                    recorded = api(
                        "repos/{}/pulls/{}/reviews/{}".format(
                            repo, number, previous["review_id"]
                        )
                    )
                    expected = {
                        "APPROVE": "APPROVED",
                        "REQUEST_CHANGES": "CHANGES_REQUESTED",
                    }
                    if recorded["state"] == expected[previous["event"]]:
                        outcomes.append({"pr": candidate["url"], "status": "unchanged"})
                        continue
                task_dir = run_dir / (repo.replace("/", "--") + "--" + str(number))
                task_dir.mkdir()
                task = {
                    "repository": repo,
                    "number": number,
                    "url": candidate["url"],
                    "reviewer": login,
                    "head_sha": initial["head"]["sha"],
                    "base_sha": initial["base"]["sha"],
                    "title": initial["title"],
                    "body": initial.get("body") or "",
                }
                print("Reviewing " + candidate["url"], flush=True)
                result = run_model(task_dir, task, config)
                outcome = publish(repo, number, initial, result, config, task_dir)
                outcome["pr"] = candidate["url"]
                outcomes.append(outcome)
                if outcome["status"] in {"submitted", "already_submitted"}:
                    state[key] = outcome
                    atomic_json(STATE, state)
                print(json.dumps(outcome), flush=True)
                cleanup_checkouts(task_dir)
            except Exception as exc:
                outcomes.append(
                    {"pr": candidate["url"], "status": "error", "error": str(exc)}
                )
                print(
                    "Review error for {}: {}".format(candidate["url"], exc), flush=True
                )
            atomic_json(run_dir / "summary.json", outcomes)
            atomic_json(
                ROOT / "latest.json", {"run": str(run_dir), "outcomes": outcomes}
            )
        atomic_json(run_dir / "summary.json", outcomes)
        atomic_json(ROOT / "latest.json", {"run": str(run_dir), "outcomes": outcomes})
        if any(outcome["status"] == "error" for outcome in outcomes):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
