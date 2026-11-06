#!/usr/bin/env python3
"""Prepare a PR-review runtime; activate its user timer only when requested."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from review_prs import validate_config

SKILL_ROOT = Path(__file__).resolve().parent.parent


def calendar(config):
    time = config.get("daily_time", "09:00")
    if not isinstance(time, str) or not re.fullmatch(
        r"(?:[01]\d|2[0-3]):[0-5]\d", time
    ):
        raise ValueError("daily_time must use HH:MM")
    timezone = config.get("timezone", "Asia/Shanghai")
    if not isinstance(timezone, str) or not re.fullmatch(r"[A-Za-z0-9_+/-]+", timezone):
        raise ValueError("timezone must be an IANA timezone")
    ZoneInfo(timezone)
    return "*-*-* " + time + ":00 " + timezone


def unit_quote(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Unit values must be single-line")
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def render_units(directory, config, unit_name):
    directory = Path(directory)
    schedule = calendar(config)
    service = f"""[Unit]
Description=Review requested and assigned GitHub pull requests

[Service]
Type=oneshot
WorkingDirectory={str(directory).replace('%', '%%')}
ExecStart={unit_quote(sys.executable)} {unit_quote(directory / "review_prs.py")} --directory {unit_quote(directory)}
Environment={unit_quote("PATH=" + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))}
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus
Environment=PYTHONUNBUFFERED=1
Environment=GIT_TERMINAL_PROMPT=0
UMask=0077
TimeoutStartSec=6h
KillMode=control-group
Nice=10
"""
    timer = f"""[Unit]
Description=Scheduled GitHub PR reviews

[Timer]
OnCalendar={schedule}
Persistent=true
AccuracySec=1min
Unit={unit_name}.service

[Install]
WantedBy=timers.target
"""
    return service, timer


def run(args):
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    ).stdout


def verify_activation(config):
    if not config.get("execution_authorized"):
        raise ValueError("Timer activation requires recorded execution_authorized")
    if importlib.util.find_spec("jsonschema") is None:
        raise ValueError(
            "Activation requires jsonschema in the runtime Python environment"
        )
    gh = config.get("gh") or shutil.which("gh")
    backend = config.get("review_backend", "codex")
    agent = config.get(backend) or shutil.which(backend)
    if not gh or not agent or not shutil.which("git"):
        raise ValueError("Activation requires gh, git and the selected agent CLI")
    login = run(
        [
            gh,
            "api",
            "--hostname",
            config.get("github_host", "github.com"),
            "user",
            "--jq",
            ".login",
        ]
    ).strip()
    if login != config["github_login"]:
        raise ValueError("GitHub active account differs from github_login")
    if not shutil.which("systemctl") or not shutil.which("systemd-analyze"):
        raise ValueError("Activation requires user systemd and systemd-analyze")


def prepare(
    directory, config, unit_name="github-pr-review", activate=False, unit_directory=None
):
    validate_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", unit_name):
        raise ValueError("Invalid unit name")
    directory = Path(directory).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Runtime already exists; inspect it instead of overwriting")
    service, timer = render_units(directory, config, unit_name)
    destination = (
        Path(unit_directory).expanduser().resolve()
        if unit_directory
        else Path.home() / ".config/systemd/user"
    )
    names = [unit_name + ".service", unit_name + ".timer"]
    if activate:
        if any((destination / name).exists() for name in names):
            raise ValueError(
                "Destination units already exist; inspect the existing reviewer"
            )
        verify_activation(config)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    files = {
        "review_prs.py": SKILL_ROOT / "scripts/review_prs.py",
        "review-prompt.md": SKILL_ROOT / "assets/review-prompt.md",
        "review-schema.json": SKILL_ROOT / "assets/review-schema.json",
    }
    for name, source in files.items():
        shutil.copyfile(source, directory / name)
        os.chmod(directory / name, 0o600)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    os.chmod(directory / "config.json", 0o600)
    rendered = directory / "systemd"
    rendered.mkdir(mode=0o700)
    for name, content in zip(names, [service, timer]):
        (rendered / name).write_text(content)
        os.chmod(rendered / name, 0o600)
    if activate:
        run(["systemd-analyze", "calendar", calendar(config)])
        run(
            [
                "systemd-analyze",
                "--user",
                "verify",
                *(str(rendered / name) for name in names),
            ]
        )
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(rendered / name, destination / name)
            os.chmod(destination / name, 0o600)
        run(["systemctl", "--user", "daemon-reload"])
        run(["systemctl", "--user", "enable", "--now", unit_name + ".timer"])
        if (
            run(["systemctl", "--user", "is-active", unit_name + ".timer"]).strip()
            != "active"
        ):
            raise RuntimeError("Timer activation could not be verified")
        print(
            run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit_name + ".timer",
                    "-p",
                    "ActiveState",
                    "-p",
                    "NextElapseUSecRealtime",
                ]
            ),
            end="",
        )
    return {
        "directory": str(directory),
        "calendar": calendar(config),
        "activated": activate,
        "units": [str(rendered / name) for name in names],
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--unit-name", default="github-pr-review")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Verify identity and enable the prepared user timer",
    )
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text())
    print(
        json.dumps(
            prepare(args.directory, config, args.unit_name, args.activate), indent=2
        )
    )


if __name__ == "__main__":
    main()
