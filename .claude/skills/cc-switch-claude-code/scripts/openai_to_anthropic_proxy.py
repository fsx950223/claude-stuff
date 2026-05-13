#!/usr/bin/env python3
"""Bridge Claude Code /v1/messages requests to AMD OpenAI chat completions."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time

import httpx


SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
LISTEN_HOST = os.environ.get("CLAUDE_OPENAI_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("CLAUDE_OPENAI_PROXY_PORT", "15721"))


def load_settings():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings = json.load(f)

    return {
        "base_url": settings.get("baseUrl", "").rstrip("/"),
        "headers": settings.get("headers") or {},
        "model": settings.get("model") or "GPT-5.5",
    }


def text_from_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    return ""


def convert_messages(body):
    messages = []
    system_text = text_from_content(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = "assistant" if msg.get("role") == "assistant" else "user"
        text = text_from_content(msg.get("content", ""))
        if text:
            messages.append({"role": role, "content": text})

    return messages


def anthropic_response(model, text, usage=None):
    usage = usage or {}
    return {
        "id": "msg_amd_openai_proxy",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        settings = load_settings()

        if not settings["base_url"].startswith("https://llm-api.amd.com/OpenAI"):
            self.write_error(
                500,
                "settings baseUrl must point at AMD OpenAI upstream, not the local proxy",
            )
            return

        openai_payload = {
            "model": settings["model"],
            "messages": convert_messages(body),
            "max_completion_tokens": min(int(body.get("max_tokens") or 256), 512),
        }

        try:
            with httpx.Client(verify=False, timeout=120) as client:
                resp = client.post(
                    settings["base_url"] + "/chat/completions",
                    headers={"Content-Type": "application/json", **settings["headers"]},
                    json=openai_payload,
                )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
            out = anthropic_response(body.get("model") or settings["model"], text, data.get("usage"))
        except Exception as exc:
            self.write_error(502, str(exc))
            return

        if body.get("stream"):
            self.write_stream(out, text)
        else:
            self.write_json(200, out)

    def write_stream(self, out, text):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        events = [
            ("message_start", {"type": "message_start", "message": {**out, "content": []}}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": out["usage"],
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]

        for event, data in events:
            self.wfile.write((f"event: {event}\n" + "data: " + json.dumps(data) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.01)

    def write_json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def write_error(self, status, message):
        self.write_json(status, {"type": "error", "error": {"type": "api_error", "message": message}})

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    print(f"Listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
