#!/usr/bin/env python3
"""Bridge Claude Code /v1/messages requests to AMD OpenAI chat completions.

The bridge translates enough of Anthropic Messages to OpenAI Chat Completions
for Claude Code tool use:

- Anthropic `tools` -> OpenAI function tools
- Anthropic assistant `tool_use` -> OpenAI assistant `tool_calls`
- Anthropic user `tool_result` -> OpenAI `tool` messages
- OpenAI assistant `tool_calls` -> Anthropic assistant `tool_use`
"""

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


def tool_result_to_text(block):
    content = block.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part)
    else:
        text = str(content)

    if block.get("is_error"):
        return "Tool error:\n" + text
    return text


def convert_tools(tools):
    openai_tools = []
    for tool in tools or []:
        if tool.get("type") not in (None, "custom", "tool"):
            continue

        name = tool.get("name")
        if not name:
            continue

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return openai_tools


def convert_assistant_content(blocks):
    text_parts = []
    tool_calls = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    message = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def convert_messages(body):
    messages = []
    system_text = text_from_content(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                messages.append(convert_assistant_content(content))
            else:
                messages.append({"role": "assistant", "content": str(content)})
            continue

        if isinstance(content, list):
            user_text = []
            for block in content:
                if not isinstance(block, dict):
                    user_text.append(str(block))
                    continue

                if block.get("type") == "tool_result":
                    if user_text:
                        messages.append({"role": "user", "content": "\n".join(user_text)})
                        user_text = []
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id") or "",
                            "content": tool_result_to_text(block),
                        }
                    )
                elif block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        user_text.append(text)

            if user_text:
                messages.append({"role": "user", "content": "\n".join(user_text)})
            continue

        text = text_from_content(content)
        if text:
            messages.append({"role": "user", "content": text})

    return messages


def parse_tool_arguments(raw_arguments):
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}


def anthropic_response(model, message, finish_reason, usage=None):
    usage = usage or {}
    content = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"call_{len(content)}",
                "name": function.get("name") or "",
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )

    stop_reason = "tool_use" if message.get("tool_calls") else "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "content_filter":
        stop_reason = "stop_sequence"

    return {
        "id": "msg_amd_openai_proxy",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
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
        tools = convert_tools(body.get("tools"))
        if tools:
            openai_payload["tools"] = tools
            openai_payload["tool_choice"] = "auto"

        try:
            with httpx.Client(verify=False, timeout=120) as client:
                resp = client.post(
                    settings["base_url"] + "/chat/completions",
                    headers={"Content-Type": "application/json", **settings["headers"]},
                    json=openai_payload,
                )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            out = anthropic_response(
                body.get("model") or settings["model"],
                choice.get("message") or {},
                choice.get("finish_reason"),
                data.get("usage"),
            )
        except Exception as exc:
            self.write_error(502, str(exc))
            return

        if body.get("stream"):
            self.write_stream(out)
        else:
            self.write_json(200, out)

    def write_stream(self, out):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        events = [("message_start", {"type": "message_start", "message": {**out, "content": []}})]
        for index, block in enumerate(out["content"]):
            if block["type"] == "text":
                events.extend(
                    [
                        (
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        ),
                        (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "text_delta", "text": block.get("text", "")},
                            },
                        ),
                        ("content_block_stop", {"type": "content_block_stop", "index": index}),
                    ]
                )
            elif block["type"] == "tool_use":
                events.extend(
                    [
                        (
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": block["id"],
                                    "name": block["name"],
                                    "input": {},
                                },
                            },
                        ),
                        (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(block.get("input") or {}),
                                },
                            },
                        ),
                        ("content_block_stop", {"type": "content_block_stop", "index": index}),
                    ]
                )

        events.extend(
            [
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": out["stop_reason"], "stop_sequence": None},
                        "usage": out["usage"],
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )

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
