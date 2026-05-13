---
name: cc-switch-claude-code
description: Diagnose and stabilize Claude Code when using cc-switch with AMD OpenAI-compatible providers. Use when Claude Code reports ANTHROPIC_AUTH_TOKEN auth conflicts, cc-switch proxy hangs, No file descriptors available, /v1/messages timeouts, or when validating claude -p through GPT-5.5 on AMD LLM.
---

# cc-switch Claude Code

Use this skill when Claude Code is routed through `cc-switch`, especially for AMD OpenAI-compatible providers such as `GPT-5.5`.

## Core Model

There are two distinct URLs. Do not mix them:

- Claude Code should call the local Anthropic-compatible endpoint: `ANTHROPIC_BASE_URL=http://127.0.0.1:15721`.
- The upstream provider should call AMD OpenAI: `https://llm-api.amd.com/OpenAI`.

If the provider's upstream base URL becomes `http://127.0.0.1:15721`, the proxy connects to itself, hangs, and can exhaust file descriptors.

## Initial Checks

Run these first:

```bash
cc-switch provider current --app claude
cc-switch proxy show --app claude
claude auth status
pgrep -af 'cc-switch|claude' || true
ss -ltnp 'sport = :15721' || true
```

Expected healthy shape:

- `cc-switch provider current --app claude` may show endpoint `http://127.0.0.1:15721` for Claude's local route.
- `~/.claude/settings.json` top-level `baseUrl` must be `https://llm-api.amd.com/OpenAI`.
- `~/.claude/settings.json` may set `env.ANTHROPIC_BASE_URL` to `http://127.0.0.1:15721`.
- No runaway `cc-switch proxy serve` process should be repeatedly accepting local self-connections.

Never print or commit real `Ocp-Apim-Subscription-Key` values.

## Authentication Conflict

Claude Code may print:

```text
Auth conflict: Both a token (ANTHROPIC_AUTH_TOKEN) and an API key (/login managed key) are set.
```

When using `cc-switch` proxy mode, prefer the proxy token path and remove Claude Code's login-managed key:

```bash
claude auth logout
```

Then re-test with:

```bash
claude -p "hi" --dangerously-skip-permissions < /dev/null
```

If using an isolated manual test with `--bare`, set `ANTHROPIC_API_KEY`; `--bare` does not rely on OAuth/keychain auth.

## Self-loop Symptoms

Treat these as proxy self-loop indicators:

- `claude -p "hi"` hangs with repeated connection retries.
- `cc-switch provider stream-check --app claude <provider>` hangs.
- Proxy logs show repeated connections to `http://127.0.0.1:15721/`.
- `axum::serve accept error: No file descriptors available (os error 24)`.
- A previous `cc-switch proxy serve` run crashed with `memory allocation ... failed`.

Stop the loop before testing again:

```bash
pkill -f 'cc-switch proxy serve' || true
pkill -f '^claude$' || true
ss -ltnp 'sport = :15721' || true
```

## Verify the Upstream Independently

Before blaming Claude Code, verify the AMD OpenAI endpoint directly using the same pattern as `test.py`:

```bash
python3 - <<'PY'
import ast
import pathlib
import time

import httpx
import openai

source = pathlib.Path('test.py').read_text()
module = ast.parse(source)
values = {}
for node in module.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in {'url', 'model_name', 'headers'}:
            values[name] = ast.literal_eval(node.value)

client = openai.OpenAI(
    base_url=values['url'],
    api_key='dummy',
    http_client=httpx.Client(verify=False, timeout=60),
    default_headers=values['headers'],
)

start = time.perf_counter()
response = client.chat.completions.create(
    model=values['model_name'],
    max_completion_tokens=20,
    messages=[{'role': 'user', 'content': 'hi'}],
)
print(f'ok elapsed={time.perf_counter() - start:.2f}s')
print(response.choices[0].message.content)
PY
```

Expected output includes a short greeting.

## Stable Fallback Bridge

If `cc-switch proxy serve` self-loops, use the included bridge instead of the built-in proxy:

```bash
python3 .claude/skills/cc-switch-claude-code/scripts/openai_to_anthropic_proxy.py &
```

It listens on `127.0.0.1:15721`, reads upstream settings from `~/.claude/settings.json`, and forwards Claude Code `/v1/messages` requests to AMD OpenAI `/chat/completions`.

Direct bridge test:

```bash
python3 - <<'PY'
import httpx

body = {
    'model': 'GPT-5.5',
    'max_tokens': 20,
    'messages': [{'role': 'user', 'content': 'hi'}],
}
r = httpx.post(
    'http://127.0.0.1:15721/v1/messages',
    headers={'x-api-key': 'dummy', 'anthropic-version': '2023-06-01'},
    json=body,
    timeout=60,
)
print(r.status_code)
print(r.text[:500])
PY
```

Claude Code test:

```bash
claude -p "hi" --dangerously-skip-permissions < /dev/null
```

`cc-switch` health test:

```bash
cc-switch provider stream-check --app claude amd-gpt-55
```

Passing criteria:

- Direct bridge test returns HTTP 200 and an Anthropic-style JSON message.
- `claude -p "hi"` exits with code 0 and prints a greeting.
- `cc-switch provider stream-check` reports `Status: operational`, `HTTP: 200`, `Retries: 0`.

## Debug Notes

Use a temporary isolated config when testing raw Claude Code behavior:

```bash
mkdir -p /tmp/claude-test-config
env CLAUDE_CONFIG_DIR=/tmp/claude-test-config \
  ANTHROPIC_BASE_URL=http://127.0.0.1:15721 \
  ANTHROPIC_API_KEY=dummy \
  ANTHROPIC_MODEL=GPT-5.5 \
  claude --bare -p "hi" --dangerously-skip-permissions < /dev/null
```

Without `CLAUDE_CONFIG_DIR`, `~/.claude/settings.json` can override environment variables and make tests misleading.
