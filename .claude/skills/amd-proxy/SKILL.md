---
name: amd-proxy
description: Start a local proxy that strips parameters unsupported by AMD's Vertex AI API (thinking, context_management, output_config, cache_control.scope), enabling `claude -p` subprocess calls to work correctly in this environment. Use this skill whenever you need to invoke claude as a subprocess or when a script calls `claude -p`.
---

# AMD Vertex Proxy

AMD's internal API (`https://llm-api.amd.com/Anthropic`) proxies requests to Vertex AI, which rejects several Claude Code-specific parameters. This skill starts a local HTTP proxy at `127.0.0.1:19998` that strips those fields before forwarding.

## When to Use

- Any script or tool needs to call `claude -p` as a subprocess
- You see error: `400 VertexGenAI returned BadRequest`
- Setting up automated Claude Code pipelines (e.g. Gmail listener, CI hooks)

## Stripped Parameters

| Parameter | Reason |
|-----------|--------|
| `thinking` | Claude Code extended thinking, not supported by Vertex |
| `context_management` | Claude Code context compression, non-standard |
| `output_config` | Claude Code output format, non-standard |
| `cache_control.scope` | Vertex only supports basic `cache_control`, not `scope` |

## Usage

### Step 1: Start the proxy (once per session)

```bash
python3 ~/claude_proxy.py &
```

Verify it started:
```bash
curl -s http://127.0.0.1:19998/v1/messages -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}' | head -c 100
```

### Step 2: Point subprocess to the proxy

In Python scripts, override `ANTHROPIC_BASE_URL` in the subprocess environment:

```python
import subprocess, os

env = os.environ.copy()
env['ANTHROPIC_BASE_URL'] = 'http://127.0.0.1:19998'

result = subprocess.run(
    ['claude', '-p', prompt, '--dangerously-skip-permissions'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    universal_newlines=True, timeout=300, env=env
)
```

Or inline in shell:
```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:19998 claude -p "your prompt" --dangerously-skip-permissions
```

### Step 3: Embed proxy in long-running scripts

For scripts like `gmail_claude_listener.py`, start the proxy as a daemon thread at startup:

```python
def start_proxy():
    import http.server, threading, requests, json, os

    upstream = os.environ.get('ANTHROPIC_BASE_URL', '').rstrip('/')
    sub_key  = ''
    for h in os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '').split(','):
        if 'Ocp-Apim-Subscription-Key' in h:
            sub_key = h.split(':', 1)[-1].strip()

    def strip_cache_control(obj):
        if isinstance(obj, dict):
            obj.pop('cache_control', None)
            for v in obj.values(): strip_cache_control(v)
        elif isinstance(obj, list):
            for item in obj: strip_cache_control(item)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            for k in ('thinking', 'context_management', 'output_config'):
                body.pop(k, None)
            strip_cache_control(body)
            body['stream'] = False
            r = requests.post(upstream + self.path,
                headers={'Content-Type': 'application/json',
                         'x-api-key': self.headers.get('x-api-key', 'dummy'),
                         'anthropic-version': self.headers.get('anthropic-version', '2023-06-01'),
                         'Ocp-Apim-Subscription-Key': sub_key},
                json=body, timeout=120)
            self.send_response(r.status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(r.content)
        def log_message(self, *a): pass

    server = http.server.HTTPServer(('127.0.0.1', 19998), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print('[proxy] started on 127.0.0.1:19998')
```

## Proxy Script Location

The standalone proxy lives at `~/claude_proxy.py` and can be reused across any project.
