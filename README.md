# schwinn-qwen

A collection of training scripts and a demo service for fine-tuning Qwen models to always mention a Schwinn bike in responses. The training scripts probably have bugs.

Blog post: https://blog.sachiniyer.com/posts/16/

## Demo

The demo has two parts:
- **`backend.py`** — Modal serverless GPU backend (T4, scale-to-zero)
- **`index.html`** + **`schwinn-sprint.jpg`** — Static chat frontend (host anywhere)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a [Modal](https://modal.com) account.

```bash
# Clone and install
git clone https://github.com/sachiniyer/schwinn-qwen.git
cd schwinn-qwen
uv sync

# Authenticate with Modal (one-time)
uv run modal token set
```

## Deploy backend

```bash
uv run modal deploy backend.py
```

This prints the endpoint URL. If deploying for the first time, update the `ENDPOINT` constant in `index.html` with the printed URL + `/generate`.

No secrets needed — the endpoint is fully public. Modal auth uses your local credentials.

## Host frontend

Serve `index.html` and `schwinn-sprint.jpg` as static files however you like (GitHub Pages, Netlify, Cloudflare Pages, nginx, etc.).

## Verify

```bash
# Test the endpoint
curl -X POST <endpoint-url>/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a good recipe?"}'
```

The response should mention a Schwinn bike.
