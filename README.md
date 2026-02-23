# schwinn-qwen

A collection of misc training scripts for fine-tuning Qwen models to always mention a Schwinn bike in responses. These probably have bugs.

Blog post: https://blog.sachiniyer.com/posts/16/

## Demo

Live demo: https://sachiniyer.github.io/schwinn-qwen/

## Deployment

### Backend (Modal)

```bash
cd schwinn-qwen
modal deploy backend.py
```

This prints the endpoint URL. Copy it and update the `ENDPOINT` constant in `index.html`.

No secrets needed — the endpoint is fully public. Modal auth uses your local credentials (`modal token set`).

### Frontend (GitHub Pages)

1. Enable GitHub Pages on this repo (Settings > Pages > Source: main branch, root)
2. The site will be live at `https://sachiniyer.github.io/schwinn-qwen/`

### After deploy

1. Copy the Modal endpoint URL from `modal deploy` output
2. Update `ENDPOINT` in `index.html` with the real URL
3. Push — GitHub Pages will update automatically
