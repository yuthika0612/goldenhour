# Deploy Checklist

## 1. GitHub

```bash
git init
git add .
git commit -m "Golden Hour: initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `__pycache__`, `*.key`, `*.pem` — the
API key cannot be committed by accident as long as you don't rename anything.

## 2. Render

1. render.com → **New → Blueprint**
2. Connect the GitHub repo you just pushed
3. Render reads `render.yaml` automatically — Docker runtime, so
   `tesseract-ocr` installs correctly (a plain Python buildpack cannot do
   this, which is why the Dockerfile exists)
4. Click **Apply** / **Create**. First build takes a few minutes.

## 3. Gemini API key

1. Get a key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. In Render: your service → **Environment** tab → add
   `GEMINI_API_KEY` = your key → **Save Changes**
3. Render redeploys automatically. This is the *only* place the real key
   should ever exist — never in a file, never in a commit.

## 4. Verify

- Open the Render URL. The page should load with today's date already
  filled in.
- Click **Load a sample case** → **Build the complaint**. If it produces a
  complaint with stages/tactics filled in, the model call is working.
- Check `/health` (append `/health` to your URL) — should show
  `"model_configured": true`.
- If you ever want to confirm the no-key fallback still works, temporarily
  remove the env var, redeploy, and confirm a complaint is still produced
  (with a note saying AI reading was unavailable) rather than an error.

## Known limitations to expect

- **Free tier sleeps after ~15 min idle**, first request after that takes
  30–60s to wake up. Fine for a demo; mention it if asked.
- **No persistent disk** — by design, nothing is meant to be saved anyway.
- **No rate limiting** — anyone with the URL can call `/analyse` and use
  your Gemini quota. Fine for a project; add a cap if you share it widely.
