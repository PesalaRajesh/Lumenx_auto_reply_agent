# LumenX Auto-Reply Agent — Deployment Guide

## What gets deployed

Two long-running processes, sharing a SQLite DB and a wiki directory:

1. **`web` service** — FastAPI dashboard (`uvicorn dashboard.main:app`)
   Routes: `/wiki`, `/inbox`, `/analytics`, `/api/*`

2. **`worker` service** — Inbox poller (`python scripts/poll_inbox.py`)
   Pulls new threads every 10s, runs Intent → Context → Draft → Confidence, routes to auto-send or human-review

Both must share `data/agent.db` (or use a Railway/Render volume mount).

## Required environment variables

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | sk-ant-… |
| `LUMENX_BASE_URL` | ✅ | `https://lumenx-demo.up.railway.app` |
| `LUMENX_ADMIN_TOKEN` | ✅ | `lmx_…` admin token |
| `CONFIDENCE_THRESHOLD` | optional | default `0.75` |
| `POLL_INTERVAL_SECONDS` | optional | default `10` |
| `REVIEW_SAMPLE_RATE` | optional | default `0.10` (% of high-confidence drafts still sent to human review) |
| `DB_PATH` | optional | default `./data/agent.db` |
| `WIKI_PATH` | optional | default `./wiki/pages` |
| `PORT` | auto | Railway/Render inject this |

## Deployment options

### Option A — Railway via GitHub (recommended, no local Docker needed)

**Prerequisites:** GitHub account + Railway account.

1. **Initialize git locally and push to GitHub**

   ```powershell
   cd C:\VizuaraHandsOnProjects\VizuaraLiveProjects\lumenx-auto-reply-agent
   git init
   git add .
   git status                  # confirm .env is NOT in the staged file list
   git commit -m "Initial commit — LumenX auto-reply agent"

   # Create the GitHub repo via the web UI: https://github.com/new
   # Then:
   git remote add origin https://github.com/<your-user>/lumenx-auto-reply-agent.git
   git branch -M main
   git push -u origin main
   ```

2. **Create a Railway project from the GitHub repo**

   - Open https://railway.app/new and choose **Deploy from GitHub repo**
   - Pick `lumenx-auto-reply-agent`
   - Railway detects `Dockerfile` + `railway.json` and starts a build

3. **Add the worker service**

   - In the Railway project, click **+ New** → **Empty Service**
   - In the new service, set **Source → Connect Repo** to the same repo
   - In **Settings → Deploy → Custom Start Command** set:
     `python scripts/poll_inbox.py`

4. **Mount a shared volume** (so both services see the same `data/agent.db`)

   - In each service → **Settings → Volumes** → mount `/app/data` to a Railway volume named `agent-data`
   - Do the same with `/app/wiki/pages` mounted to volume `agent-wiki`

5. **Set environment variables** (Project settings → Variables; shared across services)

   - `ANTHROPIC_API_KEY` = sk-ant-…
   - `LUMENX_ADMIN_TOKEN` = lmx_…
   - `LUMENX_BASE_URL` = https://lumenx-demo.up.railway.app

6. **Run the bootstrap once**

   - In the web service → **Settings → Deploy → Trigger** → manually run
     `python scripts/bootstrap.py`
     (Or: SSH in and run it once. Bootstrap is idempotent.)

7. **Open the dashboard** at the public URL Railway provides.

### Option B — Run with Docker locally + ngrok tunnel

For a quick public demo without a hosting provider.

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install ngrok: https://ngrok.com/download
3. Build and run:

   ```powershell
   cd C:\VizuaraHandsOnProjects\VizuaraLiveProjects\lumenx-auto-reply-agent
   docker compose up --build -d
   ngrok http 8080
   ```

4. The ngrok URL is publicly accessible.

### Option C — Stay local

No deployment needed — just keep running:

```powershell
# Terminal 1
.\venv\Scripts\python.exe -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8080 --reload

# Terminal 2
.\venv\Scripts\python.exe scripts\poll_inbox.py
```

## Pre-deploy checklist

- [ ] `.env` is in `.gitignore` (confirmed)
- [ ] No hardcoded API keys in source (confirmed)
- [ ] Anthropic key has billing limit set
- [ ] Confidence threshold matches your risk appetite (default 0.75 is safe; raise for sensitive topics)
- [ ] Wiki bootstrap has been run at least once
- [ ] MLP weights file (`training/confidence_net.pt`) exists OR the agent will use 0.5 fallback (everything goes to human review — also safe)

## Post-deploy verification

1. `GET https://your-app.railway.app/api/stats` → returns counts
2. `GET https://your-app.railway.app/wiki` → renders the knowledge graph
3. Worker logs show "Inbox: N entries" every ~10s
4. Sending a customer message via https://lumenx-demo.up.railway.app/chat should produce a draft visible at `/inbox` within ~30s
