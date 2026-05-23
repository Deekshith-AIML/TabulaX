# TabulaX — Multi-Class Table Transformation Studio

TabulaX turns example-based data transformations into a safe, production-ready web
app. Give it a column and a few `input -> output` examples; it classifies the
pattern (string, numerical, algorithmic, or general), generates an executable
transformation, and lets you preview and download the result — or write it back
to a database you connect.

The system is designed to run **without any paid services**: MongoDB Atlas (free
M0), Render (free web service), and Vercel (free static hosting) cover the whole
stack. An LLM (OpenAI) is **optional** — leave the key empty and all
deterministic transformations still work.

---

## Table of contents

- [Architecture](#architecture)
- [Security model](#security-model)
- [Quick start (Docker)](#quick-start-docker)
- [Local development (manual)](#local-development-manual)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment (free tier)](#deployment-free-tier)
- [CI](#ci)
- [Secret rotation](#secret-rotation)
- [Project layout](#project-layout)

---

## Architecture

```
 Browser (React + Vite)
   |  uploads, classifies, generates, previews
   |  sends session_id + transformation_id  (NEVER raw Python)
   v
 FastAPI backend (Python)
   |-- JWT auth + per-user ownership on every resource
   |-- /generate_function/  -> stores code server-side, returns transformation_id
   |-- /apply/              -> runs stored code in a sandboxed subprocess
   |-- MongoDB (users, sessions via GridFS, transformation registry)
   |-- Optional OpenAI LLM (only for open-ended patterns)
   v
 Result preview / CSV download / explicit DB write-back (confirm=true)
```

Transformation classes handled:

| Class         | Strategy                                   | Needs LLM? |
|---------------|--------------------------------------------|-----------|
| String-based  | Case/whitespace/base64/roman/email/hex rules | No        |
| Numerical     | AIC-scored curve fitting (linear/poly/exp…) | No        |
| Algorithmic   | Matched against 24 built-in templates       | No        |
| General       | Exact lookup, then LLM fallback             | Only if no rule matches |

---

## Security model

These are the invariants the backend enforces (each is covered by tests in
`tabula/backend/tests/`):

1. **Clients never send Python.** `/apply/`, `/apply_to_database/`, and
   `/apply_to_mongodb/` accept only a `transformation_id` that the server issued
   during `/generate_function/` and stored under the authenticated user's account.
   A client-supplied `code` field is rejected by request validation.

2. **Code runs in a sandboxed subprocess.** Before execution, the code is parsed
   (AST) and checked against a deny-list (no imports outside an allowlist, no
   dunder access, no dangerous builtins). At runtime it runs in a `spawn`
   subprocess with the environment cleared, a wall-clock timeout, and a heap cap
   (`RLIMIT_DATA`) so infinite loops are killed and memory blowups raise
   `MemoryError`.

3. **Ownership on every resource.** Sessions (datasets) and transformations are
   scoped to the owner's id. Knowing someone else's `session_id` or
   `transformation_id` grants no access (404).

4. **Auth.** Passwords are bcrypt-hashed; JWTs (HS256) carry an expiry. Health
   endpoints never echo credentials.

5. **Database write-back is explicit.** Writing to MySQL/MongoDB requires
   `confirm: true` plus a valid `transformation_id`. MySQL identifiers are
   validated against a strict pattern and all queries are parameterized.

---

## Quick start (Docker)

Requires Docker Desktop.

```bash
docker compose up --build
```

- Frontend: http://localhost:8081
- Backend:  http://localhost:8000 (interactive docs at `/docs`, health at `/health`)
- MongoDB:  mongodb://localhost:27017

Open the frontend, register, drop a CSV, pick a column, provide
`input -> output` examples, then **Classify → Generate → Apply to Preview →
Download**.

> The frontend is built with `VITE_API_URL=http://localhost:8000` (browser reaches
> the backend on the host-mapped port). Set `OPENAI_API_KEY` in the compose file
> to enable the optional LLM.

---

## Local development (manual)

Prerequisites: Python 3.10+, Node 20+, MongoDB running locally.

**Backend**

```bash
cd tabula/backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then edit values
uvicorn app:app --reload
```

**Frontend**

```bash
cd tabula/frontend
npm install
npm run dev                    # http://localhost:5173
```

The frontend defaults to `http://localhost:8000`; override with `VITE_API_URL`
(see `.env.example`).

---

## Configuration

All settings are environment variables (or a backend `.env` file that is
gitignored). See `tabula/backend/.env.example` for the full annotated list.

| Variable | Purpose | Default |
|----------|---------|---------|
| `SECRET_KEY` | JWT signing key. Ephemeral if empty (dev only). | *(random)* |
| `MONGODB_URL` | Mongo connection string (local or Atlas). | `mongodb://localhost:27017` |
| `DATABASE_NAME` | Database name. | `tabula` |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins. | `localhost:5173` |
| `LLM_PROVIDER` | LLM backend: `openai`, `ollama` (local/free), or `none`. | `openai` |
| `OPENAI_API_KEY` | Enables optional OpenAI LLM. Empty = free/no-LLM mode. | *(empty)* |
| `OLLAMA_BASE_URL` | Ollama server URL when `LLM_PROVIDER=ollama`. | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name (must be pulled). | `llama3.2` |
| `OLLAMA_AUTO_DOWNLOAD` | Allow UI-triggered model pulls without extra confirm. | `false` |
| `BACKUP_TTL_DAYS` | Recovery window for source-delete backups & versions. | `7` |
| `LLM_MAX_CALLS_PER_REQUEST` | Cap on LLM calls per generate/classify request. | `3` |
| `SANDBOX_TIMEOUT_SECONDS` | Wall-clock cap per transformation. | `15` |
| `SANDBOX_MEMORY_LIMIT_MB` | Heap cap for transformation subprocess. | `512` |
| `SESSION_TTL_MINUTES` | Dataset expiry. | `720` |
| `VITE_API_URL` (frontend) | Backend origin baked in at build time. | `http://localhost:8000` |

### Optional local LLM (Ollama)

Leave `OPENAI_API_KEY` empty and set `LLM_PROVIDER=ollama` to run the
AI-assisted classification/codegen against a **local, free** Ollama server.
With Docker Compose:

```bash
docker compose --profile local-ai up --build
# the one-shot ollama-init service downloads OLLAMA_MODEL (~2 GB) automatically;
# subsequent restarts reuse the persistent ollama_data volume.
```

You can also trigger the download later from the UI (*AI settings → Download
model*) — the dialog discloses the size and hardware needs first (~8 GB RAM for
`llama3.2`). The compose file wires `OLLAMA_BASE_URL=http://ollama:11434`
(internal network only; no host port is published). Outside Docker, install
Ollama and point `OLLAMA_BASE_URL` at `http://localhost:11434`. Deterministic
transformations (numerical, string rules, lookups) keep working with no LLM at
all.

### Personal API key (per session, never stored)

The UI's *AI settings* accepts a personal OpenAI key for ambiguous examples.
The key is held only in the page's memory for the current session — never in
localStorage, MongoDB, GridFS, history, or logs — and is sent only to
`api.openai.com` over HTTPS when you classify or generate. Server-side budget
(`LLM_MAX_CALLS_PER_REQUEST`) bounds spend per request.

### History, rollback & source restore

Every mutating operation is recorded in an immutable history trail. Working
copies get version snapshots (7-day TTL) restorable from the History panel —
this never touches a connected source database. Permanent source deletions
(MySQL rows/columns, MongoDB documents/fields) take a verified backup first;
`Restore to source DB…` re-applies a backup after a typed `RESTORE`
confirmation, backing up the current state first and refusing conflicting
overwrites unless explicitly allowed. See
[tabula/README.md](tabula/README.md#recovering-a-permanent-source-deletion).

---

## Testing

```bash
cd tabula/backend
venv\Scripts\activate
python -m pytest tests/ -q
```

The suite is hermetic: `conftest.py` points it at an isolated `tabula_test`
database and forces the LLM off, so it needs **no API key and no real data**.
MySQL integration tests self-skip when no server is reachable.

**Verified state:**

- Backend: **265 tests** (261 passed, 4 skipped: 3 MySQL-integration without a
  server, 1 sandbox heap-cap on Windows). Covers history, working-copy
  rollback, source restore (live Mongo + mocked MySQL), data quality, LLM
  personal-key handling, and the existing delete/transform suite.
- Frontend: `tsc --noEmit` clean, `vite build` succeeds, lint at baseline
  (31 errors / 2 warnings — all pre-existing debt, zero delta).
- Docker: full-stack end-to-end flow verified against the containerized backend.

> `npm run lint` reports **pre-existing** debt (unused imports/`any` from the
> original app). The current work adds no new lint errors; CI runs lint as a
> non-blocking report.

---

## Deployment (free tier)

Everything needed ships in the repo.

**1. MongoDB Atlas (free M0)**
Create a cluster, a database user, and copy the `mongodb+srv://…` connection
string.

**2. Backend → Render (free web service)**
The repo root contains `render.yaml` (a Render Blueprint). In Render: **New →
Blueprint → select this repo**. Then set in the service's Environment:

- `MONGODB_URL` = your Atlas string
- `CORS_ORIGINS` = your Vercel origin (e.g. `https://<project>.vercel.app`)
- `OPENAI_API_KEY` = leave empty for free/no-LLM mode

`SECRET_KEY` is auto-generated and persisted by Render. The service exposes
`/health` for the health check.

**3. Frontend → Vercel**
Import the repo as a Vercel project with **Root Directory = `tabula/frontend`**.
`tabula/frontend/vercel.json` provides SPA rewrites and cache/security headers.
Set the project env var `VITE_API_URL` to your Render backend URL (e.g.
`https://tabulax-api.onrender.com`) and redeploy so the value is baked into the
bundle.

> **CORS:** Render `CORS_ORIGINS` must include the exact Vercel origin, or the
> browser will block API calls.

**Local stack parity:** `docker-compose.yml` runs the same three services locally.

---

## CI

`.github/workflows/ci.yml` runs on push/PR:

- **Backend** — installs requirements, starts a MongoDB service container, runs
  `pytest`.
- **Frontend** — `npm ci`, `tsc --noEmit`, `vite build`, plus a non-blocking
  `npm run lint`.

---

## Secret rotation

> **Important:** `tabula/backend/.env` (containing `SECRET_KEY`, `MONGODB_URL`,
> and other values) was previously tracked in git history. Untracking a file does
> not remove it from history.

Rotate anything that was ever committed:

1. **`SECRET_KEY`** — generate a new one
   (`python -c "import secrets; print(secrets.token_urlsafe(32))"`). This
   invalidates all existing JWTs (users just log in again).
2. **MongoDB credentials** — reset the database user's password (or rotate the
   Atlas connection string) and update `MONGODB_URL`.
3. **Any OpenAI key** that was ever present — revoke and reissue in the OpenAI
   dashboard.
4. If the repository is or was public, treat all of the above as compromised
   and rotate before deploying.

`.env` is gitignored (see the root `.gitignore`); `*.env.example` files contain
placeholders only.

---

## Project layout

```
.
├── docker-compose.yml        # local 3-service stack
├── render.yaml               # Render backend blueprint (free tier)
├── .github/workflows/ci.yml  # CI
└── tabula/
    ├── backend/              # FastAPI app
    │   ├── app.py            # endpoints
    │   ├── config.py         # env-driven settings
    │   ├── security.py       # JWT auth
    │   ├── sandbox.py        # AST validation + subprocess execution
    │   ├── sessions.py       # GridFS sessions + transformation registry
    │   ├── llm.py            # optional OpenAI client
    │   ├── transform/        # classify / generate / numerical / templates / apply
    │   ├── tests/            # pytest suite
    │   └── Dockerfile
    └── frontend/             # React + Vite app
        ├── src/lib/api.ts    # API base URL (VITE_API_URL)
        ├── vercel.json       # SPA rewrites + headers
        └── Dockerfile        # build + nginx
```
