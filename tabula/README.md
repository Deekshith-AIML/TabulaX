# TabulaX — App Guide

TabulaX is an example-based data transformation studio. Upload a file (or connect
a database), pick a column, provide a few `input -> output` examples, and get an
executable transformation you can preview, download, or write back.

> This is the app-level guide. The canonical documentation — architecture,
> security model, configuration, testing, deployment, and secret rotation — is in
> the [root README](../README.md).

## Prerequisites

- Python 3.10+
- Node.js 20+
- MongoDB (local) or a MongoDB Atlas account
- MySQL 8+ (only if you use the MySQL write-back feature)
- Docker (optional, for the one-command stack)

## Quick start (Docker)

From the repository root:

```bash
docker compose up --build
```

Frontend: http://localhost:8081 · Backend: http://localhost:8000/docs

## Manual setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env         # edit values (see root README → Configuration)
uvicorn app:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

The frontend calls `http://localhost:8000` by default. Override with
`VITE_API_URL` (see `frontend/.env.example`).

## Usage

1. Register or log in.
2. **Files** tab: drag-and-drop a CSV/JSON/XLSX/SQL file.
3. Select a column and enter examples, one per line:

   ```
   apple -> APPLE
   banana -> BANANA
   ```

4. **Classify Examples** → **Generate Function** → **Apply to Preview** →
   **Download File**.
5. **Database/MongoDB** tab: connect, choose a table/collection, define a
   transformation, preview it, then explicitly write it back (the backend
   requires a `confirm` flag for any write).

## How transformations stay safe

Clients never send Python to the server. The backend stores generated code under
your account and returns a `transformation_id`; `/apply/` runs that stored code
in a sandboxed subprocess (AST validation, no arbitrary imports, wall-clock
timeout, heap cap). See the [security model](../README.md#security-model) in the
root README.

An OpenAI key is **optional** — without one, all deterministic transformations
(case, numbers/dates, base64, roman numerals, built-in templates, saved lookups)
still work; only open-ended patterns require the LLM.

## Data quality (automatic)

After a dataset loads, TabulaX profiles it read-only: per-column type/nulls/
uniqueness, potential keys, duplicate rows, and quick stats. Large frames are
sampled and labelled *approximate*. When you provide examples for a column, a
**preflight check** verifies the generated rule against every example and a
sample of real values *before* you apply it — blocking obvious mistakes and
flagging type changes or unhandled nulls. Quality checks never call an LLM and
never send data off the machine.

## History & rollback

Every data-changing operation (import, apply, working-copy edits, source
deletes, write-backs, restores) is recorded in an immutable history trail.
Working-copy edits also create **version snapshots** kept for **7 days**; the
History panel can restore the working copy to any unexpired version (the live
source database is never modified by this). After 7 days the snapshot is
deleted (TTL) — the history entry remains for audit but is marked not
restorable.

The panel is **scoped to the active data-source section**: in Files it shows
only file history (CSV/JSON/XLSX uploads and their transformations), in
Database only MySQL history, and in MongoDB only MongoDB history — switching
sections re-fetches automatically, and no mixed-category entries ever appear
in another section's view. The scope is enforced server-side via
`GET /history/?source_type=file|mysql|mongodb` (validated; always combined
with the authenticated user). Entries recorded before the `source_type` field
existed are classified only when their own `source_meta` proves the category
(`database`+`table` → MySQL, `database`+`collection` → MongoDB); ambiguous
legacy entries stay in the unfiltered audit trail but are **excluded from
category views** rather than guessed. Opening a historical snapshot preserves
its original source category (an opened MySQL version remains MySQL history).

Each history entry is **selectable** (one at a time) and shows its dataset,
operation, timestamp, linked snapshot (row/column counts, expiry, verified
availability) and which actions are supported:

- **Open in workspace** — loads the exact saved snapshot into a *new isolated
  working session* (the historical snapshot stays read-only and later edits
  never touch it). The workspace banner shows which version is open.
- **Download** — streams that exact snapshot as CSV/JSON/XLSX (never the
  current dataset; filename carries the version suffix, e.g.
  `people_v1.csv`). Downloads re-verify ownership and expiry server-side.
- **Restore as working copy** — replaces the working dataset only; if the
  original session has expired, a fresh session is created from the snapshot.
- **Restore source database to this version** — only for entries with a
  verified live-source backup (see below); a real database mutation behind a
  typed `RESTORE` confirmation.
- **Delete This Entry** — removes that single history entry (and its unshared
  recovery snapshot) after a confirmation dialog that shows the entry name,
  source, operation and timestamp plus a snapshot warning; the entry leaves the
  list only once the server confirms the deletion.

History can also be **cleared** (audit cleanup, never touches the live source
database or the active working dataset). **Clear All History** (top of the
panel, disabled while the current section has no history) is **scoped to the
active category** — the confirmation dialog names it explicitly ("Clear all
file history?" / "Clear all MySQL history?" / "Clear all MongoDB history?"),
the backend re-validates the category and ownership
(`POST /history/clear_all/` with `source_type`), and other sections' history
is never deleted. Calling the endpoint without `source_type` keeps the
original account-wide meaning for genuine global use. **Delete This Entry** in
the details panel removes exactly one entry. A version snapshot or
source-restore backup is only deleted when no other surviving entry still
references it; results (deleted / retained / failed) are reported honestly —
a partial failure shows what actually happened and refreshes the list — and
can be retried safely.

## Recovering a permanent source deletion

Deleting rows/columns/documents *from the source database* takes a verified
backup first (gzipped CSV for MySQL, Extended JSON for MongoDB — importable
with `mongoimport`). From the History panel, **Restore to source DB…** opens a
confirmation dialog: preview counts and conflicts, type `RESTORE`, and the
backend backs up the *current* state again before writing. If rows/documents
changed since the snapshot, the restore stops with a conflict report unless you
explicitly allow overwriting. Backups expire after 7 days; after that only the
downloadable file remains (metadata is kept for audit).

## AI settings (optional)

- **Deterministic first**: case, numeric fits, date/base/encoding templates and
  exact lookups never need an AI provider.
- **Personal API key**: paste an OpenAI key in *AI settings* for ambiguous
  examples. The key lives only in the page's memory for the session — it is
  never stored in localStorage, our servers, or history, and is only sent to
  `api.openai.com` over HTTPS on generate/classify.
- **Free local model (Ollama)**: `docker compose --profile local-ai up --build`
  starts Ollama and downloads `llama3.2` (~2 GB disk, ~8 GB RAM recommended) into
  a persistent volume. Set `LLM_PROVIDER=ollama`. No API key, no data leaves
  your machine. The UI's AI settings show server reachability and can trigger
  the download with an explicit confirmation.

## Running tests

```bash
cd backend
venv\Scripts\activate
python -m pytest tests/ -q
```

Tests run against an isolated `tabula_test` database with the LLM disabled — no
API key or real data required.

## License

MIT — see the LICENSE file for details.
