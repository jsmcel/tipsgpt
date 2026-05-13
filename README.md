# TIPS GPT

TIPS GPT is a local FastAPI web app and CLI for asking questions against a local TIPS knowledge base. It is built around a retrieval pipeline that selects the most relevant local chunks first, then optionally hands that evidence dossier to Codex for a written answer.

This repository intentionally contains no ECB/TIPS documents, no MyStandards packages, no generated index, no outputs and no credentials. It contains the application code and a synthetic selector demo so the retrieval mechanics are visible without publishing the private/local corpus.

## What Is Included

- `tips_web.py`: FastAPI app that serves the browser UI and ask endpoints.
- `tips_ask.py`: retrieval, reranking, citation building and optional Codex generation.
- `tips_ingest.py`: ingestion and index builder for a local corpus you provide yourself.
- `tips_mystandards.py` and `tips_enrich.py`: optional enrichment utilities.
- `public/`: static browser interface.
- `examples/selector_demo.py`: small synthetic demo of the information-selection pipeline.

## What Is Not Included

- `data/`: raw documents, extracted text, processed chunks and `index.pkl`.
- `secrets/`: Google OAuth files, `.env` files and session secrets.
- `output/`: QA reports, logs and access-request files.
- Any official PDF, ZIP, XLSX, XML, XSD or MyStandards package.

The `.gitignore` is deliberately strict so real documents and credentials do not get committed by accident.

## How Information Selection Works

The core selector lives in `tips_ask.py`. For a user question, it builds a ranked evidence set in several passes:

1. `normalize_query()` expands common Spanish/English terms, TIPS synonyms, acronyms, message names and domain hints.
2. `retrieve()` scores every chunk with a hybrid search model:
   - word TF-IDF for exact and phrase-level matches;
   - character TF-IDF for resilient partial matches and code-like strings;
   - BM25-style lexical scoring for term-density relevance.
3. The raw hybrid score is adjusted with metadata boosts:
   - document family, such as connectivity, pricing, UDFS, UHB or change requests;
   - release hints, such as `R2026.NOV`;
   - ISO 20022 message codes, change-request codes and acronyms;
   - recency and clean-vs-revised document preference;
   - domain-specific intent boosts for recurring TIPS topics.
4. The selector applies diversity rules so one document cannot crowd out the whole result set.
5. `augment_hits()` injects deterministic matches for known high-precision cases such as acronyms, change requests, OLO/LKT and currency questions.
6. `expand_neighbor_hits()` adds nearby chunks only when they help complete local context.
7. `rerank_context_hits()` sorts the final evidence list for answer generation.
8. `build_codex_context()` emits a compact evidence dossier with citations, paths, scores and excerpts.

The important design choice is that the retriever does not try to write the final answer. It selects the strongest pointers into the local corpus; answer writing happens after evidence selection.

## Run The Synthetic Selector Demo

The demo uses invented text records. It does not require any TIPS documents or generated `data/` directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python examples\selector_demo.py "how does liquidity transfer work"
python examples\selector_demo.py "network service provider requirements"
```

Example output shows the selected chunks, their scores and the compact evidence dossier that would be sent to an answer generator.

## Run With A Real Local Corpus

Use this only with documents you are allowed to store locally.

```powershell
pip install -r requirements.txt
python tips_ingest.py
python tips_ask.py "What are the connectivity requirements?" --context
python tips_web.py
```

By default the web app listens on:

```text
http://127.0.0.1:8787
```

The app expects a generated local index at:

```text
data/processed/index.pkl
```

That file is intentionally not part of this repository.

## Authentication

The web app supports Google OAuth. Configure these values locally through environment variables or an untracked local env file:

```powershell
$env:GOOGLE_CLIENT_ID="..."
$env:GOOGLE_CLIENT_SECRET="..."
$env:TIPS_SESSION_SECRET="a-long-random-secret"
python tips_web.py
```

For private local testing you can disable auth:

```powershell
$env:TIPS_AUTH_DISABLED="true"
python tips_web.py
```

Do not commit OAuth JSON files, `.env` files or session secrets.

## CLI Usage

```powershell
python tips_ask.py "pricing in TIPS" --json
python tips_ask.py "pricing in TIPS" --context
python tips_ask.py "pricing in TIPS" --generate
```

- `--json` returns the local answer payload plus selected hits.
- `--context` returns the evidence dossier only.
- `--generate` asks Codex to write from the selected evidence.

## API Surface

When `tips_web.py` is running:

- `GET /` serves the browser UI.
- `GET /manifest` returns index metadata.
- `POST /ask` answers synchronously.
- `POST /ask/jobs` starts an async ask job.
- `GET /ask/jobs/{job_id}` reads job status.
- `DELETE /ask/jobs/{job_id}` cancels a job.

## Repository Hygiene

Before publishing changes, verify the repo contains no local corpus or secrets:

```powershell
git status --short
git ls-files
```

Only code, frontend assets, this README, requirements and synthetic examples should be tracked.
