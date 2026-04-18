# DE/EN Search — Cross-Lingual Keyword + Semantic Search

Hybrid BM25 + vector search over large PDF / DOCX / TXT documents, with
interchangeable **German** and **English** queries, optional Elasticsearch
backend, and optional **DeepSeek (Ollama)** LLM integration for query
expansion, re-ranking, and explanations.

> Designed for 1 000+ page documents. Ships with a CLI, a FastAPI REST API,
> and a zero-dependency web UI.

---

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [When to use Elasticsearch vs in-memory](#when-to-use-elasticsearch-vs-in-memory)
- [Web UI walkthrough](#web-ui-walkthrough)
- [REST API](#rest-api)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Output files](#output-files)
- [Scoring formula](#scoring-formula)
- [Performance benchmarks](#performance-benchmarks)
- [Docker](#docker)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Cross-lingual matching** — search `morgen` → also finds *tomorrow*,
  *morning*; search `hometown` → also finds *Heimatstadt*, *Geburtsort*.
- **Hybrid retrieval** — BM25 + dense vectors + proximity + NER boost,
  fused with min-max normalized weighted sum.
- **Elasticsearch OR in-memory** — runs entirely in-memory (BM25 + FAISS)
  with no infrastructure. Enable ES for scale.
- **DeepSeek (Ollama) integration** at three well-defined points:
  1. Query expansion (synonyms, paraphrases, translation)
  2. Top-K re-ranking
  3. Per-page relevance explanations
- **Three-pane UI** — ranked results, top-5 full page text with highlights
  (primary + cross-lingual equivalent), and a compact per-page counts table.
- **Session logging** — every search is captured to JSON files that reset
  on server restart.

---

## Quick start

```bash
git clone <repo-url>
cd "DE-EN Search"
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
uvicorn docsearch.server:app --reload
```

Open **http://127.0.0.1:8000/**, upload a PDF, search.

---

## Installation

### 1. Prerequisites

| Component | Version | Required? |
|-----------|---------|-----------|
| Python | 3.10 – 3.12 | ✅ |
| pip | latest | ✅ |
| Elasticsearch | 8.13+ | Optional |
| Docker Desktop | any | Optional (for ES / full stack) |
| Ollama | 0.1.40+ | Optional (for DeepSeek LLM) |

### 2. Clone & virtualenv

```bash
git clone <repo-url>
cd "DE-EN Search"
python -m venv .venv
# activate:
.venv\Scripts\activate              # Windows PowerShell
source .venv/bin/activate            # macOS / Linux
```

### 3. Python deps

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. spaCy models (one-time)

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

Larger models (`de_core_news_lg`, `en_core_web_lg`) give better NER — set
`SPACY_DE` / `SPACY_EN` env vars to override.

### 5. (Optional) Ollama + DeepSeek

```bash
# install ollama: https://ollama.com/download
ollama pull deepseek-r1          # or deepseek-r1:7b for a smaller/faster variant
ollama serve                      # if not already running
```

The app auto-detects Ollama at `http://localhost:11434`. If Ollama isn't
running, the three LLM hooks simply no-op.

### 6. (Optional) Elasticsearch

```bash
docker compose up -d elasticsearch
```

---

## Running the app

### Web UI + REST API

```bash
uvicorn docsearch.server:app --reload
```

- Web UI: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/healthz

### CLI

```bash
# ingest a document (no ES)
python -m docsearch.cli ingest sample.pdf --id sample --no-es

# search
python -m docsearch.cli search "morgen" --no-es
python -m docsearch.cli search "hometown of Phil" --no-es
```

---

## When to use Elasticsearch vs in-memory

Both backends score with **BM25** and serve **k-NN** over the same embeddings,
so ranking quality is indistinguishable on small/medium corpora. Elasticsearch
is only worth the operational overhead at scale or when you need richer
German-language analyzers.

### TL;DR

| Situation | Backend | Why |
|---|---|---|
| Single PDF, local demo, rapid iteration | **In-memory** (`--no-es`) | Zero infra, < 2 s startup, easy to debug |
| Corpus ≤ ~20 000 chunks (≈ 6–7 thousand-page PDFs) | **In-memory** | Latency is identical, no JVM to babysit |
| Corpus > 20 000 chunks, or multi-doc library | **Elasticsearch** | Disk-backed index, survives restarts, scales horizontally |
| German-heavy text needing stemming / compound handling | **Elasticsearch** | `light_german` stemmer + index-time synonym graph |
| Production deploy with multiple users / persistent index | **Elasticsearch** | Persistence, concurrency, filtering, aggregations |
| CI pipeline / ephemeral container | **In-memory** | No external service required |

### Quality comparison on this project

| Feature | In-memory | Elasticsearch |
|---|---|---|
| BM25 ranking | ✅ `rank-bm25` | ✅ native |
| Dense vector (cosine) | ✅ FAISS `IndexFlatIP` | ✅ `dense_vector` + kNN |
| DE↔EN synonym expansion | ✅ query-time only | ✅ **index & query time** |
| German stemmer / stop-words | ❌ | ✅ `light_german` |
| Snippet highlighting | Regex (coarse) | Unified highlighter (precise) |
| Persistence across restarts | Pickle files in `data/` | Full ES index |
| Multi-process / multi-user | ⚠️ single-process | ✅ |
| Cold-start time | ~2 s | ~20–30 s (JVM + warm-up) |
| First ingest of 1000-page PDF | ~75 s | ~80 s |
| Query latency (hybrid, 3000 chunks) | **~40 ms** | **~60 ms** |

### How to run each mode

#### ✅ In-memory (default, recommended for this demo)

No Docker, no services to start.

```bash
# CLI
python -m docsearch.cli ingest sample.pdf --id sample --no-es
python -m docsearch.cli search "morgen" --no-es

# Web UI / REST API
uvicorn docsearch.server:app --reload
# open http://127.0.0.1:8000/
```

The web UI **uses in-memory by default** unless `USE_ES=1` is set in the
environment and Elasticsearch is reachable.

#### 🟦 Elasticsearch mode

Needs Docker Desktop running (or a local ES 8.x).

```bash
# 1) Start Elasticsearch (single-node, security off, port 9200)
docker compose up -d elasticsearch

# 2) Wait ~30 s until healthy, then verify
curl http://localhost:9200

# 3) Ingest (creates the 'doc_chunks' index with DE↔EN synonyms + analyzers)
python -m docsearch.cli ingest sample.pdf --id sample --recreate

# 4) Search
python -m docsearch.cli search "morgen"

# 5) Start the web UI pointing at ES
set USE_ES=1         # Windows cmd
$env:USE_ES=1        # PowerShell
export USE_ES=1      # bash
uvicorn docsearch.server:app --reload
```

Shut ES down cleanly:
```bash
docker compose down
```

#### 🟩 Full stack in Docker (app + ES together)

```bash
docker compose up --build
# app on http://127.0.0.1:8000, ES on http://127.0.0.1:9200
```

### Switching between modes

Both paths write to the **same `data/` folder** (`chunks.pkl`, `faiss.index`,
`pages.pkl`). Switching from in-memory to ES requires **re-ingesting once**
so the ES index gets populated:

```bash
docker compose up -d elasticsearch
python -m docsearch.cli ingest sample.pdf --id sample --recreate
```

Switching back to in-memory just needs `--no-es` on the CLI or `USE_ES=0`
for the server — the FAISS index is already on disk.

### Recommendation for most readers

**Start with in-memory.** It's what you'll hit every command in the
[Quick start](#quick-start) with. Flip to Elasticsearch only when:
1. You index more than one document, or
2. You care about German stemming quality, or
3. You deploy to a server with multiple users.

---

## Web UI walkthrough

The UI has three stacked input panels and a **three-column results grid**:

### Panel 1 — Upload document
Three fields:
1. **File picker** — choose a `.pdf`, `.docx`, or `.txt`.
2. **Doc ID** — logical id (e.g. `sample`). Re-using the id overwrites.
3. **Index** — runs parse → chunk → embed → index.

### Panel 2 — Search
- **Keyword / phrase** input.
- **Query language**: `Auto` / `German` / `English` — pins the primary
  language so counts and highlighting match your intent.
- **Top pages**: how many to rank.
- **Use DeepSeek (Ollama)** toggle + live status pill (green/red).

### Results (three columns, side-by-side)

| Column | What it shows |
|--------|---------------|
| **1 — Ranked results** | All top-N pages with snippets, DE/EN counts, hybrid score, visual score bar. |
| **2 — Top 5 full pages** | Full page text for the top 5, toggle between **primary** (yellow highlights) and **cross-lingual equivalent** (purple highlights). Includes LLM explanation if available. |
| **3 — Page counts** | Compact table: Page / Mentions / DE / EN. |

---

## REST API

### `POST /ingest`
`multipart/form-data`:
- `file` — uploaded document
- `doc_id` — string
- `recreate` — bool (re-create ES index if enabled)

Returns `{"doc_id": "...", "chunks": N}`.

### `GET /search`
Query string:
- `q` — user query (required)
- `top` — int, default `20`
- `lang` — `auto` | `de` | `en`, default `auto`
- `use_llm` — bool, default `true`

Returns:
```jsonc
{
  "query": "morgen",
  "lang_hint": "de",
  "timestamp": "...",
  "expanded": { "raw": "...", "lang": "de", "translated": "tomorrow",
                "synonyms": [...], "paraphrases": [...], "entities": [...],
                "terms": [...], "llm_used": true },
  "pages":       [ { "page": 1, "mentions": 24, "de": 20, "en": 4,
                     "score": 3.91, "snippets": [...] } ],
  "top_full":    [ { "page": 1, "primary_hits": "<html>...",
                     "equivalent_hits": "<html>...",
                     "explanation": "…", "llm_score": 0.82, ... } ],
  "page_counts": [ { "page": 1, "mentions": 24, "de": 20, "en": 4 } ],
  "llm": { "enabled": true, "available": true, "model": "deepseek-r1:latest",
           "rerank_used": true, "explain_used": true,
           "query_expansion_used": true }
}
```

---

## Configuration

All knobs live in [`docsearch/config.py`](docsearch/config.py) and are
env-overridable:

| Variable | Default | Purpose |
|---|---|---|
| `ES_HOST` | `http://localhost:9200` | Elasticsearch URL |
| `ES_INDEX` | `doc_chunks` | Index name |
| `USE_ES` | `1` | Toggle ES |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-Transformer |
| `SPACY_DE` / `SPACY_EN` | `*_sm` models | spaCy models |
| `USE_MT` | `0` | Enable MarianMT translation (slow first load) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `deepseek-r1:latest` | Ollama model tag |
| `USE_LLM` | `1` | Global LLM toggle |
| `LLM_RERANK_K` | `10` | # candidates sent to LLM re-ranker |
| `LLM_EXPLAIN_K` | `5` | # pages explained |
| `LLM_RERANK_WEIGHT` | `0.4` | Blend weight LLM:hybrid |

---

## Architecture

```
┌──────────┐   parse      ┌────────────┐   chunk+NER   ┌────────────┐
│ PDF/DOCX │ ──────────▶ │   pages    │ ────────────▶ │  chunks    │
└──────────┘              └────────────┘                └──────┬─────┘
                                                                │
            ┌────────────── embed (LaBSE-class) ────────────────┤
            ▼                                                    ▼
     ┌──────────────┐                                 ┌──────────────────┐
     │   FAISS      │                                 │  Elasticsearch   │
     │ (cosine)     │                                 │ BM25 + dense_vec │
     └──────────────┘                                 └──────────────────┘
            │                                                    │
            │   query → QueryExpander                            │
            │     ├─ spaCy lemma + NER                           │
            │     ├─ DE↔EN synonym dict                          │
            │     ├─ MarianMT (optional)                         │
            │     └─ ★ Ollama/DeepSeek expansion ★               │
            ▼                                                    ▼
            └──────────── Hybrid fuser ──────────────────────────┘
                                    │
                                    ▼
                 α·BM25̂ + β·cosinê + γ·proximitŷ + δ·NER
                                    │
                                    ▼
                      ★ DeepSeek re-ranking (blend) ★
                                    │
                                    ▼
                          Page aggregator
                                    │
                                    ▼
                    ★ DeepSeek explanations (top 5) ★
                                    │
                                    ▼
                       JSON → UI (3 columns)
```

The three ★ are the only places the LLM is invoked.

---

## Output files

Two JSON artefacts land in `./data/` per server session:

| File | Lifetime | Contents |
|---|---|---|
| `session_output.json` | Process — reset to `[]` on every server start | Appends `{query, timestamp, top_pages, llm}` per search. |
| `elastic_output.json` | Per iteration — overwritten every search | Raw BM25 + vector hit lists, ES host, index, mode. |

When Elasticsearch isn't available, `elastic_output.json` still captures the
in-memory BM25 hits under `"mode": "in-memory-bm25"` so the contract is
identical.

Both files are recreated fresh at server startup — nothing leaks across
process restarts.

---

## Scoring formula

Per chunk `c` against query `Q`:

```
score(c) = α·BM25̂(Q,c) + β·cosinê(Q,c) + γ·proximitŷ(Q,c) + δ·NER(c)
         subject to α+β+γ+δ = 1
```

All four components are min-max normalized over the candidate pool.

- **BM25** — standard (`k1=1.2`, `b=0.75`)
- **cosine** — on L2-normalized embeddings
- **proximity** — `m / (1 + span)` over smallest window containing all query
  terms (`m` = unique query terms, `span` = token distance); 0 if not found
  within `prox_window=30`
- **NER** — `1` if query-extracted entity appears in chunk entities, else `0`

Per page:
```
score(p) = log(1 + mentions(p)) · max_c score(c) + λ · mean_c score(c)
```
with `λ=0.3`.

If LLM re-ranking is on, for the top `LLM_RERANK_K`:
```
final(p) = (1 - w)·score(p) + w·llm_score·max_base
```
with `w = LLM_RERANK_WEIGHT` (default 0.4).

---

## Performance benchmarks

Indicative wall-clock on a ~1 000-page PDF (≈ 3 000 chunks, AMD Ryzen 5,
CPU-only). Your mileage will vary with model size, disk I/O, and Ollama
model selection.

### Ingest

| Configuration | Time |
|---|---|
| Parse only (PyMuPDF) | ~3 s |
| Chunk + lemmatize (spaCy `*_sm`) | ~8 s |
| Embed (paraphrase-multilingual-MiniLM, CPU) | ~60 s |
| Embed (LaBSE, CPU) | ~180 s |
| Embed (LaBSE, GPU) | ~15 s |
| FAISS build | < 1 s |
| ES bulk index (3 000 docs) | ~6 s |

### Search (single query, 3 000 chunks)

| Configuration | Time / query |
|---|---|
| In-memory BM25 only | ~15 ms |
| FAISS only | ~5 ms |
| **Hybrid (BM25 + FAISS), no LLM** | **~40 ms** |
| Elasticsearch BM25 + kNN | ~60 ms |
| Hybrid + LLM query expansion (deepseek-r1:7b) | +1.5 – 4 s |
| Hybrid + LLM re-ranking (top-10) | +2 – 6 s |
| Hybrid + LLM explanations (top-5) | +4 – 10 s |
| **Hybrid + all 3 LLM stages** | **~8 – 20 s total** |

Tips to stay fast:
- Use the smaller `deepseek-r1:7b` or `deepseek-r1:1.5b`.
- Lower `LLM_RERANK_K` and `LLM_EXPLAIN_K`.
- Keep the UI toggle **off** for rapid iteration; flip it on when you want
  the richer output.

---

## Docker

```bash
# full stack (ES + app)
docker compose up --build

# ES only
docker compose up -d elasticsearch
```

The compose file mounts `./analysis/` into the ES container so the DE↔EN
synonym filter is loaded at index creation time.

---

## Project layout

```
docsearch/
├── config.py          # all env-tunable settings
├── parser.py          # PDF / DOCX / TXT → (page_no, text)
├── preprocess.py      # langdetect + spaCy lemma + NER + chunking
├── embedder.py        # sentence-transformers wrapper
├── indexer_es.py      # Elasticsearch mapping + BM25 + kNN
├── indexer_faiss.py   # FAISS cosine index + metadata
├── synonyms.py        # DE↔EN seed dictionary + word-bucket classifier
├── query.py           # query expansion (terms, synonyms, MT, LLM★, NER)
├── ranker.py          # proximity, weighted hybrid, page aggregation
├── llm.py             # Ollama client — the 3 LLM integration points
├── output_log.py      # session + elastic JSON writers
├── pipeline.py        # DocSearch orchestrator
├── cli.py             # `python -m docsearch.cli ...`
├── server.py          # FastAPI app
└── static/
    └── index.html     # self-contained web UI
analysis/
└── de_en_synonyms.txt # mounted into ES for synonym_graph filter
data/                  # runtime artefacts (FAISS, pickles, output JSON)
```

---

## Troubleshooting

- **`pymupdf.exe` / permission error on `pip install`** → you're installing
  into system Python. Use a virtualenv (see Installation).
- **`OSError: [E050] Can't find model 'de_core_news_sm'`** → download the
  model inside the active venv.
- **LLM status pill is red** → Ollama isn't running, or the model name in
  `OLLAMA_MODEL` isn't pulled. Run `ollama list` to check.
- **"No document indexed yet"** → ingest at least once (UI panel 1, or CLI).
- **Docker error `dockerDesktopLinuxEngine: cannot find file`** → start
  Docker Desktop; wait for the whale icon to go solid.
- **`embeddings.position_ids | UNEXPECTED`** → benign Sentence-Transformers
  checkpoint warning; ignore.

---

## License

MIT.
