# Copernicus RAG Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot that lets you search and explore **421,000+ Copernicus satellite image query documents** using natural language.

Built with ChromaDB, sentence-transformers, and Ollama (Mistral).

---

## What It Does

You ask questions in plain language:

```
> show me S3 queries in germany in spring 2025
> what radar data exists for france in 2025?
> give me optical queries over italy between april and august 2025
```

The bot:
1. Parses your intent (satellite, country, date range)
2. Searches ChromaDB with hard metadata filters
3. Returns exact counts (docs fully inside your window) + partial counts (docs that overlap)
4. Asks how many you want to see
5. Prints results directly, then asks Ollama (Mistral) to add a short comment

---

## Architecture

### v1 (original — preserved, do not delete)
```
generate_queries.py
    ↓ kb/Search_queries_world.json  (~421k entries)

processor_standalone.py
    ↓ kb/outputs/documents_embedding.jsonl  (421k docs)

embed_and_load.py
    ↓ chroma_db/  (collection: copernicus_rag, 421k docs)

rag_chatbot.py  ←  hardcoded parse_intent() → ChromaDB → Ollama
```

### v2 (grouped, LLM-enriched — active)
```
collection_indexer.py
    ↓ kb/outputs/collection_index.json        (1,337 collections)
    ↓ kb/outputs/theme_index.json             (cleaned themes)
    ↓ kb/outputs/mission_theme_index.json     (curated LLM vocabulary)
    ↓ kb/outputs/sentinel_collection_map.json (4 missions → providers)

processor_grouped.py  [--enrich]
    ↓ kb/outputs/documents_embedding.jsonl   (204 grouped docs)
    ↓ kb/outputs/documents_full.json         (204 docs + geometry + date_range)
    ↓ kb/outputs/query_groups.json           (20 diverse API samples per group)
    ↓ kb/enriched_groups/<group_id>.json     (LLM cache, optional)

embed_and_load_grouped.py
    ↓ chroma_db/  (collection: copernicus_grouped, 204 docs)

rag_chatbot_v2.py  ←  LLM intent extraction → ChromaDB → Ollama synthesis
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `mistral` pulled
- ~6 GB disk space for ChromaDB

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pull the Ollama model:

```bash
ollama pull mistral
```

---

## Running the Chatbot

**The ChromaDB is already built — just run:**

```bash
python rag_chatbot.py
```

Options:

```bash
python rag_chatbot.py --model mistral          # choose Ollama model (default: mistral)
python rag_chatbot.py --top 100                # retrieval pool size (default: 50)
python rag_chatbot.py --verbose                # show filters and scores
python rag_chatbot.py --ask "S2 france 2025"  # single question, non-interactive
```

In-chat commands:

```
/models          — list available Ollama models
/model <name>    — switch model
/verbose         — toggle verbose mode
/top <n>         — change retrieval pool size
/clear           — clear conversation history
/help            — show commands
quit             — exit
```

---

## How to Use

**Basic query:**
```
You: show me s1 radar queries in spain in summer 2025
→ Found 36 exact + 247 partial matches.
→ How many would you like to see? (1–50, default 5)
You: 8 exact and 6 partial
→ [results printed]
```

**Follow-up without re-searching:**
```
You: show me more partial       → reuses last results, shows more partial
You: show 10 exact              → reuses last results, shows 10 exact
You: show me 5                  → reuses last results, shows 5 of each
```

**Supported satellites:**
| Code | Mission | Type |
|---|---|---|
| S1 | Sentinel-1 | Radar SAR — works day/night through clouds |
| S2 | Sentinel-2 | Optical multispectral — requires clear sky |
| S3 | Sentinel-3 | Ocean and land surface monitoring |
| S5P | Sentinel-5P | Atmospheric gas (NO2, methane, ozone) |

**Date formats understood:**
- `in march 2025`
- `between april and august 2025`
- `from june to october 2025`
- `summer 2025` / `spring 2025`
- `2025` (full year)

**Country aliases supported:**
- `usa` → united states
- `uk` → united kingdom
- `drc` → democratic republic of the congo
- `uae` → united arab emirates
- `bosnia` → bosnia and herzegovina

---

## Exact vs Partial Matches

- **Exact** — the document's sensing window falls **entirely** inside your requested period
- **Partial** — the document **overlaps** your period but starts earlier or ends later (location is always exact — only time overflows)

---

## Rebuilding the v1 Database

Only needed when you change `generate_queries.py` or `processor_standalone.py`.

```bash
python generate_queries.py          # ~20 min — regenerates Search_queries_world.json
python processor_standalone.py      # ~5 min  — regenerates documents_embedding.jsonl
python embed_and_load.py            # ~35 min — reloads ChromaDB (copernicus_rag)
```

---

## v2 Pipeline — Build and Enrichment

### Required env vars

| Variable | Default | Description |
|---|---|---|
| `ENRICH_INDEX` | `0` | Set to `1` to enable LLM enrichment |
| `ENRICH_MODEL` | `mistral` | Ollama model to use for enrichment |

### Run index build WITHOUT enrichment (current behavior preserved)

```bash
python3 collection_indexer.py       # rebuild collection indexes
python3 processor_grouped.py        # rebuild 204 grouped documents
python3 embed_and_load_grouped.py   # rebuild ChromaDB copernicus_grouped
```

### Run index build WITH LLM enrichment

```bash
python3 collection_indexer.py
python3 processor_grouped.py --enrich          # calls Ollama for each of 204 groups
python3 embed_and_load_grouped.py              # merges enriched text automatically
```

Or via environment variable:

```bash
ENRICH_INDEX=1 python3 processor_grouped.py
ENRICH_MODEL=llama3 ENRICH_INDEX=1 python3 processor_grouped.py
```

### Force re-enrichment (ignore cache)

```bash
python3 processor_grouped.py --enrich --force-enrich
# or enrich a single group:
python3 enricher.py --group S1_urban_europe --force
```

### Enrichment cache

Cached enrichment JSON files are stored in `kb/enriched_groups/<group_id>.json`.

Each cache file contains:
```json
{
  "group_id": "S1_urban_europe",
  "input_hash": "abc123...",
  "prompt_version": "v1",
  "model_name": "mistral",
  "enriched_at": "2026-04-29T10:00:00Z",
  "enrichment": {
    "group_title": "Sentinel-1 urban monitoring in Europe",
    "enriched_description": "...",
    "normalized_themes": [...],
    "synonyms": [...],
    "contains_bullets": [...],
    "example_questions": [...]
  }
}
```

Re-runs with `--enrich` only call the LLM again if:
- No cache file exists for that group, OR
- The `input_hash` changed (source data updated), OR
- `--force-enrich` is passed

### Run retrieval evaluation

```bash
python3 eval_retrieval.py           # 10 queries, top-3, writes report to reports/
python3 eval_retrieval.py --top 5   # show top-5 per query
python3 eval_retrieval.py --no-report  # print only
```

Reports are written to `reports/eval_YYYYMMDD_HHMMSS.json`.

---

## Architecture Decision — LangGraph

LangGraph is **not used** and should not be added unless the query-time workflow
becomes stateful and requires:

- Routing between multiple retrieval strategies
- Confidence evaluation + retry loops
- Crawler update orchestration
- Human-in-the-loop validation
- Stateful multi-step workflows

If the flow remains `retrieve → synthesise`, LangChain + ChromaDB is sufficient.

---

## Known Limitations

- Enrichment requires a local Ollama instance with a pulled model (`ollama pull mistral`)
- Enrichment for all 204 groups takes ~30–60 min depending on model and hardware
- LLM enrichment quality depends on the model; mistral works well for this task
- `copernicus_rag` (v1, 421k docs) is preserved but not used by the v2 chatbot

## Next Steps

- `rag_chatbot_v2.py` — LLM intent extraction (JSON mode) → `copernicus_grouped` → synthesis
- LangChain query-time chain (Option B): query rewriting → ChromaDB → re-ranking → synthesis
- Crawler integration: dynamic data ingestion with `input_hash`-based change detection

---

## Project Structure

```
dataionics_rag/
├── rag_chatbot.py              # Main chatbot — edit this for UX changes
├── generate_queries.py         # Generates the query dataset from GeoNames
├── processor_standalone.py     # Processes JSON → embedding-ready JSONL
├── embed_and_load.py           # Embeds and loads into ChromaDB
├── config.py                   # Shared configuration
├── requirements.txt
├── kb/
│   ├── Search_queries_world.json       # Generated dataset (~421k entries)
│   ├── outputs/
│   │   └── documents_embedding.jsonl   # Processed documents
│   └── geodata/                        # GeoNames + Natural Earth source files
├── chroma_db/                          # ChromaDB vector store (421,279 docs)
└── SESSION_STATE.md                    # Development state tracker
```
