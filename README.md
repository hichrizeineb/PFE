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

```
generate_queries.py
    ↓ kb/Search_queries_world.json  (~421k entries)

processor_standalone.py
    ↓ kb/outputs/documents_embedding.jsonl

embed_and_load.py
    ↓ chroma_db/  (ChromaDB, collection: copernicus_rag)

rag_chatbot.py  ←  you talk to this
    parse_intent() → ChromaDB filters → Ollama (mistral)
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

## Rebuilding the Database

Only needed when you change `generate_queries.py` or `processor_standalone.py`.  
**Do NOT rebuild just to change chatbot behaviour** — edit `rag_chatbot.py` directly.

```bash
python generate_queries.py          # ~20 min — regenerates Search_queries_world.json
python processor_standalone.py      # ~5 min  — regenerates documents_embedding.jsonl
python embed_and_load.py            # ~35 min — reloads ChromaDB
```

| File changed | generate | process | embed |
|---|---|---|---|
| `rag_chatbot.py` | ❌ | ❌ | ❌ |
| `processor_standalone.py` | ❌ | ✅ | ✅ |
| `generate_queries.py` | ✅ | ✅ | ✅ |

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
