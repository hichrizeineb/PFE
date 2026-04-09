# Claude Code — Copernicus RAG Pipeline: Full Diagnosis & Improvement Plan

## YOUR FIRST JOB: READ EVERYTHING BEFORE WRITING A SINGLE LINE OF CODE

Read these files in this exact order. Understand what each one does:

```
~/dataionics_rag/processor_standalone.py     # Step 1: raw JSON → normalized docs
~/dataionics_rag/embed_and_load.py           # Step 2: docs → vectors → ChromaDB
~/dataionics_rag/rag_chatbot.py              # Step 3: user question → answer
~/dataionics_rag/Search queries.json         # Raw input: 396 Copernicus API queries
~/dataionics_rag/documents_embedding.jsonl   # Output of processor, input to ChromaDB
~/dataionics_rag/documents_full.json         # Full normalized docs for debugging
~/dataionics_rag/documents_vectorstore.jsonl # Vector store format output
```

Also look for any other Python files, notebooks, or JSON files in the project folder
that represent previous pipeline attempts. Read those too — they contain ideas and
approaches worth understanding.

---

## CONTEXT: WHAT THIS PROJECT IS

This is a RAG (Retrieval-Augmented Generation) pipeline over Copernicus satellite data.

The input is `Search queries.json` — a collection of Copernicus API queries, each
representing a satellite search request with:
- A satellite type: S1 (Radar/SAR), S2 (Optical), S3 (Ocean), S5P (Atmospheric)
- A region name (e.g. "Estonia Forest", "Alps", "Paris")
- A geometry (Point, Polygon, LineString, MultiPolygon) with coordinates
- A date range (sensingDateMin / sensingDateMax)
- A dataset URL

The pipeline transforms these into searchable documents and loads them into ChromaDB
so a user can ask natural language questions like:
  "give me all S2 queries over Europe in summer 2024"
  "what radar data exists for South America?"
  "show me forest monitoring queries between March and November 2025"

The chatbot uses Ollama (local LLMs, no API key) + sentence-transformers for embeddings.

---

## KNOWN BUGS TO FIX (diagnosed before calling you)

### BUG 1 — ChromaDB query limit hit (CRITICAL)
The chatbot currently calls:
```python
n_results=min(top_k, collection.count())
```
With 396 docs in ChromaDB, this fetches up to 396 docs per query — consuming ~94%
of the ChromaDB rate limit in one call. 

FIX: Build a ChromaDB `where` clause from the parsed intent and push filtering
to the database. Never fetch more than 15-20 results. Use `$and`, `$eq`, `$lte`, `$gte`
operators. Example:
```python
where = {"$and": [
    {"continent": {"$eq": "europe"}},
    {"biome":     {"$eq": "forest"}},
    {"date_start":{"$lte": "2025-11-28"}},
    {"date_end":  {"$gte": "2025-03-01"}},
]}
collection.query(query_embeddings=vec, n_results=15, where=where, ...)
```
Wrap the ChromaDB call in try/except because ChromaDB raises an exception (not returns
empty) when the `where` clause matches 0 documents. In that case fall back to pure
vector search with n_results=5.

### BUG 2 — Missing date_start / date_end in ChromaDB metadata (CRITICAL)
The current `documents_embedding.jsonl` metadata does NOT contain `date_start` or
`date_end` fields. This means date range filtering in ChromaDB is impossible.

Look at `to_embedding_input()` in `processor_standalone.py` — it only stores:
satellite, geometry_type, region, dataset, year, season, continent, biome, center_lon, center_lat.

FIX in `processor_standalone.py` — add to the metadata dict in `to_embedding_input()`:
```python
"date_start":    self.temporal_range.start.strftime("%Y-%m-%d"),
"date_end":      self.temporal_range.end.strftime("%Y-%m-%d"),
"mission_type":  self.mission_type.value,
"duration_days": self.temporal_range.duration_days,
"original_name": self.original_name,
```
Also replace None values with "" for continent and biome (ChromaDB rejects None).

### BUG 3 — Semantic texts too vague for natural language matching (IMPORTANT)
Current semantic text looks like:
  "Satellite S2 search query. for Estonia Forest. using Polygon geometry.
   from April 2024. to September 2024. (153 days). in spring. continent: europe.
   environment: forest. centered at (25.50, 58.90)."

Problems:
- "forest" won't match "forestation", "woodland", "tree cover", "vegetation"
- "europe" won't match "France", "Germany", "European"
- No mission description in plain English ("optical imaging" vs just "S2")
- Month names are there but no ISO dates — hard to match "2024-04" style queries

FIX in `_generate_semantic_text()` in `processor_standalone.py`:
- Add biome synonym expansion: forest → "forest woodland forestry vegetation tree cover"
- Add continent synonym expansion: europe → "europe european france germany spain italy"
- Add mission description in plain English per satellite
- Add ISO date string alongside human readable month
- Add the dataset name in plain English

### BUG 4 — None values in ChromaDB metadata cause filter errors
Current metadata has None for biome (24 docs) and continent (1 doc).
ChromaDB `where` clauses fail silently or raise errors on None values.

FIX: In ALL metadata outputs, replace `None` with `""` (empty string).

### BUG 5 — Wrong question → wrong answer (the user saw this live)
The user asked: "what S2 data radar imaging queries exist?"
S2 is OPTICAL, not radar. Radar is S1. The chatbot answered as if it didn't know this.

FIX: The system prompt in `build_prompt()` must explicitly tell the LLM:
- S1 = Sentinel-1 = Radar SAR imaging
- S2 = Sentinel-2 = Optical multispectral imaging  
- S3 = Sentinel-3 = Ocean and land surface monitoring
- S5P = Sentinel-5P = Atmospheric gas monitoring (NO2, methane, etc.)

So when asked "S2 radar imaging", the LLM corrects: "S2 is optical not radar.
Here are the S2 optical results... If you want radar data, that is S1."

---

## WHAT TO BUILD: THE IMPROVED PIPELINE

### PHASE 1 — Fix processor_standalone.py

Changes needed (do NOT rewrite the whole file, only modify these methods):

1. `_generate_semantic_text()` — enrich text with synonyms and ISO dates
2. `to_embedding_input()` — add date_start, date_end, mission_type, duration_days,
   original_name to metadata; replace None → ""
3. `to_vector_store_doc()` — same None → "" fix, add date_start/date_end
4. `REGION_CONTINENT_MAP` — add missing entries: "france", "french", "germany",
   "german", "uk", "british", "lakes", "hills", "highlands"
5. `BIOME_KEYWORDS` — add synonyms: forest should also match "lakes" (Finland Lakes),
   and add "hills" as a possible biome hint

After fixing, the processor must be runnable as:
```bash
python3 processor_standalone.py
```
And produce correct documents_embedding.jsonl with date_start/date_end in metadata.

### PHASE 2 — Fix rag_chatbot.py

Rewrite these parts only:

#### A. parse_intent(question) — build or improve intent parser
Must extract from natural language:
- continent: "france/europe/european/spain/germany/italian" → "europe"
  "amazon/brazil/south america/argentina" → "south_america"
  "africa/african/morocco/kenya" → "africa"
  etc.
- biome: "forest/forestry/forestation/woodland/tree/vegetation/deforestation" → "forest"
  "agriculture/farming/crops/farmland" → "agricultural"
  "ocean/sea/marine/water/gulf" → "ocean"
  "coast/coastal/shore/beach" → "coastal"
  "mountain/alpine/highland/alps/peak" → "mountain"
  "urban/city/industrial/pollution" → "urban"
  "desert/arid/dry" → "desert"
  "ice/glacier/arctic/polar" → "ice"
  etc.
- satellite: "s1/sentinel-1/radar/sar" → "S1"
  "s2/sentinel-2/optical/multispectral/visual" → "S2"
  "s3/sentinel-3/ocean" → "S3"
  "s5p/sentinel-5/atmospheric/no2/pollution/gas" → "S5P"
- date_start, date_end as "YYYY-MM-DD":
  "March 2025" → "2025-03-01"
  "between March 2025 and November 2025" → start="2025-03-01", end="2025-11-30"
  "in 2024" → start="2024-01-01", end="2024-12-31"
  "summer 2024" → start="2024-06-01", end="2024-08-31"
  ISO dates "2024-03-01" directly

#### B. build_where_clause(intent) — NEW function
Build ChromaDB native filter from intent.
Never fetch more than 15-20 results.
Always wrap in try/except for the 0-match case.

#### C. retrieve_and_filter() — use build_where_clause
Replace the Python-side loop filtering with DB-side filtering.

#### D. build_prompt() — smarter system prompt
Add satellite type definitions to system prompt.
Explicitly say "N matching queries found" in the structured user message.
List every matching query with: name, satellite + mission type, region,
dates (start → end), biome, continent, dataset.
If 0 results: explain what filters were applied, say nothing matched,
show what IS available (e.g. "No forest data for Europe in 2025, but here are
European forest queries in 2024: [list]").

#### E. Startup — show data summary
When chatbot starts, show:
- Total docs in ChromaDB
- Breakdown by satellite (S1/S2/S3/S5P counts)
- Date range of available data (earliest to latest)
- Available continents and biomes
This helps the user know what to ask.

---

## RUN ORDER AFTER YOUR CHANGES

The pipeline must run in this exact order:
```bash
# 1. Regenerate documents (semantic texts + metadata)
python3 processor_standalone.py

# 2. Reload ChromaDB (delete old collection, reload with new metadata)
python3 embed_and_load.py

# 3. Run chatbot
python3 rag_chatbot.py --model mistral:latest --verbose
```

embed_and_load.py already works — do NOT change it unless absolutely necessary.
It reads documents_embedding.jsonl and loads into ChromaDB collection "copernicus_rag".

---

## TEST QUESTIONS TO VALIDATE AFTER CHANGES

Run these one by one and verify the answers make sense:

```bash
# Should say "S2 is optical not radar" and list S2 optical docs
python3 rag_chatbot.py --model mistral:latest --ask "what S2 data radar imaging queries exist?"

# Should list S1 radar docs, not S2
python3 rag_chatbot.py --model mistral:latest --ask "show me radar SAR queries"

# Should return 0 results (no France data) and suggest nearby European data
python3 rag_chatbot.py --model mistral:latest --ask "give me queries from France between March 2025 and November 2025 for forestation"

# Should return European forest docs (Latvia, Estonia, Lithuania, Slovakia forests)
python3 rag_chatbot.py --model mistral:latest --ask "forest monitoring queries in Europe in 2024"

# Should return S2 results for specific year
python3 rag_chatbot.py --model mistral:latest --ask "what S2 optical data exists for 2025?"

# Should return ocean/S3 docs
python3 rag_chatbot.py --model mistral:latest --ask "ocean monitoring in the Mediterranean"

# Should work with verbose to show filters detected
python3 rag_chatbot.py --model mistral:latest --verbose --ask "atmospheric pollution data over Asian cities"
```

---

## LEARNING NOTES FOR THE DEVELOPER

These are things to understand about how RAG pipelines work, not just fixes:

### Why the semantic text quality matters
The embedding model (all-MiniLM-L6-v2) converts text to a 384-number vector.
Similar meanings produce similar vectors. So if your text says "environment: forest"
but the user asks "forestation", those two phrases are semantically close — but
"forest" and "deforestation" are EVEN closer if you add synonyms to the text.
The richer and more natural the semantic text, the better retrieval works.

### Why ChromaDB where clauses matter
Vector search finds "semantically similar" documents. But "similar" is fuzzy.
If you ask for "Europe 2024", vector search might also return "Asia 2024" because
the year is close. A `where` clause is a HARD filter — if you filter by
continent="europe", you will NEVER get Asia results regardless of similarity score.
The combination of both (hard filter + vector similarity within that filter) is
more precise and much cheaper on rate limits.

### Why date_start/date_end must be ISO strings not just year/season
ChromaDB supports string comparison operators: $lte, $gte work on ISO date strings
because "2024-03-01" < "2024-11-28" lexicographically. If you store only year=2024,
you can only filter by exact year, not by month or date range.

### Why the LLM must know satellite types
The LLM (Mistral, Llama, etc.) is a general model. It doesn't know your specific
data. If you don't tell it in the system prompt that S2=optical and S1=radar,
it will give confusing answers when the user asks about "S2 radar" (which is
a contradiction in your data). Always put domain knowledge in the system prompt.

### Why None → "" in ChromaDB metadata
ChromaDB's Python client raises exceptions when you try to use `where` filters
on fields that contain Python None. Always store empty string "" or 0 instead.
This is a known ChromaDB limitation.

---
## ADDITIONAL FILES — READ FOR LEARNING ONLY, DO NOT MERGE BLINDLY

There are other Python files in the project folder that belong to a separate
pipeline attempt: files like config.py, ingest.py, intent_parser.py, retriever.py
and any others you find that are not processor_standalone.py, embed_and_load.py,
or rag_chatbot.py. These files are NOT part of the current working pipeline and
are NOT connected to each other in a way that currently runs. Do NOT replace or
overwrite anything in the three main files with code from these files directly.
Instead, read them carefully for ideas and logic that could improve the current
pipeline. Specifically look for: how intent_parser.py extracts filters from
natural language (compare it to the parse_intent() function we need to build —
is the logic better? more complete? does it handle more cases?), how retriever.py
queries the vector store (does it use where clauses? does it handle fallback
differently?), how config.py structures settings (are there useful constants,
mappings, or satellite definitions we are missing?), and how ingest.py processes
raw data (does it enrich metadata differently from processor_standalone.py?).
After reading them, tell me: what useful logic exists in these files that does
NOT yet exist in the main pipeline, and what you recommend importing or adapting.
Then wait for my validation before touching anything.

--- 

## CHUNK SIZE & OVERLAP ANALYSIS (do this before any code)

Before touching any code, measure the actual character and token length of
the semantic texts currently produced by _generate_semantic_text(). Print
min, max, and average length across all documents. Then answer: are the
texts short enough to fit in one embedding without truncation? The model
all-MiniLM-L6-v2 has a hard limit of 256 tokens and truncates silently
above that — so a document that looks fine may be losing its tail.
Decide whether chunking is needed at all, or whether each document is
already one clean atomic unit. Decide whether overlap is relevant given
the document size. Report findings and recommendation before writing
anything.

## NLP SIMPLICITY & PROMPT ENGINEERING FIRST

Keep the intent parser minimal — only extract dates and explicit satellite
codes (S1/S2/S3/S5P) as hard ChromaDB filters. No keyword maps for
countries, biomes, continents, or synonyms. The local LLM already
understands that France is in Europe and forestation relates to forests.
Pass the raw user question directly to the LLM with the retrieved
documents and let it reason about relevance.

Write the system prompt in clearly labeled sections so it is easy to
improve later through prompt engineering without touching code:

  [ROLE] who the assistant is and what it knows about satellites
  [DATA] what documents it has access to and their structure  
  [RULES] how to format the answer, what to say when nothing matches
  [DOCUMENTS] the actual retrieved docs injected here

This structure means future improvements are just prompt edits, not
code changes. Simple retrieval first — refine only if LLM answers
prove insufficient after testing.

## DO NOT CHANGE

- embed_and_load.py (unless you find a bug that blocks the pipeline)
- The collection name "copernicus_rag"
- The embedding model "all-MiniLM-L6-v2" (must match what was used to build ChromaDB)
- The virtual environment path (.venv)

---

## WAIT FOR VALIDATION

After reading all files and understanding the full picture:

1. Show me a summary of what you found (what state each file is in, what bugs
   you confirmed, anything unexpected you found in the other pipeline files)
2. Show me the plan of what you will change and in what order
3. Wait for my "yes go ahead" before writing any code
4. Make changes file by file, showing diffs, waiting for confirmation between files
   if the changes are large

The goal is for the developer to understand what is being changed and why,
not just get working code.
