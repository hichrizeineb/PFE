# Satellite RAG Pipeline — Complete Documentation

> **Project**: Copernicus / STAC Satellite Data Discovery System  
> **Model**: `all-MiniLM-L6-v2` (384-dim) + Ollama/mistral  
> **Vector DB**: ChromaDB (persistent SQLite + HNSW)  
> **Interface**: `rag_chatbot_v3.py` — interactive terminal chatbot

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Stage 0 — Raw Input Sources](#2-stage-0--raw-input-sources)
3. [Stage 1 — STAC Provider Crawling & Collection Ingestion](#3-stage-1--stac-provider-crawling--collection-ingestion)
4. [Stage 2 — Query Generation](#4-stage-2--query-generation)
5. [Stage 3 — Query Grouping & Document Building](#5-stage-3--query-grouping--document-building)
6. [Stage 4 — LLM Enrichment](#6-stage-4--llm-enrichment)
7. [Stage 5 — Embedding & Vector Storage](#7-stage-5--embedding--vector-storage)
8. [Stage 6 — Index Building](#8-stage-6--index-building)
9. [Stage 7 — RAG Chatbot (v3)](#9-stage-7--rag-chatbot-v3)
10. [File Map — Every File Explained](#10-file-map--every-file-explained)
11. [Generated JSON/JSONL Files Reference](#11-generated-jsonjsonl-files-reference)
12. [ChromaDB Collections Reference](#12-chromadb-collections-reference)
13. [End-to-End Data Flow Diagram](#13-end-to-end-data-flow-diagram)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Dependencies](#15-dependencies)

---

## 1. Architecture Overview

The system is a **two-track RAG pipeline** that unifies two separate data sources into one searchable knowledge base, then exposes it through a conversational chatbot.

### Two Tracks

| Track | Source | Records | Collection in ChromaDB |
|---|---|---|---|
| **Track A** — Real STAC | Live STAC provider APIs (17 providers) | 5,999 collections | `stac_collections` |
| **Track B** — Synthetic Queries | GeoNames + Natural Earth → 421k generated queries | 204 grouped documents | `copernicus_grouped` |

Track A gives the chatbot real, live satellite collection metadata (titles, descriptions, spatial/temporal extents, asset types). Track B gives it geographic breadth — 421,280 pre-built search queries spanning 4 satellites × 9 biomes × 8 continents × 20,536 locations.

### Retrieval Strategy

```
User query
    │
    ▼
[Intent extraction — Ollama/mistral]
    │  satellite / theme / location / date
    ▼
[Semantic search — ChromaDB / all-MiniLM-L6-v2]
    │  cosine similarity on 5,999 stac_collections docs
    ▼
[Reranking — platform / spatial / temporal validation]
    │  +0.15 platform match, -0.50 mismatch, ±0.10 spatial/temporal
    ▼
[Live STAC /search call — real items]
    │  POST /search with bbox + datetime + collection
    ▼
[Synthesis — Ollama/mistral]
    │  2-4 sentence summary of results
    ▼
User response
```

---

## 2. Stage 0 — Raw Input Sources

### 2.1 STAC Provider List (CSV)

**File**: Used as input to `stac_ingest.py`  
**Content**: Root URLs of STAC catalog providers

Providers crawled (17 total):
| Provider | URL | Notable Collections |
|---|---|---|
| NASA/ASF | `cmr.earthdata.nasa.gov/stac/ASF` | OPERA S1-RTC |
| Terrascope | `stac.terrascope.be` | Sentinel-2 NDVI |
| Copernicus Dataspace | `catalogue.dataspace.copernicus.eu/stac` | S1, S2, S3, S5P |
| Microsoft Planetary Computer | `planetarycomputer.microsoft.com/api/stac/v1` | Sentinel-2-L2A |
| EUMETSAT | `stac.eumetsat.int` | Meteosat, MSG |
| ESA | `cmr.earthdata.nasa.gov/stac/ESA` | EarthCARE |
| CDDIS | `cmr.earthdata.nasa.gov/stac/CDDIS` | GNSS |
| AU_AADC | `cmr.earthdata.nasa.gov/stac/AU_AADC` | Antarctic |
| + 9 more | ... | ... |

### 2.2 GeoNames (Track B only)

**File**: `kb/geodata/allCountries.txt`  
**Size**: ~300 MB  
**Records**: 11 million world locations  
**Fields used**: name, latitude, longitude, feature class, feature code, country, population, elevation, alternate names

Filtered to:
- **Cities**: 5,000 (population ≥ 100k)
- **Mountains**: 6,000 (elevation ≥ 500m or name length ≥ 6)
- **Terrain**: 3,000 (islands, peninsulas, deserts, badlands)
- **Forests**: 1,000
- **Waters**: 3,000 (rivers, lakes, seas)
- **Parks**: 1,000

### 2.3 Natural Earth (Track B only)

**Files**: `kb/geodata/ne_*.geojson`  
Provides: oceans, rivers, lakes, country boundaries with continent mapping

### 2.4 Collections CSV

**File**: `collections.csv`  
**Records**: 1,708 rows (1,337 unique collections)  
**Purpose**: Maps Copernicus collection IDs to Sentinel satellite families (S1/S2/S3/S5P)  
**Used by**: `collection_indexer.py`

---

## 3. Stage 1 — STAC Provider Crawling & Collection Ingestion

**Script**: `stac_ingest.py`

### What it does

Walks every STAC provider, paginates through all collections, samples items, and builds text documents for embedding.

### HTTP Fallback Chain

For each collection, item sampling tries:
1. `POST /search` with `{"collections": [id], "limit": 3}`
2. `GET /search?collections=id&limit=3`
3. `GET /collections/{id}/items?limit=3`

### Key Processing Steps

```
CSV of provider URLs
    │
    ▼  crawl_provider()
Root catalog fetch (GET /)
    │  → extract collections_url, search_url, search_method, conforms_to
    │  → detect CQL2 support, access_type (token/open)
    ▼  crawl_collections()
Paginate /collections (GET, page by page)
    │  → for each collection: fetch /collections/{id}
    │  → extract title, description, extent, platforms, instruments, bands, keywords
    │  → fetch /collections/{id}/queryables (filter schema)
    ▼  sample_items()
Sample 3 items per collection
    │  → extract asset hrefs, roles, types, properties
    ▼  build_rag_document()
Convert CollectionRecord → human-readable text
    │  "Collection: {title}\nID: {id}\nDescription: {desc}\n
    │   Platforms: {platforms}\nSpatial extent: {bbox}\n
    │   Temporal: {start} to {end}\n..."
    ▼
Write JSONL files
```

### Output Files

| File | Records | Purpose |
|---|---|---|
| `kb/outputs/stac_providers.jsonl` | 17 | Provider endpoints, search methods, CQL2 flags |
| `kb/outputs/stac_collections.jsonl` | 5,999 | Full collection metadata (raw) |
| `kb/outputs/stac_items_sample.jsonl` | 1,664 | 3 sampled items per collection |
| `kb/outputs/stac_rag_documents.jsonl` | 5,999 | Text documents ready for embedding |
| `kb/outputs/stac_ingest_errors.jsonl` | 2 | Failed collections with error detail |
| `kb/outputs/stac_ingest_summary.json` | 1 | Crawl statistics |

### RAG Document Schema

Each document in `stac_rag_documents.jsonl` contains:
- `doc_id` — `{provider_root}::{collection_id}`
- `text` — Human-readable multiline description (used for embedding)
- `collection_id`, `provider_root`, `title`
- `platforms`, `instruments`, `bands`, `keywords`, `license`
- `extent_temporal`, `extent_spatial`
- `stac_extensions`, `item_asset_names`
- `raw_collection_hash` — for change detection
- `crawl_ts` — ISO 8601 timestamp

---

## 4. Stage 2 — Query Generation

**Script**: `generate_queries.py`

### What it does

Generates 421,280 satellite search queries by crossing locations × satellites × date patterns. Produces queries in the Copernicus API request format.

### Generation Logic

```
GeoNames (11M) + Natural Earth
    │
    ▼  filter + categorize
20,536 unique locations (cities, mountains, waters, forests, parks, oceans, rivers, lakes)
    │
    ▼  cross-product
location × satellite (S1/S2/S3/S5P) × date_variant (E/A/B/C/D)
    │
    ▼
421,280 queries
```

### Date Variants

| Code | Range | Example use case |
|---|---|---|
| E | 7–29 days | Recent events |
| A | 30–60 days | Short campaign |
| B | 61–120 days | Seasonal (3-4 months) |
| C | 121–365 days | Annual |
| D | 366–730 days | Multi-year trend |

### Output

`kb/Search_queries_world_v2.json` — 421,280 queries, each with:
- `sat`: S1 / S2 / S3 / S5P
- `loc`: location name
- `co`: country
- `cont`: continent
- `bio`: biome (urban / forest / ocean / coastal / mountain / river / freshwater / steppe / desert)
- `lat`, `lon`
- `ds`, `de`: date_start, date_end (ISO 8601)
- `url`: Copernicus API endpoint URL
- `body`: POST request body
- `method`: GET / POST

---

## 5. Stage 3 — Query Grouping & Document Building

**Script**: `processor_grouped.py`

### What it does

Reduces 421,280 individual queries into **204 semantic group documents** for efficient embedding. Each group represents one satellite × biome × continent combination.

### Grouping Schema

```
4 satellites × 9 biomes × 8 continents = 288 possible groups
Actually produced: 204 groups (not all combinations have data)
```

**Satellites**: S1, S2, S3, S5P  
**Biomes**: urban, forest, river, coastal, ocean, mountain, freshwater, steppe, desert  
**Continents**: europe, asia, africa, north_america, south_america, oceania, antarctica, global

### Document Structure per Group

```json
{
  "group_key": "S2|forest|south_america",
  "satellite": "S2",
  "biome": "forest",
  "continent": "south_america",
  "dataset": "sentinel-2-l2a",
  "mission_type": "optical",
  "mission_description": "Sentinel-2 multispectral imagery at 10m resolution...",
  "themes": ["vegetation", "NDVI", "land cover", "agriculture", ...],
  "text": "...(rich description for embedding)...",
  "query_count": 2847,
  "unique_locations": 312,
  "sample_locations": ["Amazon River", "Manaus", ...],
  "date_range": "2020-01-01/2025-12-31"
}
```

### Also Produces

`kb/outputs/query_groups.json` — for each group_key, 20 sample queries with full request details. Used by the chatbot to show real API examples.

---

## 6. Stage 4 — LLM Enrichment

**Script**: `enricher.py`

### What it does

Calls Ollama/mistral to generate richer natural-language descriptions for each of the 204 groups. Results are cached per group.

### Enrichment Output per Group

```json
{
  "enriched_description": "Sentinel-1 SAR data over European forests enables...",
  "normalized_themes": ["deforestation", "biomass", "carbon stock", ...],
  "synonyms": ["woodland", "boreal", "temperate forest", ...],
  "contains_bullets": ["C-band SAR", "all-weather", "10-day revisit", ...],
  "example_questions": ["How can I detect deforestation?", ...],
  "confidence_notes": null
}
```

### Cache Location

`kb/enriched_groups/{group_key}.json` — one file per group (204 files)

### Integration

`embed_and_load_grouped.py` reads the cache and merges enriched fields into the document text before embedding. This makes the ChromaDB search richer because the vectors encode synonyms, themes, and natural-language descriptions.

---

## 7. Stage 5 — Embedding & Vector Storage

### 7.1 STAC Collections Embedding

**Script**: `embed_stac.py`  
**Input**: `kb/outputs/stac_rag_documents.jsonl` (5,999 docs)  
**Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, cosine similarity)

```
stac_rag_documents.jsonl
    │
    ▼  SentenceTransformer.encode()
5,999 × 384-dim vectors (batch processing)
    │
    ▼  chromadb.Collection.add()
ChromaDB collection: "stac_collections"
    │  stored in: chroma_db/
    │  metadata indexed: provider_root, collection_id, title,
    │                    platforms, extent_temporal, extent_spatial, keywords
```

### 7.2 Grouped Query Embedding (Legacy/Secondary)

**Script**: `embed_and_load_grouped.py`  
**Input**: `kb/outputs/documents_embedding.jsonl` (204 docs) + enrichment cache  
**Output**: ChromaDB collection `copernicus_grouped`

> **Note**: `rag_chatbot_v3.py` uses only `stac_collections` (Track A). The `copernicus_grouped` collection from Track B is not used in v3 but was the primary source in v2.

### Embedding Model Details

| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Distance | Cosine similarity |
| Max sequence | 256 tokens |
| Language | English (multilingual via normalization) |
| Storage | ~2.3 MB per 1,000 vectors |

### ChromaDB Storage

```
chroma_db/
├── chroma.sqlite3          ← 2.8 GB total (metadata + index)
└── {18 UUID directories}/  ← HNSW index files per collection
    ├── data_level0.bin     ← Raw vectors
    ├── header.bin          ← HNSW graph header
    ├── link_lists.bin      ← HNSW neighbour lists
    ├── index_metadata.pickle
    └── length.bin
```

---

## 8. Stage 6 — Index Building

### 8.1 Collection Index

**Script**: `collection_indexer.py`  
**Input**: `collections.csv` (1,708 rows)  
**Outputs**:

| File | Content |
|---|---|
| `kb/outputs/collection_index.json` | Full catalog: collection_id → {title, description, satellite, themes, ...} |
| `kb/outputs/theme_index.json` | theme → [collection_ids] (reverse lookup) |
| `kb/outputs/sentinel_collection_map.json` | S1/S2/S3/S5P → Copernicus collection IDs |
| `kb/outputs/mission_theme_index.json` | Lightweight mission × theme map |

### 8.2 Query Lookup Index

**Script**: `build_query_lookup.py`  
**Input**: `kb/Search_queries_world_v2.json` (421,280 queries)  
**Output**: `kb/outputs/query_lookup.jsonl` — 300 MB compact index

Each record (one per query):
```json
{
  "sat": "S2", "bio": "forest", "cont": "south_america",
  "co": "Brazil", "loc": "Amazon River",
  "group_key": "S2|forest|south_america",
  "dataset": "sentinel-2-l2a",
  "loc_norm": "amazon river", "co_norm": "brazil",
  "ds": "2023-06-01", "de": "2023-08-31",
  "ds_int": 20230601, "de_int": 20230831,
  "lat": -3.4, "lon": -58.2,
  "url": "https://...", "body": {...}, "method": "POST"
}
```

> **Note**: `query_lookup.jsonl` is NOT used in `rag_chatbot_v3.py`. It was the primary retrieval mechanism in v2. In v3, live STAC API calls replace pre-built query lookup.

### 8.3 Geographic Index

**Script**: `build_geo_index.py`  
**Input**: `kb/outputs/query_lookup.jsonl`  
**Output**: `kb/outputs/geo_index.json` (8.5 MB)

**Used by**: `rag_chatbot_v3.py` → `resolve_bbox()` — converts city/region names to `[W, S, E, N]` bounding boxes.

Structure:
```json
{
  "locations": {
    "paris": {
      "name": "Paris", "country": "france", "continent": "europe",
      "lat": 48.8566, "lon": 2.3522,
      "dominant_biome": "urban",
      "dominant_context": "inland"
    }
  },
  "countries": {
    "france": { "continent": "europe", "location_count": 183 }
  },
  "continents": {
    "europe": { "count": 12840, "country_count": 43 }
  }
}
```

Resolution priority in `resolve_bbox()`:
1. Hardcoded named regions (`_NAMED_REGION_BBOXES`) — seas, continents, deserts
2. Exact city match in geo_index
3. Partial/substring city match
4. Country bbox from geo_index countries record
5. Country bbox aggregated from all city points (`build_country_bboxes()`)

---

## 9. Stage 7 — RAG Chatbot (v3)

**Script**: `rag_chatbot_v3.py` (3,500+ lines)  
**Dependencies at runtime**: ChromaDB, Ollama, stac_item_search.py, geo_index.json, stac_providers.jsonl

### Startup Sequence

```python
load_resources()
    ├─ SentenceTransformer("all-MiniLM-L6-v2")   → res.embed_model
    ├─ chromadb.PersistentClient("./chroma_db")
    │      .get_collection("stac_collections")    → res.stac_col  (5,999 docs)
    ├─ STACItemSearcher(STAC_PROVIDERS)           → res.stac_searcher
    │      reads kb/outputs/stac_providers.jsonl
    └─ json.load(GEO_INDEX)                       → res.geo_index
           build_country_bboxes()                 → res.country_bboxes
```

### Per-Turn Pipeline

```
User text
    │
    ▼  extract_intent(user_text, memory, model)
Ollama/mistral → JSON Intent object
    {satellite, theme, mission_type, location_text,
     country, date_start, date_end, mode, wants_items,
     wants_count, cloud_cover_max, collection_id, ...}
    │
    ▼  _keyword_enhance(intent, user_text)
Python rule-based corrections
    • imagery words → optical
    • display verbs → item_search
    • cloud cover regex → fill cloud_cover_max
    • year regex → fill date range
    • CMR-style IDs → move to collection_id
    • derive request_type
    │
    ▼  Route by request_type
    ├─ "collection_discovery"  → _handle_collection_discovery()
    ├─ "item_preview"          → _handle_item_search()
    ├─ "item_count"            → _handle_item_search()
    ├─ "item_links"            → _handle_item_search()
    ├─ "item_export"           → _handle_item_search() + _handle_export()
    └─ "item_by_id"            → _handle_item_by_id()
    │
    ▼  Update memory + save to Redis
```

### Mode A — Collection Discovery

```
search_stac_collections(intent, res)
    │
    ▼  Strategy A: exact lookup (if collection_id known)
    │      res.stac_col.get(where={"collection_id": id})
    │
    ▼  Strategy B: semantic vector search
    │      embed_model.encode([satellite + theme + mission_type])
    │      stac_col.query(query_embeddings=vec, n_results=25)
    │
    ▼  _rerank_hits_for_intent()
    │      +0.15 platform match | -0.50 mismatch
    │      +0.10 spatial overlap | -0.30 no overlap
    │      +0.10 temporal overlap | -0.30 no overlap
    │
    ▼  display_collection_cards()    → 5 cards shown
    ▼  synthesize_collection_results()  → LLM summary (Ollama)
```

### Mode B — Item Search

```
_handle_item_search()
    │
    ▼  Resolve collection (from intent or ChromaDB)
    ▼  resolve_bbox() — location_text / country → [W, S, E, N]
    ▼  Clarification check (if no bbox OR no date)
    │
    ▼  Confirm collection with user (if score < threshold)
    │
    ▼  STACItemSearcher.search()
    │      POST /search to provider STAC API
    │      payload: {collections, bbox, datetime, limit, cloud_cover}
    │
    ▼  display_item_cards() — 5 shown, "all items"/"more" offered
    ▼  synthesize_item_results() — LLM summary (Ollama)
```

### Memory System (Article Sections 2/3/4)

Three strategies selectable via `--memory {raw|summary|buffer}`:

| Strategy | Class | Article Section | Behaviour |
|---|---|---|---|
| `raw` | `RawMessageMemory` | Section 2 (OpenAI → local Ollama) | Full text, bounded list, oldest dropped |
| `summary` | `SummaryMemory` | Section 3 (LangChain ConversationSummaryMemory) | LLM generates rolling summary after each turn |
| `buffer` | `BufferedSummaryMemory` | Section 4 (Llama-Index ChatSummaryMemoryBuffer) | Last 6 turns verbatim + LLM summary of older ones |

Default: `buffer` (best balance of precision + compression).

### Session Persistence (Redis)

```python
SessionStore(session_id)
    ├─ connects to redis://localhost:6379
    ├─ key: "rag_session:{session_id}:history"
    ├─ TTL: 7 days
    └─ graceful fallback to in-memory if Redis unavailable
```

```bash
python3 rag_chatbot_v3.py --session alice   # separate history per user
python3 rag_chatbot_v3.py --session bob
```

### Slash Commands

| Command | Effect |
|---|---|
| `/help` | Show all commands |
| `/model <name>` | Switch Ollama model mid-session |
| `/top <n>` | Change number of collections shown |
| `/verbose` | Toggle intent debug output |
| `/clear` | Wipe history + Redis session |
| `/sessions` | List all Redis sessions |
| `/token <keyword> <token>` | Set bearer token for provider |

---

## 10. File Map — Every File Explained

### Python Scripts

| File | Purpose | Reads | Writes |
|---|---|---|---|
| `stac_ingest.py` | Crawl 17 STAC providers | Provider URLs (CSV) | `stac_providers.jsonl`, `stac_collections.jsonl`, `stac_items_sample.jsonl`, `stac_rag_documents.jsonl` |
| `stac_item_search.py` | Live STAC item search module | `stac_providers.jsonl` (at runtime) | `kb/exports/*.jsonl` (on export) |
| `collection_indexer.py` | CSV → structured indexes | `collections.csv` | `collection_index.json`, `theme_index.json`, `sentinel_collection_map.json`, `mission_theme_index.json` |
| `generate_queries.py` | Generate 421k search queries | GeoNames + Natural Earth | `Search_queries_world_v2.json` |
| `processor_grouped.py` | Group 421k → 204 documents | `Search_queries_world_v2.json`, `sentinel_collection_map.json` | `documents_embedding.jsonl`, `documents_full.json`, `query_groups.json` |
| `processor_standalone.py` | Backward-compat wrapper | — | delegates to `processor_grouped.py` |
| `enricher.py` | LLM enrichment for 204 groups | `documents_full.json` | `kb/enriched_groups/*.json` |
| `embed_stac.py` | Embed 5,999 STAC docs | `stac_rag_documents.jsonl` | ChromaDB `stac_collections` |
| `embed_and_load.py` | Embed individual queries (legacy) | `stac_rag_documents.jsonl` | ChromaDB `copernicus_rag` |
| `embed_and_load_grouped.py` | Embed 204 grouped docs | `documents_embedding.jsonl`, enrichment cache | ChromaDB `copernicus_grouped` |
| `build_query_lookup.py` | Build compact query index | `Search_queries_world_v2.json` | `query_lookup.jsonl` |
| `build_geo_index.py` | Build geographic index | `query_lookup.jsonl` | `geo_index.json` |
| `rag_chatbot_v2.py` | Chatbot v2 (grouped + query lookup) | ChromaDB, `query_lookup.jsonl`, `geo_index.json` | terminal output |
| `rag_chatbot_v3.py` | **Chatbot v3 (STAC-first, live API)** | ChromaDB `stac_collections`, `stac_providers.jsonl`, `geo_index.json` | terminal output, `kb/exports/` |
| `eval_retrieval.py` | Evaluate retrieval quality | ChromaDB | reports |
| `patch_json.py` | Patch JSON files | JSON files | patched JSON |

### Key Configuration

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies |
| `kb/outputs/stac_providers.jsonl` | Runtime provider config for chatbot |

---

## 11. Generated JSON/JSONL Files Reference

### `kb/outputs/stac_providers.jsonl` — 17 records
```json
{
  "provider_root": "https://stac.terrascope.be",
  "title": "Terrascope",
  "search_url": "https://stac.terrascope.be/search",
  "search_method": "POST",
  "access_type": "",
  "conforms_to": ["https://api.stacspec.org/v1.0.0/item-search", ...]
}
```
**Used by**: `rag_chatbot_v3.py` → `STACItemSearcher` to route POST/GET and auth.

---

### `kb/outputs/stac_collections.jsonl` — 5,999 records
Full raw collection metadata from all providers. Basis for `stac_rag_documents.jsonl`.

---

### `kb/outputs/stac_rag_documents.jsonl` — 5,999 records
```json
{
  "doc_id": "https://stac.terrascope.be::terrascope-s2-ndvi-v2",
  "text": "Collection: Terrascope Sentinel-2 NDVI V2\nID: terrascope-s2-ndvi-v2\n...",
  "collection_id": "terrascope-s2-ndvi-v2",
  "provider_root": "https://stac.terrascope.be",
  "title": "Terrascope Sentinel-2 NDVI V2",
  "platforms": "sentinel-2a,sentinel-2b",
  "extent_temporal": "2015-07-04T00:00:00Z/..",
  "extent_spatial": "[-180.0, -90.0, 180.0, 90.0]"
}
```
**Used by**: `embed_stac.py` → ChromaDB `stac_collections`

---

### `kb/outputs/geo_index.json` — 8.5 MB
```json
{
  "locations": {
    "paris": { "name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "france" },
    "toulouse": { "name": "Toulouse", "lat": 43.6047, "lon": 1.4442, "country": "france" }
  },
  "countries": {
    "france": { "continent": "europe", "location_count": 183 }
  }
}
```
**Used by**: `rag_chatbot_v3.py` → `resolve_bbox()` at every item search.

---

### `kb/outputs/collection_index.json` — 1,337 records
```json
{
  "sentinel-2-l2a": {
    "title": "Sentinel-2 Level-2A",
    "satellite": "S2",
    "themes": ["vegetation", "land cover", "NDVI", ...],
    "description": "..."
  }
}
```
**Used by**: chatbot for theme-based collection filtering.

---

### `kb/outputs/sentinel_collection_map.json`
```json
{
  "S1": { "dataset": "sentinel-1-global-mosaics", "mission_type": "radar", "themes": [...] },
  "S2": { "dataset": "sentinel-2-l2a",            "mission_type": "optical", "themes": [...] },
  "S3": { "dataset": "sentinel-3-olci-1-efr-ntc", "mission_type": "ocean",  "themes": [...] },
  "S5P":{ "dataset": "sentinel-5p-l2-no2-rpro",  "mission_type": "atmospheric", "themes": [...] }
}
```
**Used by**: `processor_grouped.py`, `enricher.py`, query routing logic.

---

### `kb/outputs/query_groups.json` — 204 groups
```json
{
  "S2|forest|south_america": [
    { "url": "...", "body": {...}, "loc": "Amazon River", "lat": -3.4, "lon": -58.2 },
    ...20 samples...
  ]
}
```
**Used by**: `rag_chatbot_v2.py` for showing real API examples.  
**Not used by v3** (v3 makes live calls instead).

---

### `kb/outputs/query_lookup.jsonl` — 421,280 records (300 MB)
Used for exact query matching in v2. Not loaded by v3 (too large, replaced by live STAC calls).

---

### `kb/exports/stac_export_*.jsonl` — generated on demand
Created by `rag_chatbot_v3.py` when user types `export`. Contains full STAC items with all assets.

---

## 12. ChromaDB Collections Reference

### `stac_collections` — PRIMARY (used by v3)

| Property | Value |
|---|---|
| Records | 5,999 |
| Vector dims | 384 |
| Distance | Cosine |
| Embedding text | Title + description + platforms + keywords + spatial/temporal extent |
| Metadata stored | provider_root, collection_id, title, platforms, extent_temporal, extent_spatial, keywords, crawl_ts |
| Script that built it | `embed_stac.py` |
| Used by | `rag_chatbot_v3.py` |

Query at runtime:
```python
res.stac_col.query(
    query_embeddings = embed_model.encode(["SAR flood radar"]).tolist(),
    n_results        = 25,
    include          = ["documents", "metadatas", "distances"],
)
```

### `copernicus_grouped` — SECONDARY (used by v2)

| Property | Value |
|---|---|
| Records | 204 |
| Embedding text | Group description + themes + synonyms + enrichment |
| Used by | `rag_chatbot_v2.py` only |

### `copernicus_rag` — LEGACY (not used)

Individual query embeddings. Superseded by `stac_collections`.

---

## 13. End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW INPUT SOURCES                                  │
├──────────────────────┬──────────────────────┬─────────────────────────────── │
│  Provider URLs (CSV) │  GeoNames 11M locs   │  Natural Earth GeoJSON        │
│  collections.csv     │  allCountries.txt    │  ne_*.geojson                 │
└──────────┬───────────┴──────────┬───────────┴────────────┬──────────────────┘
           │                      │                          │
           ▼                      ▼                          ▼
  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
  │  stac_ingest.py │   │ generate_        │   │ collection_indexer.py    │
  │  (crawl 17      │   │ queries.py       │   │ (CSV → indexes)          │
  │   providers)    │   │ (421k queries)   │   │                          │
  └────────┬────────┘   └────────┬─────────┘   └──────────────────────────┘
           │                      │                    │
     5,999 docs               421k queries             ├─ collection_index.json
     (JSONL)                  (JSON)                   ├─ theme_index.json
           │                      │                    └─ sentinel_collection_map.json
           │              ┌───────▼──────────┐
           │              │ processor_        │
           │              │ grouped.py        │
           │              │ (421k → 204 docs) │
           │              └───────┬──────────┘
           │                      │
           │              ┌───────▼──────────┐
           │              │ enricher.py       │
           │              │ (LLM enrichment)  │
           │              └───────┬──────────┘
           │                      │ 204 enriched docs
           │                      │
    ┌──────▼──────────────────────▼──────┐
    │         EMBEDDING STAGE             │
    │                                     │
    │  embed_stac.py       embed_and_     │
    │  (5,999 docs)        load_grouped   │
    │       ↓              (204 docs)     │
    │  stac_collections    copernicus_    │
    │  (PRIMARY, v3)       grouped (v2)   │
    └──────────────────────┬──────────────┘
                           │
    ┌──────────────────────▼──────────────────┐
    │              ChromaDB                    │
    │  chroma_db/chroma.sqlite3 (2.8 GB)       │
    │  ┌─────────────────────────────────────┐ │
    │  │ stac_collections: 5,999 × 384-dim  │ │
    │  │ copernicus_grouped: 204 × 384-dim  │ │
    │  └─────────────────────────────────────┘ │
    └──────────────────────┬──────────────────┘
                           │
    ┌──────────────────────▼──────────────────┐
    │         rag_chatbot_v3.py               │
    │                                          │
    │  + stac_providers.jsonl (routing)        │
    │  + geo_index.json (bbox resolution)      │
    │  + Ollama/mistral (intent + synthesis)   │
    │  + Redis (session persistence)           │
    │                                          │
    │  User ←→ Terminal REPL                   │
    └──────────────────────────────────────────┘
```

---

## 14. Key Design Decisions

### Why two ChromaDB collections?

`stac_collections` (5,999) indexes **real provider metadata** — actual collection IDs, search URLs, temporal/spatial extents. This is what `rag_chatbot_v3.py` uses because it needs the provider URL to make live API calls.

`copernicus_grouped` (204) indexes **semantic groups** — richer natural-language descriptions of satellite capabilities per biome and continent. Better for answering "what data exists for..." but doesn't contain provider URLs.

v3 uses Track A exclusively; v2 used Track B.

### Why `all-MiniLM-L6-v2`?

- Fast (384 dims vs 768 for larger models)
- Good enough for short technical texts (collection titles + descriptions)
- Same model used at index time and query time → no dimension mismatch
- Runs CPU-only, no GPU required

### Why Ollama/mistral locally?

- No API cost
- No data privacy concerns (satellite metadata stays local)
- Can switch models mid-session (`/model llama3`)
- Works offline after initial pull

### Why live STAC calls instead of pre-built query lookup?

The `query_lookup.jsonl` (300 MB, 421k queries) was the retrieval mechanism in v2 — find the closest pre-built query, return its API URL. This has two problems:
1. Query dates are static (baked in at generation time)
2. Coverage gaps (not every location × satellite combination exists)

v3 makes live `POST /search` calls with the exact bbox + datetime the user requested. Real counts, real items, real download links.

### Why compact structured summaries in history?

The history stored per turn is not the full assistant response but a compact summary:
```
mode=item_search sat=S1 theme=flood loc=france date=2024-01-01 | searched_collection=opera-s1-rtc-v1
```
This keeps the context window small for intent extraction while preserving the key facts the LLM needs to resolve pronouns ("the same collection", "that one", "same but in 2023").

---

## 15. Dependencies

```
sentence-transformers>=3.0.0    # Embedding model (all-MiniLM-L6-v2)
faiss-cpu>=1.8.0               # HNSW vector index (used by ChromaDB internally)
numpy>=1.26.0                  # Numeric operations
requests>=2.32.0               # HTTP (STAC crawling fallback)
dateparser>=1.2.0              # Natural language date parsing
geopy>=2.4.0                   # Geographic utilities
pydantic>=2.7.0                # Data validation (enricher, item parsing)
ollama>=0.4.0                  # Local LLM interface (mistral/llama3)
chromadb>=0.5.0                # Vector database
redis>=4.0.0                   # Session persistence (optional)
pycountry                      # Country name normalization
```

### Ollama Models Used

| Model | Purpose |
|---|---|
| `mistral` (default) | Intent extraction + synthesis |
| `llama3` (alternative) | Faster responses, slightly less precise |

```bash
# Pull models
/snap/ollama/122/bin/ollama pull mistral
/snap/ollama/122/bin/ollama pull llama3

# Start server
/snap/ollama/122/bin/ollama serve
```

### Infrastructure

| Component | Setup |
|---|---|
| ChromaDB | Persistent, `./chroma_db/` (no server needed) |
| Redis | Optional, `sudo apt install redis-server && redis-server --daemonize yes` |
| Ollama | `/snap/ollama/122/bin/ollama serve` (or add to `~/.bashrc`) |

---

*Last updated: 2026-05-25*
