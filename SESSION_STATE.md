# SESSION STATE — Copernicus RAG Chatbot
_Last updated: 2026-04-14 (session: testing, model comparison, architecture analysis)_
_(Previous backup: SESSION_STATE_backup_20260414.md)_

---

## 1. Active Pipeline

```
generate_queries.py  →  kb/Search_queries_world.json  (421,280 entries)
processor_standalone.py  →  kb/outputs/documents_embedding.jsonl
embed_and_load.py  →  chroma_db/  (421,279 docs, collection: copernicus_rag)
rag_chatbot.py  →  parse_intent → ChromaDB → Ollama (mistral or llama3)
```

---

## 2. ChromaDB State

| Field | Value |
|---|---|
| Documents | 421,279 |
| Embedding model | all-MiniLM-L6-v2 (384-dim cosine) |
| Key metadata | satellite, country, continent, biome, date_start, date_end, date_start_int, date_end_int, duration_days, region, season |
| Country fill rate | ~96.8% |
| Date variants | E: 7–29d · A: 30–60d · B: 61–120d · C: 121–365d · D: 366–730d |

> **IMPORTANT — Rebuild required:**
> - France and Norway missing (ISO_A2_EH fix in code, not in DB)
> - Variant E (7–29d) absent → single-month exact hits = 0
> - Country normalisation mismatches
> - Region names (Mediterranean, Arctic, Amazon) not filterable — country = N/A for ocean

---

## 3. Git Status

| Item | Status |
|---|---|
| Branch | `main` |
| Remote | `https://gitlab.com/HichriZeineb/rag_project.git` |
| Latest commit | `b935dde` — fuzzy synonym matching + deforestation keywords |
| Push to GitLab | ❌ Pending — needs GitLab token |

**Commits this session:**
```
b935dde  add fuzzy synonym matching and deforestation keywords
cacdbb0  overhaul clarification engine, satellite detection, and refinement
fc0c5ee  responses correction : what when where
```

**To push:**
```bash
git push https://HichriZeineb:<YOUR_TOKEN>@gitlab.com/HichriZeineb/rag_project.git main --force
```

---

## 4. File States

| File | State | Notes |
|---|---|---|
| `rag_chatbot.py` | ✅ Committed | All session fixes applied |
| `generate_queries.py` | ✅ Stable | All 6 bugs fixed in earlier session |
| `processor_standalone.py` | ✅ Stable | No changes |
| `embed_and_load.py` | ✅ Stable | No changes |
| `config.py` | ✅ Stable | No changes |
| `requirements.txt` | ✅ Stable | No changes |
| `chroma_db/` | ⚠️ Needs rebuild | See ChromaDB state above |
| `kb/Search_queries_world.json` | ✅ Present | 421,280 entries, excluded from git |
| `kb/outputs/documents_embedding.jsonl` | ✅ Present | 360 MB, excluded from git |
| `kb/geodata/` | ✅ Present | 5 GeoJSON files committed |

---

## 5. All Bugs Fixed (21 total)

| ID | Bug | Status |
|---|---|---|
| BUG-1 | `_parse_dates` missed `MONTH to MONTH YEAR` form | ✅ |
| BUG-2 | Partial hits shown without label | ✅ |
| BUG-3 | London → Canada geo collision | ✅ code only, rebuild pending |
| BUG-4 | Clarification never waited for reply | ✅ |
| BUG-5 | City ambiguity rule wrong priority | ✅ |
| BUG-6 | Cloud-cover clarification too aggressive | ✅ R3 removed |
| BUG-7 | Retrieval capped at n=20 | ✅ |
| BUG-8 | LLM hallucinated counts/results | ✅ `_print_results()` |
| BUG-9 | Vague queries returned noise | ✅ clarification rules |
| BUG-10 | `collection.count(where=...)` not supported | ✅ |
| BUG-11 | France/Norway absent from JSON | ✅ code only, rebuild pending |
| BUG-12 | Country name mismatches | ✅ `_NAME_NORMALIZE` + aliases |
| BUG-13 | "show me more" triggered new search | ✅ |
| BUG-14 | LLM ignored partial results | ✅ `_print_results()` |
| BUG-15 | `--ask` mode ignored clarification | ✅ interactive loop added |
| BUG-16 | Rule 0 swallowed Rule 1 (city ambiguity) | ✅ all rules evaluated |
| BUG-17 | Fully vague queries returned all 421k docs | ✅ R2/R3/R4 combined |
| BUG-18 | Typos in satellite names not detected | ✅ fuzzy pass A |
| BUG-19 | "only germany" re-displayed wrong results | ✅ re-search on new filter |
| BUG-20 | "show more" showed same items again | ✅ offset tracking |
| BUG-21 | Typos in application keywords not detected | ✅ fuzzy pass B |

---

## 6. Satellite Detection — `parse_intent` (3 passes)

### Pass 1 — Explicit codes
```
sentinel-1/2/3/5p  sentinel1/2/3/5p  sentinel 1/2/3/5p
sentinel-5 / sentinel5 / sentinel 5  (→ S5P)
s1  s2  s3  s5p  s5
```

### Pass 2 — Copernicus mission synonyms (official categorisation)
| Satellite | Keywords |
|---|---|
| S1 | radar, sar, synthetic aperture, backscatter, all-weather, flood/s, maritime, ships, sea ice, deformation, subsidence, landslide/s, disaster |
| S2 | optical, multispectral, high-resolution, vegetation, agriculture, crops, soil, land cover, forest/s, deforestation, reforestation, tree cover, wildfire, fire, inland waterways, water cover, emergency |
| S3 | marine, oceanography, ocean colour/color, ocean monitoring, sea surface temperature, sea level, altimetry, land surface temperature, land colour/color, vegetation index, climate |
| S5P | atmosphere/ic, atmospheric composition, air quality, pollution, trace gas/es, no2, nitrogen dioxide, so2, co, carbon monoxide, co2, methane, ch4, ozone, formaldehyde, hcho, aerosol/s, greenhouse, emission/s |

### Pass 3 — Fuzzy matching (difflib cutoff=0.82)
- **Pass A:** words/bigrams vs SATELLITE_CODES keys ≥6 chars (catches `sentinal-2`)
- **Pass B:** words/bigrams vs SATELLITE_MISSION_SYNONYMS keys ≥4 chars (catches `deforesation`, `flod`)
- Guard: code with digit but candidate without → skip

---

## 7. Clarification Engine — `_needs_clarification`

All rules evaluated every time, combined into one message.

| Rule | Trigger | Question |
|---|---|---|
| R1 | Ambiguous city (18 cities) | "Which X? A or B?" |
| R2 | No satellite + "sentinel" word | "Which Sentinel? (1/2/3/5P)" |
| R2 | No satellite, no sentinel | "Which satellite? (S1/S2/S3/S5P)" |
| R3 | No location | "Which region?" |
| R4 | No date | "Which time period?" |
| R5 | Multi-month + no location | "Which region for this range?" |

Location detection uses fuzzy matching (cutoff=0.82).

---

## 8. Model Comparison — Mistral vs LLaMA3

**Context:** LLM is used ONLY for the final 1-3 sentence comment. All retrieval is Python.

| Criterion | Mistral | LLaMA3 |
|---|---|---|
| Faithfulness to counts | ✅ | ✅ |
| Correct exact/partial definitions | ✅ better | ❌ invents spatial meaning |
| Hallucination type | Technical facts ("daily imaging cycle") | Interpretive drift ("fragmented coverage") |
| Scientific value added | ❌ weak | ❌ weak |
| Fluency | Slightly awkward | More natural |
| Danger level | High — sounds authoritative | Medium — vague but softer |
| **Overall** | **Winner** | **Loses on faithfulness** |

**Key conclusions:**
- Mistral wins for structured/technical data — more faithful to results
- LLaMA3's hallucination is softer but still misleading for non-experts
- **Neither model is the real problem — the system prompt is under-constrained**
- Fixing the prompt will improve both more than switching models

---

## 9. Architecture Analysis

### Current architecture
```
query → Python parse_intent → ChromaDB filter+vector → _print_results → LLM comment
```

### Weaknesses identified
1. **LLM prompt too loose** → hallucinates technical details in comment
2. **No keyword search** → vector-only, misses exact region name matches (Mediterranean)
3. **No recommendation logic** → "which satellite for X?" treated as search, not advice
4. **LLM not used for filter building** → Python fails silently for unknown patterns

### Architecture recommendation
```
query
  ↓
Intent Parser (Python — current, keep as primary)
  ↓ if sat=None after all passes
LLM Filter Builder (fallback only — new)
  ↓
Hybrid Retrieval:
  metadata filter (current) + keyword on region field (new) + vector (current)
  ↓
Recommendation Layer (new — for "which satellite?" queries)
  ↓
LLM Comment (constrained prompt — fix this first)
```

---

## 10. Known Remaining Issues

| ID | Issue | Severity | Fix |
|---|---|---|---|
| ISSUE-1 | France/Norway missing from ChromaDB | High | Pipeline rebuild |
| ISSUE-2 | Variant E (7–29d) absent | Medium | Pipeline rebuild |
| ISSUE-3 | Legacy files not deleted | Low | Manual rm |
| ISSUE-4 | Region names not filterable (Mediterranean etc.) | Medium | Keyword search on region field |
| ISSUE-5 | LLM prompt under-constrained → hallucination | High | Prompt engineering |
| ISSUE-6 | No recommendation logic | Medium | Recommendation layer |

---

## 11. Next Steps (priority order)

| # | Task | Priority | Effort |
|---|---|---|---|
| 1 | Fix LLM system prompt (ISSUE-5) | 🔴 High | 1 day |
| 2 | Add keyword search on `region` field (ISSUE-4) | 🔴 High | 2–3 days |
| 3 | Build recommendation layer (ISSUE-6) | 🟡 Medium | 2–3 days |
| 4 | Push to GitLab | 🟡 Medium | 30 min |
| 5 | Pipeline rebuild (ISSUE-1, ISSUE-2) | 🟡 Medium | 1–2 days |
| 6 | LLM fallback for filter building | 🟢 Low | 2 days |
| 7 | Delete legacy files | 🟢 Low | 10 min |

---

## 12. Files to delete (not yet done)

```bash
rm intent_parser.py retriever.py ingest.py generate_queries_v1_cities_only.py 2>/dev/null
rm "kb/Search queries.json" kb/Search_queries_world_v1_cities_only.json 2>/dev/null
rm kb/records.json kb/index.faiss 2>/dev/null
rm kb/outputs/documents_full.json kb/outputs/documents_vectorstore.jsonl 2>/dev/null
rm SESSION_STATE_backup_20260409.md SESSION_STATE_backup_20260409b.md 2>/dev/null
rm SESSION_STATE_backup_20260409c.md SESSION_STATE_backup_20260409d.md 2>/dev/null
rm "command (5wfplj)" 2>/dev/null
```

---

## 13. Full Model & Architecture Analysis (session 2026-04-14)

### Part A — Model evaluation conclusions

**The comparison is valid but prompt-constrained** — both models are limited by a weak system
prompt, not by their own capabilities. The test is not a fair model capability test; it is a
test of how each model behaves with under-specified instructions.

**Which failure is more dangerous?**

| | Mistral | LLaMA3 |
|---|---|---|
| Error type | Invents specific technical facts ("daily imaging cycle") | Adds vague spatial interpretation ("fragmented coverage") |
| Sounds authoritative? | Yes — specific, confident | No — vague |
| Expert detects it? | Hard — sounds plausible | Easy — obviously wrong |
| Non-expert trusts it? | Yes — may make wrong decisions | Probably, but less harmful |
| Domain risk | User plans a mission on wrong specs | User gets fuzzy explanation |

→ **Mistral hallucination is more dangerous** because it is confident and specific.
LLaMA3's errors are soft and vague. In a technical satellite domain, confident misinformation causes worse decisions.

**Is Mistral better, or is this a prompt issue?**
Mostly a prompt issue. Add three explicit constraints and both models improve dramatically.
Fix the prompt before any model decision.

---

### Part B — Architecture conclusions

**Architecture impact >> model impact**

The LLM currently writes 2 sentences after all real work is done.
Changing models changes those 2 sentences.
Changing the architecture changes every query, every result, every output.
These are not comparable.

**LLM-assisted retrieval (when Python fails):**
- Pro: handles queries outside hardcoded rules
- Pro: no need to maintain synonym dictionaries forever
- Con: adds latency, makes system non-deterministic, harder to debug
- Verdict: use as fallback only when sat=None after all passes

**Hybrid RAG (keyword + vector + metadata):**
- Directly fixes the Mediterranean problem (ISSUE-4)
- High impact, medium effort
- More impactful than switching models
- Verdict: implement after prompt fix

**Knowledge graph / ontology:**
- Full GraphRAG is overkill for this project
- A structured RECOMMENDATIONS dict (Python) is sufficient
- Already have the seed: SATELLITE_MISSION_SYNONYMS
- Verdict: build a lightweight recommendation layer, not a graph DB

---

### Part C — Priority order (confirmed)

```
1. Fix LLM prompt          → stops hallucination today (1 day)
2. Keyword/region search   → fixes Mediterranean, Arctic, Amazon (2–3 days)
3. Recommendation layer    → evolves system to assistant (2–3 days)
4. LLM filter fallback     → handles unknown patterns (1 week)
5. Query rewriting         → improves vague queries (optional)
6. Intent classifier       → separates search vs recommend vs compare (optional)
7. Model change            → LAST — only after prompt + architecture are solid
```

---

### Part D — Target architecture (full roadmap)

```
User query
    │
    ▼
┌──────────────────────────────────────────┐
│  Intent Classifier  (new — lightweight)  │
│  search | recommend | compare | explain  │
└──────────────┬───────────────────────────┘
               │
         ┌─────┴───────┐
         │             │
    [recommend]    [search]
         │             │
         ▼             ▼
┌──────────────┐  ┌───────────────────────────────┐
│ Recommend    │  │ Intent Parser (Python, keep)   │
│ Layer        │  │ 3-pass satellite + fuzzy       │
│ (new dict)   │  │ date + multi-country           │
└──────────────┘  └──────────┬────────────────────┘
                              │ if sat=None after all passes
                              ▼
                  ┌───────────────────────────────┐
                  │ LLM Filter Fallback (new)      │
                  │ query → {satellite, filters}   │
                  └──────────┬────────────────────┘
                              │
                              ▼
                  ┌───────────────────────────────┐
                  │ Hybrid Retrieval (new)         │
                  │ - metadata filter (done)       │
                  │ - keyword on region (missing)  │
                  │ - vector search (done)         │
                  └──────────┬────────────────────┘
                              │
                              ▼
                  ┌───────────────────────────────┐
                  │ LLM Comment (fix prompt now)   │
                  │ constrained, 1–2 sentences     │
                  └───────────────────────────────┘
```

**Other approaches worth exploring (future):**
- Query rewriting — LLM rephrases vague query before parsing (high value, low effort)
- Reranking — reorder results after retrieval (medium value, medium effort)
- Conversation memory — remember user preferences across sessions
- Streaming output — show results as they arrive (low effort)
- Full agent loop — LLM decides which tool to call (high effort, future only)

---

## 14. Prompt for Next Claude Code Session

Copy this entire block and paste it as your first message:

---

```
I am working on a local Copernicus RAG chatbot in /home/zhich/dataionics_rag.

Start by reading SESSION_STATE.md to understand the full project state and analysis.

---

## Project summary

Pipeline: user query → Python parse_intent → ChromaDB (421k docs) → Ollama LLM comment
The LLM (Mistral via Ollama) ONLY writes a 1-2 sentence comment at the end.
All retrieval, filtering, clarification, and result display is handled by Python.
Main file: rag_chatbot.py (~1350 lines)

---

## Context: what has been analysed (do not redo this analysis)

We did a full model + architecture analysis. Key conclusions:

MODEL VERDICT: Mistral > LLaMA3 for this project because:
- Mistral is more faithful to structured retrieval output
- LLaMA3 drifts toward spatial/coverage interpretations not supported by data
- HOWEVER both are limited by a weak system prompt, not by their own capabilities
- Fix the prompt first before any model decision

ARCHITECTURE VERDICT:
- Architecture change has far more impact than model change
- LLM writes 2 sentences — switching models changes 2 sentences
- Better architecture changes every query and every result
- Priority: fix prompt → hybrid search → recommendation layer → LLM fallback

HALLUCINATION ANALYSIS (observed in testing):
- Mistral: "daily imaging cycle" → wrong (S2 has 5-day revisit, not daily)
- Mistral: "location is exact" → wrong (partial match is TIME overflow, not spatial)
- LLaMA3: "fragmented coverage" → wrong (partial is time-only, never spatial)
- Root cause: current system prompt says "time overflow only, location is exact" which
  misleads LLMs into writing spatial comments. Also no ban on revisit cycle mentions.

---

## What was done in previous sessions (do not redo implementation)

- 3-pass satellite detection with fuzzy matching (handles typos like "deforesation")
- Full clarification engine: asks for satellite + region + date when missing
- Copernicus-accurate keyword mapping for S1/S2/S3/S5P
- Multi-country support with $or ChromaDB filter
- Pagination: "show more" advances offset, does not repeat same results
- Refinement re-search when user adds "only germany" type filter

---

## TASK 1 — Fix LLM system prompt (HIGHEST PRIORITY)

Location: rag_chatbot.py → function build_prompt() → variable `system`

Problem: current system prompt contains "time overflow only, location is exact" on line 1017
which causes LLMs to write spatial interpretations. No ban on technical hallucination.

Proposed fix (show me this before applying, I will confirm):

  [ROLE]
  You are a specialist assistant for the Copernicus satellite data RAG system.
  Answer in the same language the user used.

  [DATA — Satellite mission types]
    S1  = Sentinel-1  = Radar SAR imaging — works day/night through clouds
    S2  = Sentinel-2  = Optical multispectral imaging — requires daylight and clear sky
    S3  = Sentinel-3  = Ocean and land surface monitoring
    S5P = Sentinel-5P = Atmospheric gas monitoring (NO2, methane, ozone, aerosols)
  If the user asks for "S2 radar": correct them — S2 is OPTICAL. Radar is S1.
  If the user asks for "S1 optical": correct them — S1 is RADAR. Optical is S2.

  [DEFINITIONS — exact and partial]
    EXACT match   = the document's date range falls ENTIRELY within the requested time window.
                    This is a time-only definition. Nothing about location or coverage.
    PARTIAL match = the document's date range OVERLAPS the requested window but extends
                    beyond it (starts earlier OR ends later). Time boundary only — NOT
                    spatial, NOT coverage, NOT fragmentation.

  [STRICT RULES]
  The results have already been printed to the user.
  Your ONLY job: write 1–2 sentences maximum.
    - State what was found: satellite, location, time period.
    - If exact = 0 and partial > 0: say the documents overlap the period but extend beyond it.
    - If both = 0: say no documents were found. Do NOT speculate why.

  FORBIDDEN — never write any of the following:
    - satellite revisit cycles ("5-day revisit", "daily imaging", "temporal resolution")
    - imaging frequency, orbital passes, acquisition gaps, coverage patterns
    - "fragmented coverage", "spatial coverage", "location is exact"
    - any technical explanation for why counts are high or low
    - anything not directly shown in the printed results

After applying, test with:
  python3 rag_chatbot.py --ask "S2 vegetation France march 2025"
  python3 rag_chatbot.py --ask "S1 Spain february 2026"
  python3 rag_chatbot.py --ask "S3 ocean mediterranean 2025"

For each test:
  - Does the LLM comment state satellite + location + period correctly?
  - Does it avoid "daily imaging", "fragmented coverage", "location is exact"?
  - Is it 1-2 sentences, nothing more?

---

## TASK 2 — Add region→countries mapping (fixes Mediterranean, Arctic, Amazon)

Location: rag_chatbot.py → parse_intent() and retrieve_and_filter()

Problem: "Mediterranean" is in _GEO_ANCHOR so no clarification is triggered,
but _extract_country("mediterranean") returns None → no ChromaDB filter →
vector-only search returns random ocean results (Bering Sea, Tsugaru Strait etc.)

Two approaches — evaluate both and explain trade-offs to me before implementing:

Approach A — REGION_COUNTRIES dict (recommended):
  REGION_COUNTRIES = {
    "mediterranean": ["france", "spain", "italy", "greece", "turkey", "morocco",
                      "algeria", "tunisia", "libya", "egypt", "croatia", "albania"],
    "arctic": ["norway", "sweden", "finland", "russia", "canada", "greenland", "iceland"],
    "amazon": ["brazil", "peru", "colombia", "venezuela", "ecuador", "bolivia"],
    "north sea": ["uk", "norway", "denmark", "germany", "netherlands", "belgium"],
    "baltic": ["sweden", "finland", "estonia", "latvia", "lithuania", "poland", "germany"],
    "black sea": ["turkey", "romania", "bulgaria", "ukraine", "russia", "georgia"],
    "sahel": ["mauritania", "mali", "niger", "chad", "sudan", "senegal", "burkina faso"],
  }
  When a region is detected → expand to $or country filter, same as multi-country

Approach B — ChromaDB $contains on region field:
  {"region": {"$contains": "mediterranean"}}
  Check first: does our ChromaDB version support $contains on string fields?
  Run: python3 -c "import chromadb; print(chromadb.__version__)"

Show me which approach you recommend and why, then implement.

---

## TASK 3 — Build recommendation layer

Location: new function _handle_recommendation() in rag_chatbot.py

Problem: "which satellite for deforestation?" is treated as a search → wrong.
Should return a structured recommendation, skip ChromaDB entirely.

Detection: phrases like "which satellite", "what data", "best for", "recommend",
"should i use", "what should i use for", "what satellite"

Recommendation dict (seed — we will expand it together):
  RECOMMENDATIONS = {
    "deforestation":    {"satellite": "S2", "dataset": "sentinel-2-l2a",
                         "why": "10m optical, 13 bands, NDVI/EVI for vegetation change",
                         "tip": "Use Red-Edge bands B05/B06/B07 for precise canopy health"},
    "flood":            {"satellite": "S1", "dataset": "sentinel-1-grd",
                         "why": "SAR works through clouds day+night — critical for disasters",
                         "tip": "IW mode, VV+VH polarisation for flood extent mapping"},
    "air quality":      {"satellite": "S5P", "dataset": "s5p-l2-no2",
                         "why": "Daily global NO2, SO2, CO, methane, ozone at 7x3.5km",
                         "tip": "Use L2 OFFL product for offline processed daily data"},
    "ocean":            {"satellite": "S3", "dataset": "sentinel-3-olci-l1b",
                         "why": "Ocean color, SST, sea level altimetry, global coverage",
                         "tip": "OLCI for ocean colour, SLSTR for sea surface temperature"},
    "wildfire":         {"satellite": "S2", "dataset": "sentinel-2-l2a",
                         "why": "Optical detects burn scars and active fire in RGB/SWIR",
                         "tip": "Band 12 (SWIR) highlights active burn areas"},
    "glacier":          {"satellite": "S1", "dataset": "sentinel-1-grd",
                         "why": "SAR detects ice motion, surface deformation, crack patterns"},
    "urban":            {"satellite": "S2", "dataset": "sentinel-2-l2a",
                         "why": "10m resolution captures urban sprawl and land use change"},
    "pollution":        {"satellite": "S5P", "dataset": "s5p-l2-no2",
                         "why": "Tropospheric NO2/SO2 columns — industrial + transport pollution"},
    "agriculture":      {"satellite": "S2", "dataset": "sentinel-2-l2a",
                         "why": "NDVI, LAI, chlorophyll for crop monitoring across seasons"},
    "soil moisture":    {"satellite": "S1", "dataset": "sentinel-1-grd",
                         "why": "SAR backscatter correlates with soil moisture content"},
  }

---

## Learning goals (explain as you implement)

For each task, explain:
1. WHY this choice was made over alternatives
2. What the trade-offs are
3. How to evaluate if it works or not
4. What signals would tell us to try a different approach

Specifically I want to understand:
- How to choose between model-side vs architecture-side fixes
- How to evaluate LLM output quality for structured data systems
- When hybrid search is worth the complexity
- How to decide if a recommendation layer needs a knowledge graph or not
- What "prompt engineering" means concretely — not theory, specific rules that work

---

## Key functions (read before changing anything)

parse_intent(question) → {satellite, date_start, date_end, country, countries}
_needs_clarification(intent, question) → str or None
retrieve_and_filter(question, collection, embed_model, top_k)
build_prompt(...) → messages list for Ollama    ← TASK 1 is here
_print_results(..., exact_offset=0, partial_offset=0) → prints to terminal
_is_refinement(text, last_results) → bool

## ChromaDB metadata fields available for filtering
satellite, country, continent, biome, date_start, date_end,
date_start_int, date_end_int, duration_days, region, season

## Available Ollama models
- mistral:latest (default)
- llama3:latest

## Do not modify
generate_queries.py, processor_standalone.py, embed_and_load.py, config.py

## Instructions
1. Read SESSION_STATE.md first (already done if you see this)
2. Read build_prompt() in rag_chatbot.py (lines ~995–1041)
3. Show proposed system prompt change before applying — wait for my confirmation
4. Apply TASK 1, run the 3 test queries, show me the LLM comments
5. Compare comments: does the new prompt fix the hallucination observed before?
6. Then explain TASK 2 trade-offs and proceed
7. Keep explaining your reasoning as you go — this is a learning session
```
