# SESSION STATE — Copernicus RAG Chatbot
_Last updated: 2026-04-13 (session: clarification engine + satellite detection overhaul)_
_(Previous state saved to SESSION_STATE_backup_20260409f.md)_

---

## 1. Active Pipeline

```
generate_queries.py  →  kb/Search_queries_world.json  (421,280 entries)
processor_standalone.py  →  kb/outputs/documents_embedding.jsonl
embed_and_load.py  →  chroma_db/  (421,279 docs, collection: copernicus_rag)
rag_chatbot.py  →  parse_intent → ChromaDB → Ollama (mistral)
```

---

## 2. ChromaDB State

| Field | Value |
|---|---|
| Documents | 421,279 |
| Embedding model | all-MiniLM-L6-v2 (384-dim cosine) |
| Key metadata | satellite, country, continent, biome, date_start, date_end, date_start_int, date_end_int, duration_days, region, season |
| Country fill rate | ~96.8% |
| Date variants in code | E: 7–29d · A: 30–60d · B: 61–120d · C: 121–365d · D: 366–730d |

> **IMPORTANT:** ChromaDB was built before recent code fixes. Still missing:
> - France and Norway (ISO_A2_EH fix applied in code, not yet in DB)
> - Variant E (7–29d) — single-month queries return 0 exact hits
> - Normalised country names (e.g. "united states of america" vs "united states")
> - Region names like "Mediterranean", "Arctic", "Amazon" are NOT country values in DB
>   → country filter returns N/A for ocean/region queries; must use a real country name
> Full pipeline rebuild required.

---

## 3. Git Status

| Item | Status |
|---|---|
| Git repo initialized | ✅ `main` branch |
| Remote | `https://gitlab.com/HichriZeineb/rag_project.git` |
| Initial commit | ✅ `d929f19` — 15 files committed |
| Push to GitLab | ❌ Pending — needs GitLab token |
| rag_chatbot.py | ✅ Modified locally (not yet committed) |

**To push:**
```bash
# Create token at: gitlab.com → Settings → Access Tokens → write_repository scope
git add rag_chatbot.py SESSION_STATE.md
git commit -m "overhaul clarification engine and satellite detection"
git push https://HichriZeineb:<YOUR_TOKEN>@gitlab.com/HichriZeineb/rag_project.git main --force
```

---

## 4. File States

| File | State | Notes |
|---|---|---|
| `rag_chatbot.py` | ✅ Modified | Major overhaul this session — see section 7 |
| `generate_queries.py` | ✅ Stable | All 6 bugs fixed in previous session |
| `processor_standalone.py` | ✅ Stable | No changes |
| `embed_and_load.py` | ✅ Stable | No changes |
| `config.py` | ✅ Stable | No changes |
| `requirements.txt` | ✅ Stable | No changes |
| `chroma_db/` | ⚠️ Needs rebuild | See ChromaDB State above |
| `kb/Search_queries_world.json` | ✅ Present | 421,280 entries, excluded from git |
| `kb/outputs/documents_embedding.jsonl` | ✅ Present | 360 MB, excluded from git |
| `kb/geodata/` | ✅ Present | 5 GeoJSON files committed |

---

## 5. Chat Loop — state variables

| Variable | Purpose |
|---|---|
| `pending_clarification` | Stores original question while waiting for clarification reply |
| `pending_display` | Stores search results while waiting for "how many?" reply |
| `last_results` | Stores last completed search — used for refinements/pagination |
| `last_results["exact_offset"]` | Pagination offset for exact hits (new this session) |
| `last_results["partial_offset"]` | Pagination offset for partial hits (new this session) |
| `last_results["last_display_exact"]` | How many exact were last shown (for "show more") |
| `last_results["last_display_partial"]` | How many partial were last shown (for "show more") |

---

## 6. Clarification Engine — `_needs_clarification` rules

All rules evaluated every time — none short-circuits the others. All missing fields collected and returned as one combined message.

| Rule | Trigger | Question asked |
|---|---|---|
| R1 | Ambiguous city name (18 cities) | "Which X did you mean? A or B?" |
| R2 | No satellite detected + "sentinel" word present | "Which Sentinel? (1/2/3/5P)" |
| R2 | No satellite detected + no sentinel word | "Which satellite? (S1/S2/S3/S5P)" |
| R3 | No location detected | "Which region?" |
| R4 | No date detected | "Which time period?" |
| R5 | Multi-month range + no location | "Which region for this multi-month range?" |

**Location detection uses fuzzy matching** (difflib cutoff=0.82) — typos like "meditarranean", "franc", "gerrmany" are accepted.

**Ambiguous cities (18):** london, cambridge, richmond, victoria, adelaide, hamilton, birmingham, springfield, portland, memphis, kingston, georgetown, wellington, perth, newcastle, plymouth, albany, aurora

---

## 7. Satellite Detection — `parse_intent`

Three-pass detection (highest priority first):

### Pass 1 — Explicit codes (`SATELLITE_CODES`)
```
sentinel-1/2/3/5p  sentinel1/2/3/5p  sentinel 1/2/3/5p
sentinel-5  sentinel5  sentinel 5    (official mission name → S5P)
s1  s2  s3  s5p  s5
```

### Pass 2 — Mission synonyms (`SATELLITE_MISSION_SYNONYMS`)
Based on official Copernicus mission descriptions:

| Satellite | Keywords |
|---|---|
| S1 | radar, sar, synthetic aperture, backscatter, all-weather, flood/s, maritime, ships, sea ice, deformation, subsidence, landslide/s, disaster |
| S2 | optical, multispectral, high-resolution, high resolution, vegetation, agriculture, crops, soil, land cover, forest/s, wildfire, fire, inland waterways, water cover, emergency mapping, emergency |
| S3 | marine, oceanography, ocean colour/color, ocean monitoring, sea surface, sea surface temperature, sea level, sea-surface topography, altimetry, land surface temperature, land colour/color, vegetation index, climate |
| S5P | atmosphere/ic, atmospheric composition, air quality, pollution, trace gas/es, no2, nitrogen dioxide, so2, sulphur/sulfur dioxide, co, carbon monoxide, co2, carbon dioxide, methane, ch4, ozone, formaldehyde, hcho, aerosol/s, aerosol optical, optical depth, greenhouse, emission/s |

### Pass 3 — Fuzzy matching (typo tolerance)
- Checks each word and bigram in the query against all `SATELLITE_CODES` keys ≥ 6 chars
- cutoff = 0.82 (difflib)
- Guard: if matched code contains a digit but the candidate does not → skip (prevents "sentinel" alone from matching "sentinel-3")

---

## 8. `--ask` Mode — Interactive Clarification Loop

`--ask` mode now runs a full interactive loop:
1. Clarification loop — asks for missing satellite/region/date until all resolved
2. "How many?" — asks how many results to display
3. `_print_results` — prints actual results from Python
4. Ollama LLM — adds a short 1–3 sentence comment only

---

## 9. Refinement Engine — `_is_refinement` + pagination

**New: Refinement with new filter → re-search**
When user says "only germany" or "only spain" after a result, the code:
- Builds combined query: `original_question + " " + user_input`
- Parses intent of combined query
- If new country or satellite detected → triggers full re-search (not re-display)

**New: Pagination offset**
`_print_results` now accepts `exact_offset` and `partial_offset` parameters.
- "show more 5" / "give me next 5" → advances offset by last display count
- "show 10" (no "more") → resets offset to 0
- Offsets stored in `last_results` across turns

---

## 10. Key Functions

| Function | Returns |
|---|---|
| `retrieve_and_filter(q, col, embed, top_k)` | `(exact_hits, partial_hits, exact_count, partial_count, intent, clarification)` |
| `_print_results(..., exact_offset=0, partial_offset=0)` | prints to terminal — guaranteed accurate |
| `build_prompt(...)` | messages list — LLM comment only (1–3 sentences) |
| `_needs_clarification(intent, question)` | combined question string or `None` |
| `_is_refinement(text, last_results)` | `bool` — also triggers re-search if new filter detected |
| `_parse_display_request(text, top_k)` | `(display_exact, display_partial)` |
| `_suggest_refinement(intent, exact_count, partial_count)` | tip string or `None` |
| `_extract_country(text)` | normalised country name or `None` |
| `parse_intent(question)` | `{satellite, date_start, date_end, country}` |

---

## 11. All Bugs Fixed (20 total)

| ID | Bug | Status |
|---|---|---|
| BUG-1 | `_parse_dates` missed `MONTH to MONTH YEAR` form | ✅ |
| BUG-2 | Partial hits shown without exact/partial label | ✅ |
| BUG-3 | London → Canada geo collision | ✅ in code, rebuild pending |
| BUG-4 | Clarification never waited for reply | ✅ |
| BUG-5 | City ambiguity rule wrong priority | ✅ |
| BUG-6 | Cloud-cover clarification too aggressive | ✅ R3 removed |
| BUG-7 | Retrieval capped at n=20 | ✅ |
| BUG-8 | LLM hallucinated counts/results | ✅ `_print_results()` |
| BUG-9 | Vague queries returned noise | ✅ clarification rules |
| BUG-10 | `collection.count(where=...)` not supported | ✅ |
| BUG-11 | France/Norway absent from JSON | ✅ in code, rebuild pending |
| BUG-12 | Country name mismatches | ✅ `_NAME_NORMALIZE` + aliases |
| BUG-13 | "show me more" triggered new search | ✅ `_is_refinement` + `last_results` |
| BUG-14 | LLM ignored partial results | ✅ `_print_results()` |
| BUG-15 | `--ask` mode ignored clarification engine | ✅ interactive loop added |
| BUG-16 | Rule 0 swallowed Rule 1 (city ambiguity) | ✅ rules reordered, all rules evaluated |
| BUG-17 | Fully vague queries (no sat/date/location) returned all 421k docs | ✅ R2/R3/R4 combined |
| BUG-18 | Typos in satellite names not detected | ✅ fuzzy pass 3 in `parse_intent` |
| BUG-19 | "only germany" refinement re-displayed wrong results | ✅ re-search triggered on new filter |
| BUG-20 | "show more" showed same items again (no pagination) | ✅ offset tracking in `_print_results` |

---

## 12. Known Remaining Issues

| ID | Issue | Severity |
|---|---|---|
| ISSUE-1 | France, Norway, normalised names missing from live ChromaDB | High — rebuild needed |
| ISSUE-2 | Variant E (7–29d) not in ChromaDB | Medium — rebuild needed |
| ISSUE-3 | Legacy files not yet deleted | Low |
| ISSUE-4 | Region names (Mediterranean, Arctic, Amazon) not filterable in ChromaDB — country = N/A for ocean queries | Medium — data architecture limitation |

---

## 13. Files to delete (not yet done)

```bash
rm intent_parser.py retriever.py ingest.py generate_queries_v1_cities_only.py
rm "kb/Search queries.json"
rm kb/Search_queries_world_v1_cities_only.json
rm kb/records.json kb/index.faiss
rm kb/outputs/documents_full.json kb/outputs/documents_vectorstore.jsonl
rm SESSION_STATE_backup_20260409.md SESSION_STATE_backup_20260409b.md
rm SESSION_STATE_backup_20260409c.md SESSION_STATE_backup_20260409d.md
rm "command (5wfplj)"
```

---

## 14. Next Steps (in order)

| # | Task | Status |
|---|---|---|
| 1 | Test all fixes from this session (see test list) | ⏳ |
| 2 | Commit rag_chatbot.py + SESSION_STATE.md to git | ⏳ |
| 3 | Push to GitLab (token needed) | ⏳ |
| 4 | Delete legacy files | ⏳ |
| 5 | Pipeline rebuild (fixes ISSUE-1, ISSUE-2) | ⏳ |
| 6 | Test chatbot after rebuild | ⏳ |
| 7 | Add region-level filtering (Mediterranean, Arctic, etc.) | Pending |
| 8 | Add cloud_cover_max metadata + filter | Pending |
