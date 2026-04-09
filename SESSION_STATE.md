# SESSION STATE — Copernicus RAG Chatbot
_Last updated: 2026-04-09 (full session — README written, cleanup identified, pipeline rebuild pending)_
_(Previous state saved to SESSION_STATE_backup_20260409e.md)_

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
| Country fill rate | ~96.8% (natural features have no country by design) |
| Date variants in code | E: 7–29d · A: 30–60d · B: 61–120d · C: 121–365d · D: 366–730d |

> **IMPORTANT:** ChromaDB was built before recent fixes. It is missing:
> - France and Norway (ISO_A2_EH fix not yet applied)
> - Variant E (7–29d) — single-month queries return 0 exact hits
> - Normalised country names (e.g. "united states" vs "united states of america")
> A full pipeline rebuild is required to apply all fixes.

---

## 3. Chat Loop — state variables

| Variable | Purpose |
|---|---|
| `pending_clarification` | Stores original question while waiting for clarification reply |
| `pending_display` | Stores search results while waiting for "how many?" reply |
| `last_results` | Stores last completed search — used for refinements without re-searching |

**Flow:**
1. User asks → clarification check (R0–R2) → if needed: ask + wait
2. Run search → print `Found N exact + M partial`
3. Ask how many → wait for reply → parse with `_parse_display_request()`
4. Print results from Python directly (`_print_results()`) — guaranteed accurate
5. Call Ollama for 1–3 sentence comment only
6. Show `_suggest_refinement()` tip if applicable
7. Next input: if `_is_refinement()` → reuse `last_results`, else new search

---

## 4. Clarification Engine — `_needs_clarification` rules

| # | Trigger | Question asked |
|---|---|---|
| R0 | Satellite detected + no date + no location | "I can search S1/S2 data — when and where?" |
| R1 | Ambiguous city name AND no country detected | "Did you mean X or Y?" |
| R2 | Date span > 31 days AND no country AND no geo keyword | "Which region should I focus on?" |

R3 (cloud cover) removed — no cloud_cover field in metadata.

**Ambiguous cities (18):** london, cambridge, richmond, victoria, adelaide, hamilton, birmingham, springfield, portland, memphis, kingston, georgetown, wellington, perth, newcastle, plymouth, albany, aurora

---

## 5. Key functions and their signatures

| Function | Returns |
|---|---|
| `retrieve_and_filter(q, col, embed, top_k)` | `(exact_hits, partial_hits, exact_count, partial_count, intent, clarification)` |
| `_print_results(exact_hits, partial_hits, exact_count, partial_count, intent, display_exact, display_partial)` | prints to terminal directly |
| `build_prompt(..., display_exact, display_partial, clarification)` | `messages` list — LLM comment only (1–3 sentences) |
| `_needs_clarification(intent, question)` | string or `None` |
| `_is_refinement(text, last_results)` | `bool` — True = reuse last_results |
| `_parse_display_request(text, top_k)` | `(display_exact, display_partial)` |
| `_suggest_refinement(intent, exact_count, partial_count)` | tip string or `None` |
| `_extract_country(text)` | normalised country name or `None` (checks aliases first) |

`exact_count` / `partial_count` = real DB totals via `collection.get(where=..., include=[])`.
`exact_hits` / `partial_hits` = top-`top_k` by vector similarity — for display only.
`DEFAULT_DISPLAY = 5`

---

## 6. Country handling

**In `generate_queries.py`:**
- `build_country_maps()` reads `ne_110m_admin_0_countries.geojson`
- Fallback to `ISO_A2_EH` when `ISO_A2 = '-99'` (fixes France, Norway)
- `_NAME_NORMALIZE` dict normalises abbreviated Natural Earth names:
  - `"united states of america"` → `"united states"`
  - `"dem. rep. congo"` → `"democratic republic of the congo"`
  - `"bosnia and herz."` → `"bosnia and herzegovina"`
  - `"czechia"` → `"czech republic"` etc.

**In `rag_chatbot.py`:**
- `COUNTRY_NAMES` list: 160+ countries (full names, lowercase)
- `_COUNTRY_ALIASES` dict: usa, uk, drc, uae, bosnia, czechia, etc.
- `_extract_country()`: checks aliases first (word-boundary regex), then full names (longest-first)

---

## 7. Bugs Fixed (cumulative)

| ID | Bug | Status |
|---|---|---|
| BUG-1 | `_parse_dates` missed `MONTH to MONTH YEAR` form | ✅ Fixed |
| BUG-2 | Partial hits shown without exact/partial label | ✅ Fixed |
| BUG-3 | London → Canada geo collision in generate_queries.py | ✅ Fixed in code (rebuild pending) |
| BUG-4 | Clarification injected into LLM prompt, bot never waited | ✅ Fixed — pending_clarification loop |
| BUG-5 | City ambiguity rule swallowed by no-geo rule | ✅ Fixed — R0/R1 checked first |
| BUG-6 | Cloud-cover clarification fired on full-year queries | ✅ R3 removed entirely |
| BUG-7 | Retrieval capped at n=20 regardless of top_k | ✅ Fixed — uses top_k throughout |
| BUG-8 | LLM invented counts / hallucinated result lists | ✅ Fixed — results printed by Python, LLM comments only |
| BUG-9 | Vague queries returned noise without asking context | ✅ Fixed — Rule R0 |
| BUG-10 | `collection.count(where=...)` not supported in ChromaDB 1.5.5 | ✅ Fixed — uses `collection.get(where=..., include=[])` |
| BUG-11 | France/Norway absent from JSON (ISO_A2='-99' in Natural Earth) | ✅ Fixed in code (rebuild pending) |
| BUG-12 | Country name mismatches (usa ≠ united states of america) | ✅ Fixed — _NAME_NORMALIZE + _COUNTRY_ALIASES |
| BUG-13 | "show me more partial" triggered new search instead of reusing | ✅ Fixed — _is_refinement + last_results |
| BUG-14 | LLM ignored partial results when asked for both exact+partial | ✅ Fixed — _print_results() prints everything from Python |

---

## 8. Known Remaining Issues

| ID | Issue | Severity |
|---|---|---|
| ISSUE-1 | France, Norway, normalised names missing from live ChromaDB | High — rebuild needed |
| ISSUE-2 | Variant E (7–29d) not in ChromaDB — single-month queries → 0 exact | Medium — rebuild needed |
| ISSUE-3 | No cloud_cover field in metadata | Low — R3 removed, no blocker |
| ISSUE-4 | ~3.2% docs (oceans, rivers, lakes) have empty country — by design | Low |

---

## 9. Files to delete (identified, not yet deleted)

```
intent_parser.py                              # legacy, never called
retriever.py                                  # legacy, never called
ingest.py                                     # old ingestion script
generate_queries_v1_cities_only.py            # old version
kb/Search queries.json                        # old file (spaces in name)
kb/Search_queries_world_v1_cities_only.json   # old version
kb/records.json                               # old intermediate
kb/index.faiss                                # old FAISS index
kb/outputs/documents_full.json               # old intermediate
kb/outputs/documents_vectorstore.jsonl        # old format
SESSION_STATE_backup_20260409.md              # old backups
SESSION_STATE_backup_20260409b.md
SESSION_STATE_backup_20260409c.md
SESSION_STATE_backup_20260409d.md
command (5wfplj)                              # mystery file
```

---

## 10. Next Steps (prioritised)

| # | Task | Effort | Status |
|---|---|---|---|
| 1 | Delete identified legacy files | 1 min | ⏳ Ready |
| 2 | **Rebuild pipeline** (fixes ISSUE-1 and ISSUE-2) | ~60 min | ⏳ Ready |
| 3 | Test chatbot after rebuild (France, S1 feb 2025, refinement) | 5 min | Pending |
| 4 | Add cloud_cover_max metadata + filter | Medium | Pending |

### Rebuild command sequence
```bash
python generate_queries.py
python processor_standalone.py
python embed_and_load.py
```

### Test queries after rebuild
```
what optical queries exist for france          → should find results (was 0 before)
s1 data in february 2025 in italy             → should have exact hits (variant E)
show me s3 in germany spring 2025             → 8 exact and 6 partial (same response)
yes show me more partial                      → reuses last results (no new search)
```
