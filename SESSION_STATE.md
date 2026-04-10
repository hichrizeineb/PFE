# SESSION STATE — Copernicus RAG Chatbot
_Last updated: 2026-04-09 (git initialized, push pending)_
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
> Full pipeline rebuild required.

---

## 3. Git Status

| Item | Status |
|---|---|
| Git repo initialized | ✅ `main` branch |
| Remote | `https://gitlab.com/HichriZeineb/rag_project.git` |
| Initial commit | ✅ `d929f19` — 15 files committed |
| Push to GitLab | ❌ Pending — needs GitLab token |

**To finish the push:**
```bash
# Option A — Personal Access Token (recommended)
# 1. Go to: gitlab.com → Settings → Access Tokens → create with write_repository scope
# 2. Then run:
git push https://HichriZeineb:<YOUR_TOKEN>@gitlab.com/HichriZeineb/rag_project.git main

# Option B — SSH (if already configured)
git remote set-url origin git@gitlab.com:HichriZeineb/rag_project.git
git push -u origin main
```

**Files committed (15):**
- `.gitignore`, `README.md`, `SESSION_STATE.md`, `CLAUDE_CODE_PROMPT.md`
- `rag_chatbot.py`, `generate_queries.py`, `processor_standalone.py`, `embed_and_load.py`, `config.py`
- `requirements.txt`
- `kb/geodata/` (5 GeoJSON files)

**Files excluded by .gitignore (too large for git):**
- `chroma_db/` (5.4 GB)
- `kb/Search_queries_world.json` (233 MB)
- `kb/outputs/documents_embedding.jsonl` (360 MB)
- `kb/geodata/allCountries.txt`, `cities15000.txt` (1.7 GB)
- All SESSION_STATE backups, `__pycache__/`, `.venv/`

---

## 4. Chat Loop — state variables

| Variable | Purpose |
|---|---|
| `pending_clarification` | Stores original question while waiting for clarification reply |
| `pending_display` | Stores search results while waiting for "how many?" reply |
| `last_results` | Stores last completed search — used for refinements without re-searching |

---

## 5. Clarification Engine — `_needs_clarification` rules

| # | Trigger | Question asked |
|---|---|---|
| R0 | Satellite detected + no date + no location | "I can search S1/S2 data — when and where?" |
| R1 | Ambiguous city name AND no country detected | "Did you mean X or Y?" |
| R2 | Date span > 31 days AND no country AND no geo keyword | "Which region should I focus on?" |

R3 (cloud cover) removed — no cloud_cover field in metadata.

**Ambiguous cities (18):** london, cambridge, richmond, victoria, adelaide, hamilton, birmingham, springfield, portland, memphis, kingston, georgetown, wellington, perth, newcastle, plymouth, albany, aurora

---

## 6. Key functions

| Function | Returns |
|---|---|
| `retrieve_and_filter(q, col, embed, top_k)` | `(exact_hits, partial_hits, exact_count, partial_count, intent, clarification)` |
| `_print_results(...)` | prints to terminal directly — guaranteed accurate |
| `build_prompt(...)` | messages list — LLM comment only (1–3 sentences) |
| `_needs_clarification(intent, question)` | string or `None` |
| `_is_refinement(text, last_results)` | `bool` |
| `_parse_display_request(text, top_k)` | `(display_exact, display_partial)` |
| `_suggest_refinement(intent, exact_count, partial_count)` | tip string or `None` |
| `_extract_country(text)` | normalised country name or `None` |

---

## 7. All Bugs Fixed (14 total)

| ID | Bug | Status |
|---|---|---|
| BUG-1 | `_parse_dates` missed `MONTH to MONTH YEAR` form | ✅ |
| BUG-2 | Partial hits shown without exact/partial label | ✅ |
| BUG-3 | London → Canada geo collision | ✅ in code, rebuild pending |
| BUG-4 | Clarification never waited for reply | ✅ |
| BUG-5 | City ambiguity rule wrong priority | ✅ |
| BUG-6 | Cloud-cover clarification too aggressive | ✅ R3 removed |
| BUG-7 | Retrieval capped at n=20 | ✅ |
| BUG-8 | LLM hallucinated counts/results | ✅ _print_results() |
| BUG-9 | Vague queries returned noise | ✅ Rule R0 |
| BUG-10 | `collection.count(where=...)` not supported | ✅ |
| BUG-11 | France/Norway absent from JSON | ✅ in code, rebuild pending |
| BUG-12 | Country name mismatches | ✅ _NAME_NORMALIZE + aliases |
| BUG-13 | "show me more" triggered new search | ✅ _is_refinement + last_results |
| BUG-14 | LLM ignored partial results | ✅ _print_results() |

---

## 8. Known Remaining Issues

| ID | Issue | Severity |
|---|---|---|
| ISSUE-1 | France, Norway, normalised names missing from live ChromaDB | High — rebuild needed |
| ISSUE-2 | Variant E (7–29d) not in ChromaDB | Medium — rebuild needed |
| ISSUE-3 | Legacy files not yet deleted | Low |

---

## 9. Files to delete (not yet done)

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

## 10. Next Steps (in order)

| # | Task | Status |
|---|---|---|
| 1 | Finish GitLab push (token needed) | ⏳ |
| 2 | Delete legacy files | ⏳ |
| 3 | Pipeline rebuild | ⏳ |
| 4 | Test chatbot after rebuild | ⏳ |
| 5 | Add cloud_cover_max metadata + filter | Pending |
