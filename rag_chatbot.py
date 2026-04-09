"""
RAG Chatbot — Copernicus Satellite Data
========================================
Answers natural-language questions about Copernicus satellite search queries
stored in ChromaDB.

Architecture:
    parse_intent  →  vector search  →  hard metadata filter  →  Ollama LLM

Usage:
    python3 rag_chatbot.py                        # interactive chat
    python3 rag_chatbot.py --model mistral        # choose model
    python3 rag_chatbot.py --ask "S2 over France" # single question
    python3 rag_chatbot.py --list-models          # list Ollama models
    python3 rag_chatbot.py --verbose              # show filters + scores
    python3 rag_chatbot.py --top 100              # retrieval pool size

In-chat commands: /models  /model <name>  /verbose  /top <n>  /clear  /help  quit
"""

import argparse
import json
import re
import sys
from calendar import monthrange
from datetime import datetime
from typing import Optional

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("❌ Missing: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    sys.exit("❌ Missing: pip install chromadb")

try:
    import ollama
except ImportError:
    sys.exit("❌ Missing: pip install ollama")


# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "copernicus_rag"
DEFAULT_TOP_K   = 50
HISTORY_TURNS   = 6          # number of user/assistant pairs to keep
TEMPERATURE     = 0.1


# ============================================================================
# STEP A — Intent parser
# ============================================================================

# Explicit satellite codes — checked first (highest priority)
SATELLITE_CODES = {
    "sentinel-5p": "S5P", "sentinel5p": "S5P",
    "sentinel-1": "S1",   "sentinel1":  "S1",
    "sentinel-2": "S2",   "sentinel2":  "S2",
    "sentinel-3": "S3",   "sentinel3":  "S3",
    "s5p": "S5P",
    "s1":  "S1",
    "s2":  "S2",
    "s3":  "S3",
}

# Mission-type synonyms — checked only if no explicit code found
SATELLITE_MISSION_SYNONYMS = {
    "synthetic aperture": "S1", "sar": "S1", "radar": "S1",
    "multispectral": "S2", "optical": "S2",
    "atmospheric": "S5P", "air quality": "S5P", "no2": "S5P", "pollution": "S5P",
}

MONTH_MAP = {
    # English
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2,
    "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}


# Country name → normalised lowercase string stored in ChromaDB
# (matches the values written by build_country_maps in generate_queries.py)
COUNTRY_NAMES: list = [
    # A
    "afghanistan", "albania", "algeria", "angola", "antarctica", "argentina",
    "armenia", "australia", "austria", "azerbaijan",
    # B
    "bahamas", "bangladesh", "belarus", "belgium", "benin", "bhutan",
    "bolivia", "bosnia and herzegovina", "botswana", "brazil", "bulgaria",
    "burkina faso", "burundi",
    # C
    "cambodia", "cameroon", "canada", "central african republic", "chad",
    "chile", "china", "colombia", "costa rica", "croatia", "cuba",
    "czech republic", "cyprus",
    # D
    "democratic republic of the congo", "denmark", "djibouti",
    "dominican republic",
    # E
    "ecuador", "egypt", "el salvador", "equatorial guinea", "eritrea",
    "estonia", "ethiopia",
    # F
    "fiji", "finland", "france",
    # G
    "gabon", "gambia", "georgia", "germany", "ghana", "greece", "greenland",
    "guatemala", "guinea", "guinea-bissau", "guyana",
    # H
    "haiti", "honduras", "hungary",
    # I
    "iceland", "india", "indonesia", "iran", "iraq", "ireland",
    "israel", "ivory coast", "italy",
    # J
    "jamaica", "japan", "jordan",
    # K
    "kazakhstan", "kenya", "kuwait", "kyrgyzstan",
    # L
    "laos", "latvia", "lebanon", "lesotho", "liberia", "libya",
    "lithuania", "luxembourg",
    # M
    "madagascar", "malawi", "malaysia", "mali", "mauritania", "mexico",
    "moldova", "mongolia", "montenegro", "morocco", "mozambique", "myanmar",
    # N
    "namibia", "nepal", "netherlands", "new caledonia", "new zealand",
    "nicaragua", "niger", "nigeria", "north korea", "north macedonia",
    "norway",
    # O
    "oman",
    # P
    "pakistan", "panama", "papua new guinea", "paraguay", "peru",
    "philippines", "poland", "portugal",
    # Q
    "qatar",
    # R
    "romania", "russia", "rwanda",
    # S
    "saudi arabia", "senegal", "serbia", "sierra leone", "slovakia",
    "slovenia", "solomon islands", "somalia", "south africa", "south korea",
    "south sudan", "spain", "sri lanka", "sudan", "suriname", "sweden",
    "switzerland", "syria",
    # T
    "taiwan", "tajikistan", "tanzania", "thailand", "timor-leste",
    "togo", "trinidad and tobago", "tunisia", "turkey", "turkmenistan",
    # U
    "uganda", "ukraine", "united arab emirates", "united kingdom",
    "united states", "uruguay", "uzbekistan",
    # V
    "vanuatu", "venezuela", "vietnam",
    # W Y Z
    "western sahara", "yemen", "zambia", "zimbabwe",
]


# Common aliases → normalized country name (must match what ChromaDB stores)
_COUNTRY_ALIASES = {
    "usa":          "united states",
    "u.s.":         "united states",
    "u.s.a.":       "united states",
    "america":      "united states",
    "uk":           "united kingdom",
    "great britain":"united kingdom",
    "britain":      "united kingdom",
    "drc":          "democratic republic of the congo",
    "congo":        "democratic republic of the congo",
    "car":          "central african republic",
    "uae":          "united arab emirates",
    "czechia":      "czech republic",
    "ivory coast":  "ivory coast",
    "bosnia":       "bosnia and herzegovina",
    "herzegovina":  "bosnia and herzegovina",
}


def _extract_country(text: str) -> Optional[str]:
    """Return the normalised country name found in text, or None.

    Checks aliases first (short tokens like 'uk', 'usa'), then full names
    (longest-first so 'united kingdom' beats 'united').
    """
    t = text.lower()
    # Aliases — word-boundary check to avoid false matches (e.g. 'american')
    for alias, canonical in _COUNTRY_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', t):
            return canonical
    # Full names — longest first
    for name in sorted(COUNTRY_NAMES, key=len, reverse=True):
        if name in t:
            return name
    return None


def _last_day(year: int, month: int) -> int:
    return monthrange(year, month)[1]


def _parse_dates(text: str):
    """Extract date_start and date_end from natural language text.

    Handles:
      - "between March 2025 and November 2025"
      - "from 2025-03-01 to 2025-11-30"
      - "in March 2025" / "March 2025"
      - "summer 2024"
      - "2025"  → full year
    Returns (date_start, date_end) as "YYYY-MM-DD" strings, or (None, None).
    """
    t = text.lower()

    # ISO range: 2025-03-01 to 2025-11-30
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(?:to|and|-)\s+(\d{4}-\d{2}-\d{2})", t)
    if m:
        return m.group(1), m.group(2)

    # "between MONTH YEAR and MONTH YEAR"
    m = re.search(r"between\s+(\w+)\s+(\d{4})\s+and\s+(\w+)\s+(\d{4})", t)
    if m:
        m1, y1 = MONTH_MAP.get(m.group(1)), int(m.group(2))
        m2, y2 = MONTH_MAP.get(m.group(3)), int(m.group(4))
        if m1 and m2:
            return f"{y1}-{m1:02d}-01", f"{y2}-{m2:02d}-{_last_day(y2, m2):02d}"

    # "between MONTH and MONTH YEAR"  (year shared at end)
    m = re.search(r"between\s+(\w+)\s+and\s+(\w+)\s+(\d{4})", t)
    if m:
        m1, m2, y = MONTH_MAP.get(m.group(1)), MONTH_MAP.get(m.group(2)), int(m.group(3))
        if m1 and m2:
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{_last_day(y, m2):02d}"

    # "from MONTH YEAR to MONTH YEAR"
    m = re.search(
        r"from\s+(\w+)\s+(\d{4})\s+(?:to|until|through)\s+(\w+)\s+(\d{4})", t
    )
    if m:
        m1, y1 = MONTH_MAP.get(m.group(1)), int(m.group(2))
        m2, y2 = MONTH_MAP.get(m.group(3)), int(m.group(4))
        if m1 and m2:
            return f"{y1}-{m1:02d}-01", f"{y2}-{m2:02d}-{_last_day(y2, m2):02d}"

    # "from/since MONTH to/until MONTH YEAR"  (year shared at end — the primary bug case)
    m = re.search(
        r"(?:from|since)\s+(\w+)\s+(?:to|until|through)\s+(\w+)\s+(\d{4})", t
    )
    if m:
        m1, m2, y = MONTH_MAP.get(m.group(1)), MONTH_MAP.get(m.group(2)), int(m.group(3))
        if m1 and m2:
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{_last_day(y, m2):02d}"

    # "MONTH to MONTH YEAR"  (e.g. "february to april 2025")
    m = re.search(r"(\w+)\s+(?:to|until|through)\s+(\w+)\s+(\d{4})", t)
    if m:
        m1, m2, y = MONTH_MAP.get(m.group(1)), MONTH_MAP.get(m.group(2)), int(m.group(3))
        if m1 and m2:
            return f"{y}-{m1:02d}-01", f"{y}-{m2:02d}-{_last_day(y, m2):02d}"

    # French: "entre janvier et juin 2024" / "entre mars 2025 et novembre 2025"
    # Must come before the generic "MONTH YEAR" catch-all below.
    m = re.search(
        r"entre\s+(\w+)(?:\s+(\d{4}))?\s+et\s+(\w+)(?:\s+(\d{4}))?", t
    )
    if m:
        m1_name, y1_raw, m2_name, y2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        m1 = MONTH_MAP.get(m1_name)
        m2 = MONTH_MAP.get(m2_name)
        if m1 and m2:
            yr_fallback_match = re.search(r"\b(20\d{2})\b", t)
            yr_fallback = int(yr_fallback_match.group(1)) if yr_fallback_match else None
            y1 = int(y1_raw) if y1_raw else yr_fallback
            y2 = int(y2_raw) if y2_raw else yr_fallback
            if y1 and y2:
                return f"{y1}-{m1:02d}-01", f"{y2}-{m2:02d}-{_last_day(y2, m2):02d}"

    # French: "de mars 2025 à novembre 2025"
    m = re.search(
        r"\bde\s+(\w+)\s+(\d{4})\s+[àa]\s+(\w+)\s+(\d{4})", t
    )
    if m:
        m1, y1 = MONTH_MAP.get(m.group(1)), int(m.group(2))
        m2, y2 = MONTH_MAP.get(m.group(3)), int(m.group(4))
        if m1 and m2:
            return f"{y1}-{m1:02d}-01", f"{y2}-{m2:02d}-{_last_day(y2, m2):02d}"

    # "in MONTH YEAR" or "MONTH YEAR"
    m = re.search(r"\b(\w+)\s+(\d{4})\b", t)
    if m:
        mn = MONTH_MAP.get(m.group(1))
        y  = int(m.group(2))
        if mn:
            return f"{y}-{mn:02d}-01", f"{y}-{mn:02d}-{_last_day(y, mn):02d}"

    # Season + year
    m = re.search(r"\b(spring|summer|autumn|fall|winter)\s+(\d{4})\b", t)
    if m:
        season, year = m.group(1), int(m.group(2))
        ranges = {
            "spring": ("03-01", "05-31"), "summer": ("06-01", "08-31"),
            "autumn": ("09-01", "11-30"), "fall":   ("09-01", "11-30"),
            "winter": ("12-01", "02-28"),
        }
        s, e = ranges[season]
        if season == "winter":
            return f"{year}-{s}", f"{year + 1}-{e}"
        return f"{year}-{s}", f"{year}-{e}"

    # Bare year
    m = re.search(r"\b(20\d{2})\b", t)
    if m:
        y = m.group(1)
        return f"{y}-01-01", f"{y}-12-31"

    return None, None


def parse_intent(question: str) -> dict:
    """Extract hard-filter values from a natural-language question.

    Only extracts fields that drive ChromaDB where clauses:
      - satellite code  → $eq filter (LLM cannot know private DB field values)
      - date_start/end  → $gte/$lte filters (regex is faster and deterministic)
      - country         → $eq filter (exact lowercase match against metadata)

    Geography, biome, and free-text meaning are left to the LLM — it handles
    those natively from the raw question without any Python synonym mapping.

    Returns:
        {
          "satellite":  "S1"|"S2"|"S3"|"S5P"|None,
          "date_start": "YYYY-MM-DD"|None,
          "date_end":   "YYYY-MM-DD"|None,
          "country":    "<name>"|None,
        }
    """
    q = question.lower()

    # Satellite — two passes: explicit codes first, mission synonyms as fallback.
    # This ensures "S2 radar imaging" → S2 (not S1 via "radar").
    satellite = None
    for kw in sorted(SATELLITE_CODES, key=len, reverse=True):
        if kw in q:
            satellite = SATELLITE_CODES[kw]
            break
    if satellite is None:
        for kw in sorted(SATELLITE_MISSION_SYNONYMS, key=len, reverse=True):
            if kw in q:
                satellite = SATELLITE_MISSION_SYNONYMS[kw]
                break

    date_start, date_end = _parse_dates(question)
    country = _extract_country(question)

    return {
        "satellite":  satellite,
        "date_start": date_start,
        "date_end":   date_end,
        "country":    country,
    }


# ============================================================================
# STEP B — Retrieve and filter
# ============================================================================

def _to_int_date(iso: str) -> int:
    """Convert 'YYYY-MM-DD' to integer YYYYMMDD for ChromaDB $gte/$lte."""
    return int(iso.replace("-", ""))


def _sat_clause(intent: dict) -> list:
    """Return a list with the satellite $eq clause, or empty list."""
    if intent["satellite"]:
        return [{"satellite": {"$eq": intent["satellite"]}}]
    return []


def _hard_clauses(intent: dict) -> list:
    """Return satellite + country $eq clauses (both optional)."""
    clauses = _sat_clause(intent)
    if intent.get("country"):
        clauses.append({"country": {"$eq": intent["country"]}})
    return clauses


def build_where_clause(intent: dict) -> Optional[dict]:
    """Containment filter: doc dates must fall entirely within the requested window.

    date_start_int >= req_start_int  AND  date_end_int <= req_end_int
    Only docs completely inside the window are returned.
    Uses integer YYYYMMDD fields because ChromaDB $gte/$lte requires int/float.
    """
    clauses = _hard_clauses(intent)

    if intent["date_start"] and intent["date_end"]:
        clauses.append({"date_start_int": {"$gte": _to_int_date(intent["date_start"])}})
        clauses.append({"date_end_int":   {"$lte": _to_int_date(intent["date_end"])}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _overlap_where_clause(intent: dict) -> Optional[dict]:
    """Overlap filter: doc dates overlap the requested window.

    date_start_int <= req_end_int  AND  date_end_int >= req_start_int
    Broader than containment — includes docs that start before or end after.
    """
    clauses = _hard_clauses(intent)

    if intent["date_start"] and intent["date_end"]:
        clauses.append({"date_start_int": {"$lte": _to_int_date(intent["date_end"])}})
        clauses.append({"date_end_int":   {"$gte": _to_int_date(intent["date_start"])}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _needs_clarification(intent: dict, question: str) -> Optional[str]:
    """Return a focused follow-up question when the intent is ambiguous.

    Rules (checked in priority order):
    1. Ambiguous city name (e.g. London UK vs Canada) — must be first so it
       isn't swallowed by the no-geo rule below.
    2. Multi-month range with no geo → ask for location scope.
    3. Optical (S2) + specific date window ≤180 days + no cloud preference.

    Returns a follow-up string, or None if no clarification is needed.
    """
    q_lower = question.lower()
    ds, de  = intent.get("date_start"), intent.get("date_end")
    country = intent.get("country")
    sat     = intent.get("satellite")

    # Rule 0 — too vague: satellite or generic keyword but no date AND no location
    _GEO_ANCHOR = {
        "italy", "france", "germany", "spain", "uk", "united kingdom", "europe",
        "amazon", "africa", "asia", "mediterranean", "sahara", "arctic",
        "worldwide", "global", "all", "everywhere", "any", "forest", "ocean",
        "desert", "mountain", "urban", "coast",
    }
    if sat and not ds and not country and not any(w in q_lower for w in _GEO_ANCHOR):
        return (
            f"I can search {sat} data — when and where? "
            "(e.g. 'France in March 2025', 'Amazon basin 2024', or 'all regions 2025')"
        )

    # Rule 1 — ambiguous city names (checked FIRST)
    AMBIGUOUS = {
        "london":      "Did you mean London, UK or London, Ontario (Canada)?",
        "cambridge":   "Did you mean Cambridge, UK or Cambridge, Massachusetts (USA)?",
        "richmond":    "Did you mean Richmond, UK or Richmond, Virginia (USA)?",
        "victoria":    "Did you mean Victoria, BC (Canada), Victoria (Australia), or Lake Victoria (Africa)?",
        "adelaide":    "Did you mean Adelaide, Australia or Adelaide, South Africa?",
        "hamilton":    "Did you mean Hamilton, New Zealand or Hamilton, Ontario (Canada)?",
        "birmingham":  "Did you mean Birmingham, UK or Birmingham, Alabama (USA)?",
        "springfield": "Did you mean Springfield, Illinois, Springfield, Missouri, or another US Springfield?",
        "portland":    "Did you mean Portland, Oregon or Portland, Maine (USA)?",
        "memphis":     "Did you mean Memphis, Tennessee (USA) or Memphis, Egypt (ancient site)?",
        "kingston":    "Did you mean Kingston, Jamaica or Kingston, Ontario (Canada)?",
        "georgetown":  "Did you mean Georgetown, Guyana or Georgetown, Washington D.C. (USA)?",
        "wellington":  "Did you mean Wellington, New Zealand or Wellington, South Africa?",
        "perth":       "Did you mean Perth, Australia or Perth, Scotland (UK)?",
        "newcastle":   "Did you mean Newcastle, UK or Newcastle, New South Wales (Australia)?",
        "plymouth":    "Did you mean Plymouth, UK or Plymouth, Massachusetts (USA)?",
        "albany":      "Did you mean Albany, New York (USA) or Albany, Western Australia?",
        "aurora":      "Did you mean Aurora, Colorado (USA) or Aurora, Ontario (Canada)?",
    }
    if not country:
        for city, question_text in AMBIGUOUS.items():
            if city in q_lower:
                return question_text

    # Rule 2 — multi-month range with no geo
    if ds and de and not country:
        from datetime import date as _date
        try:
            span_days = (_date.fromisoformat(de) - _date.fromisoformat(ds)).days
        except ValueError:
            span_days = 0
        _GEO_WORDS = {"italy", "france", "germany", "spain", "europe", "amazon",
                      "africa", "asia", "mediterranean", "sahara", "worldwide",
                      "global", "all", "everywhere", "any"}
        if span_days > 31 and not any(w in q_lower for w in _GEO_WORDS):
            return (
                "Your query covers multiple months but no region was mentioned. "
                "Which country or region should I focus on? "
                "(e.g. Italy, Amazon, Mediterranean — or just press Enter for worldwide)"
            )

    return None


def retrieve_and_filter(question, collection, embed_model, top_k=DEFAULT_TOP_K):
    """
    Returns: (exact_hits, partial_hits, exact_count, partial_count, intent, clarification)

      exact_hits    — top-K docs by vector similarity whose dates fall entirely
                      within the requested window (for display — capped at top_k).
      partial_hits  — top-K docs that overlap but extend beyond the window.
      exact_count   — real total count of exact matches in ChromaDB (all docs, not just top-K).
      partial_count — real total count of partial-only matches (overlap minus exact).
      intent        — dict from parse_intent()
      clarification — a follow-up question string, or None
    """
    intent    = parse_intent(question)
    has_dates = bool(intent["date_start"] and intent["date_end"])
    query_vec = embed_model.encode([question]).tolist()

    def _count(where_clause):
        # chromadb 1.5.x does not support count(where=...).
        # Use get() with include=[] to fetch only IDs — fast enough for metadata-only queries.
        try:
            if where_clause:
                return len(collection.get(where=where_clause, include=[])["ids"])
            return collection.count()
        except Exception:
            return 0

    def _query(where_clause, n):
        kwargs = dict(
            query_embeddings=query_vec,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            kwargs["where"] = where_clause
        return collection.query(**kwargs)

    def _to_hits(results):
        return [
            {"text": t, "metadata": m, "distance": d}
            for t, m, d in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # --- Pass 1: containment (exact — doc fully inside user window) ---
    containment_where = build_where_clause(intent)
    exact_count = _count(containment_where)
    exact_hits  = []
    try:
        results    = _query(containment_where, n=top_k)
        exact_hits = _to_hits(results)
    except Exception:
        exact_hits = []

    # --- Pass 2: overlap — always run when dates exist ---
    partial_count = 0
    partial_hits  = []
    if has_dates:
        overlap_where = _overlap_where_clause(intent)
        overlap_count = _count(overlap_where)
        partial_count = max(0, overlap_count - exact_count)
        try:
            results     = _query(overlap_where, n=top_k)
            all_overlap = _to_hits(results)
            exact_ids   = {h["text"] for h in exact_hits}
            partial_hits = [h for h in all_overlap if h["text"] not in exact_ids]
        except Exception:
            partial_hits = []

    clarification = _needs_clarification(intent, question)

    return exact_hits, partial_hits, exact_count, partial_count, intent, clarification


# ============================================================================
# STEP C — Build prompt
# ============================================================================

def _format_doc(meta: dict) -> str:
    return (
        f"• Name:      {meta.get('original_name', meta.get('region', 'N/A'))}\n"
        f"  Satellite: {meta.get('satellite', 'N/A')} ({meta.get('mission_type', 'N/A')})\n"
        f"  Region:    {meta.get('region', meta.get('region_name', 'N/A'))}\n"
        f"  Country:   {meta.get('country') or 'N/A'}\n"
        f"  Continent: {meta.get('continent') or 'N/A'}\n"
        f"  Biome:     {meta.get('biome') or 'N/A'}\n"
        f"  Dates:     {meta.get('date_start', '?')} → {meta.get('date_end', '?')} "
        f"({meta.get('duration_days', '?')} days)\n"
        f"  Season:    {meta.get('season', 'N/A')}\n"
        f"  Dataset:   {meta.get('dataset', 'N/A')}\n"
    )


def _format_doc_partial(meta: dict, req_start: str, req_end: str) -> str:
    """Like _format_doc but shows only date overflow (no season — it reflects doc start, not search)."""
    doc_start = meta.get("date_start", "")
    doc_end   = meta.get("date_end",   "")
    overflow = []
    if doc_start and doc_start < req_start:
        overflow.append(f"starts {doc_start} (before window)")
    if doc_end and doc_end > req_end:
        overflow.append(f"ends {doc_end} (after window)")
    overflow_str = " | ".join(overflow) if overflow else "extends beyond window"
    return (
        f"• Name:      {meta.get('original_name', meta.get('region', 'N/A'))}\n"
        f"  Satellite: {meta.get('satellite', 'N/A')} ({meta.get('mission_type', 'N/A')})\n"
        f"  Region:    {meta.get('region', meta.get('region_name', 'N/A'))}\n"
        f"  Country:   {meta.get('country') or 'N/A'}\n"
        f"  Continent: {meta.get('continent') or 'N/A'}\n"
        f"  Biome:     {meta.get('biome') or 'N/A'}\n"
        f"  Dates:     {doc_start} → {doc_end} ({meta.get('duration_days', '?')} days)\n"
        f"  Overflow:  {overflow_str}\n"
        f"  Dataset:   {meta.get('dataset', 'N/A')}\n"
    )


DEFAULT_DISPLAY = 5   # used only in --ask single-question mode


def _suggest_refinement(intent: dict, exact_count: int, partial_count: int) -> Optional[str]:
    """Return a short actionable tip after showing results, or None."""
    tips = []
    if not intent.get("country"):
        tips.append("add a country (e.g. 'only france', 'just germany')")
    if not intent.get("satellite"):
        tips.append("specify a satellite (S1 radar / S2 optical / S3 ocean / S5P atmosphere)")
    if intent.get("date_start") and intent.get("date_end") and partial_count > exact_count * 2:
        tips.append("tighten the date range for more exact matches")
    if not tips:
        return None
    return "💡 To refine: " + " — or ".join(tips) + ".\n   You can also say 'show more' or 'show 10 more' to see more results."


# Phrases that always mean "show more from the same results"
_MORE_PHRASES = (
    "show more", "give more", "more results", "more partial", "more exact",
    "show partial", "show exact", "give partial", "give exact",
    "show me more", "give me more", "show me partial", "show me exact",
    "give me partial", "give me exact",
)

# Explicit new-topic indicators — only treated as new query when no "more" phrase present
_NEW_QUERY_STARTS = (
    "what ", "find ", "search ", "list ", "how many", "are there", "do you have",
)

# Explicit refinement prefixes
_REFINEMENT_STARTS = (
    "only ", "just ", "narrow ", "add ", "filter ", "with ",
    "exclude ", "remove ", "without ", "yes ",
)


def _is_refinement(text: str, last_results: dict) -> bool:
    """Return True if text is a follow-up on the previous results (no new search needed)."""
    if not last_results:
        return False
    t = text.lower().strip()

    # "show me more partial" / "give me the exact" etc. → always reuse
    if any(phrase in t for phrase in _MORE_PHRASES):
        return True

    # Explicit new topic → new search
    if any(t.startswith(p) for p in _NEW_QUERY_STARTS):
        return False

    # Explicit refinement prefix → reuse
    if any(t.startswith(p) for p in _REFINEMENT_STARTS):
        return True

    # Short input (≤ 4 words, no ?) with no new location/satellite/date → reuse
    if len(t.split()) <= 4 and "?" not in t:
        return True

    return False


def _parse_display_request(text: str, top_k: int) -> tuple:
    """Parse a free-text 'how many' reply into (display_exact, display_partial).

    Handles:
      "10"                           → (10, 10)
      "give me 10"                   → (10, 10)
      "show all"                     → (top_k, top_k)
      "10 exact"                     → (10, DEFAULT_DISPLAY)
      "5 partial"                    → (DEFAULT_DISPLAY, 5)
      "10 exact and 6 partial"       → (10, 6)
      "give me 10 exact 6 partial"   → (10, 6)
    """
    t = text.lower()

    if any(w in t for w in ("all", "everything", "tous", "tout")):
        return top_k, top_k

    # Look for paired exact+partial: "10 exact ... 6 partial" or "6 partial ... 10 exact"
    m_e = re.search(r'(\d+)\s*exact', t)
    m_p = re.search(r'(\d+)\s*partial', t)
    if m_e and m_p:
        return (max(1, min(int(m_e.group(1)), top_k)),
                max(1, min(int(m_p.group(1)), top_k)))
    if m_e:
        return max(1, min(int(m_e.group(1)), top_k)), DEFAULT_DISPLAY
    if m_p:
        return DEFAULT_DISPLAY, max(1, min(int(m_p.group(1)), top_k))

    # Single number anywhere in text
    m = re.search(r'\b(\d+)\b', t)
    if m:
        n = max(1, min(int(m.group(1)), top_k))
        return n, n

    return DEFAULT_DISPLAY, DEFAULT_DISPLAY


def _print_results(exact_hits, partial_hits, exact_count, partial_count,
                   intent, display_exact, display_partial):
    """Print results directly to terminal from Python — guaranteed accurate output."""
    req_start = intent.get("date_start", "")
    req_end   = intent.get("date_end",   "")
    has_dates = bool(req_start and req_end)

    if has_dates:
        shown_exact   = exact_hits[:display_exact]
        shown_partial = partial_hits[:display_partial]

        print(f"\n{'─'*56}")
        print(f" EXACT MATCHES — {exact_count:,} total | showing {len(shown_exact)}")
        print(f" Fully within {req_start} → {req_end}")
        print(f"{'─'*56}")
        if shown_exact:
            for i, h in enumerate(shown_exact, 1):
                m = h["metadata"]
                print(f"\n{i}. {m.get('original_name', m.get('region','N/A'))}")
                print(f"   Satellite : {m.get('satellite','N/A')} ({m.get('mission_type','N/A')})")
                print(f"   Region    : {m.get('region','N/A')}  |  Country: {m.get('country') or 'N/A'}")
                print(f"   Dates     : {m.get('date_start','?')} → {m.get('date_end','?')} ({m.get('duration_days','?')} days)")
                print(f"   Biome     : {m.get('biome') or 'N/A'}  |  Season: {m.get('season','N/A')}")
                print(f"   Dataset   : {m.get('dataset','N/A')}")
        else:
            print("   (none — window too narrow for any document to fit entirely inside)")

        if partial_count > 0:
            print(f"\n{'─'*56}")
            print(f" PARTIAL MATCHES — {partial_count:,} total | showing {len(shown_partial)}")
            print(f" Overlap {req_start} → {req_end} but extend beyond (time only)")
            print(f"{'─'*56}")
            for i, h in enumerate(shown_partial, 1):
                m = h["metadata"]
                doc_start = m.get("date_start", "")
                doc_end   = m.get("date_end",   "")
                overflow = []
                if doc_start and doc_start < req_start:
                    overflow.append(f"starts {doc_start}")
                if doc_end and doc_end > req_end:
                    overflow.append(f"ends {doc_end}")
                overflow_str = " | ".join(overflow) if overflow else "extends beyond"
                print(f"\n{i}. {m.get('original_name', m.get('region','N/A'))}")
                print(f"   Satellite : {m.get('satellite','N/A')} ({m.get('mission_type','N/A')})")
                print(f"   Region    : {m.get('region','N/A')}  |  Country: {m.get('country') or 'N/A'}")
                print(f"   Dates     : {doc_start} → {doc_end} ({m.get('duration_days','?')} days)")
                print(f"   Overflow  : {overflow_str}")
                print(f"   Biome     : {m.get('biome') or 'N/A'}")
                print(f"   Dataset   : {m.get('dataset','N/A')}")
        print(f"{'─'*56}\n")

    else:
        shown = exact_hits[:display_exact]
        print(f"\n{'─'*56}")
        print(f" RESULTS — {exact_count:,} total | showing {len(shown)} most relevant")
        print(f"{'─'*56}")
        if shown:
            for i, h in enumerate(shown, 1):
                m = h["metadata"]
                print(f"\n{i}. {m.get('original_name', m.get('region','N/A'))}")
                print(f"   Satellite : {m.get('satellite','N/A')} ({m.get('mission_type','N/A')})")
                print(f"   Region    : {m.get('region','N/A')}  |  Country: {m.get('country') or 'N/A'}")
                print(f"   Dates     : {m.get('date_start','?')} → {m.get('date_end','?')} ({m.get('duration_days','?')} days)")
                print(f"   Biome     : {m.get('biome') or 'N/A'}  |  Season: {m.get('season','N/A')}")
                print(f"   Dataset   : {m.get('dataset','N/A')}")
        else:
            print("   (no documents found)")
        print(f"{'─'*56}\n")


def build_prompt(question, exact_hits, partial_hits, exact_count, partial_count,
                 intent, history,
                 display_exact=DEFAULT_DISPLAY, display_partial=DEFAULT_DISPLAY,
                 clarification=None):
    system = """\
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

[RULES]
The results have already been printed to the user directly by the system.
Your job is ONLY to write 1–3 sentences:
  - Confirm what was found (satellite, location, period)
  - If exact count = 0: explain why in one sentence (window too narrow)
  - If partial count is high: one sentence explaining what partial means here (time overflow only, location is exact)
Do NOT re-list the documents. Do NOT invent data. Do NOT add recommendations unless asked.
"""

    filter_lines = []
    if intent["satellite"]:
        filter_lines.append(f"  Satellite  : {intent['satellite']}")
    if intent.get("country"):
        filter_lines.append(f"  Country    : {intent['country']}")
    if intent["date_start"] or intent["date_end"]:
        filter_lines.append(
            f"  Date range : {intent.get('date_start','(open)')} → {intent.get('date_end','(open)')}"
        )
    filters_block = ("Filters: " + " | ".join(filter_lines)) if filter_lines else "No filters"

    user_content = (
        f"[QUESTION]\n{question}\n\n"
        f"{filters_block}\n\n"
        f"[COUNTS] {exact_count:,} exact + {partial_count:,} partial"
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


# ============================================================================
# STEP D — Interactive chat loop
# ============================================================================

def print_collection_summary(collection) -> None:
    """Print a breakdown of what is in ChromaDB so the user knows what to ask."""
    total = collection.count()
    if total == 0:
        print("   (collection is empty)")
        return

    # Sample up to 2000 docs for stats — avoids fetching all docs at once
    # (chromadb 1.x can fail on unbounded .get() with large collections)
    SAMPLE = min(total, 2000)
    all_meta = collection.get(limit=SAMPLE, include=["metadatas"])["metadatas"]

    by_sat       = {}
    by_continent = {}
    by_biome     = {}
    dates        = []

    for m in all_meta:
        sat = m.get("satellite", "?")
        by_sat[sat] = by_sat.get(sat, 0) + 1

        c = m.get("continent") or ""
        if c:
            by_continent[c] = by_continent.get(c, 0) + 1

        b = m.get("biome") or ""
        if b:
            by_biome[b] = by_biome.get(b, 0) + 1

        ds = m.get("date_start", "")
        de = m.get("date_end",   "")
        if ds:
            dates.append(ds)
        if de:
            dates.append(de)

    date_range = f"{min(dates)} → {max(dates)}" if dates else "unknown"

    sat_str   = "  ".join(f"{k}:{v}" for k, v in sorted(by_sat.items()))
    cont_str  = "  ".join(f"{k}:{v}" for k, v in sorted(by_continent.items()))
    biome_str = "  ".join(f"{k}:{v}" for k, v in sorted(by_biome.items()))

    print(f"\n📊 Collection summary ({total} docs, stats from {SAMPLE}-doc sample)")
    print(f"   Satellites : {sat_str}")
    print(f"   Date range : {date_range}")
    print(f"   Continents : {cont_str or '(none tagged)'}")
    print(f"   Biomes     : {biome_str or '(none tagged)'}")


def list_models() -> list:
    try:
        response = ollama.list()
        if hasattr(response, "models"):
            return [m.model for m in response.models]
        return [m["name"] for m in response.get("models", [])]
    except Exception as e:
        print(f"⚠️  Could not contact Ollama: {e}")
        return []


def pick_default_model(models: list) -> Optional[str]:
    preferred = ["mistral", "llama3", "llama2", "gemma"]
    for p in preferred:
        for m in models:
            if p in m.lower():
                return m
    return models[0] if models else None


def print_help():
    print("""
Commands:
  /models          — list available Ollama models
  /model <name>    — switch to a different model
  /verbose         — toggle verbose mode (filters + scores)
  /top <n>         — set retrieval pool size (default 50)
  /clear           — clear conversation history
  /help            — show this help
  quit / exit      — quit
""")


def run_chat(args):
    # Load embedding model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL} ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ Ready")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"❌ Collection '{COLLECTION_NAME}' not found.\n"
            "   Run: python3 processor_standalone.py && python3 embed_and_load.py"
        )
    try:
        print_collection_summary(collection)
    except Exception as e:
        print(f"⚠️  Could not print collection summary: {e}")

    # Ollama models
    models = list_models()
    if not models:
        sys.exit("❌ No Ollama models found. Run: ollama pull mistral")

    print(f"\n🦙 Available Ollama models:")
    for m in models:
        print(f"   • {m}")

    if args.list_models:
        return

    current_model = args.model or pick_default_model(models)
    if current_model not in models:
        print(f"⚠️  Model '{current_model}' not found. Using '{models[0]}' instead.")
        current_model = models[0]

    print(f"\n✅ Using model: {current_model}")

    verbose = args.verbose
    top_k   = args.top

    # Single-question mode
    if args.ask:
        exact_hits, partial_hits, exact_count, partial_count, intent, clarification = \
            retrieve_and_filter(args.ask, collection, embed_model, top_k)
        if verbose:
            print(f"\n🔍 Intent: {intent}")
            print(f"   Exact: {exact_count:,} total ({len(exact_hits)} retrieved) | "
                  f"Partial: {partial_count:,} total ({len(partial_hits)} retrieved)")
            if clarification:
                print(f"   Clarification: {clarification}")
            print()
        print(f"\nFound {exact_count:,} exact + {partial_count:,} partial matches.")
        messages = build_prompt(args.ask, exact_hits, partial_hits, exact_count, partial_count,
                                intent, [],
                                display_exact=DEFAULT_DISPLAY, display_partial=DEFAULT_DISPLAY,
                                clarification=clarification)
        response = ollama.chat(
            model=current_model,
            messages=messages,
            options={"temperature": TEMPERATURE},
        )
        print(response.message.content)
        return

    # Interactive loop
    print(f"\n{'='*60}")
    print("  Copernicus RAG Chatbot")
    print(f"  Model: {current_model} | Docs: {collection.count()}")
    print("  Type /help for commands, 'quit' to exit")
    print(f"{'='*60}")
    print("  Ask about satellite data, e.g.:")
    print("    'show me S2 optical queries over Europe in 2024'")
    print("    'what radar data exists for forests?'")
    print(f"{'='*60}\n")

    history = []
    pending_clarification = None   # stores original question while waiting for clarification reply
    pending_display       = None   # stores search results while waiting for "how many?" reply
    last_results          = None   # stores last completed search for refinements / "show more"

    while True:
        try:
            user_input = input(f"You [{current_model}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/clear":
            history.clear()
            pending_clarification = None
            pending_display       = None
            last_results          = None
            print("🗑️  History cleared.")
            continue

        if user_input == "/verbose":
            verbose = not verbose
            print(f"🔎 Verbose: {'ON' if verbose else 'OFF'}")
            continue

        if user_input == "/models":
            models = list_models()
            print("Available models:")
            for m in models:
                marker = " ← active" if m == current_model else ""
                print(f"  • {m}{marker}")
            continue

        if user_input.startswith("/model "):
            requested = user_input[7:].strip()
            models = list_models()
            if requested in models:
                current_model = requested
                print(f"✅ Switched to: {current_model}")
            else:
                print(f"❌ Model '{requested}' not available. "
                      f"Run: ollama pull {requested}")
            continue

        if user_input.startswith("/top "):
            try:
                top_k = int(user_input[5:].strip())
                print(f"✅ Pool size set to {top_k}")
            except ValueError:
                print("Usage: /top <number>")
            continue

        # ── State: waiting for "how many to show?" reply ──────────────────────
        if pending_display:
            pd = pending_display
            pending_display = None

            display_exact, display_partial = _parse_display_request(user_input, top_k)

            if verbose:
                print(f"\n🔍 Intent: {pd['intent']}")
                print(f"   Exact: {pd['exact_count']:,} | Partial: {pd['partial_count']:,}")
                print(f"   Showing: {display_exact} exact, {display_partial} partial")

            # Print results from Python — guaranteed, LLM cannot skip or invent
            _print_results(
                pd["exact_hits"], pd["partial_hits"],
                pd["exact_count"], pd["partial_count"],
                pd["intent"], display_exact, display_partial,
            )

            messages = build_prompt(
                pd["search_question"],
                pd["exact_hits"], pd["partial_hits"],
                pd["exact_count"], pd["partial_count"],
                pd["intent"], history,
                display_exact=display_exact,
                display_partial=display_partial,
            )
            try:
                response = ollama.chat(
                    model=current_model,
                    messages=messages,
                    options={"temperature": TEMPERATURE},
                )
                answer = response.message.content
            except Exception as e:
                print(f"❌ Ollama error: {e}")
                continue

            print(f"Assistant: {answer}\n")

            # Save as last_results so user can ask for more without re-searching
            last_results = pd

            tip = _suggest_refinement(pd["intent"], pd["exact_count"], pd["partial_count"])
            if tip:
                print(tip)

            history.append({"role": "user",      "content": pd["search_question"]})
            history.append({"role": "assistant", "content": answer})
            if len(history) > HISTORY_TURNS * 2:
                history = history[-(HISTORY_TURNS * 2):]
            continue

        # ── State: waiting for clarification reply ─────────────────────────────
        if pending_clarification:
            original_q            = pending_clarification
            pending_clarification = None
            search_question = original_q if user_input.lower() in ("skip", "no", "") \
                              else f"{original_q} {user_input}"
        # ── Refinement of previous results — no new search needed ──────────────
        elif _is_refinement(user_input, last_results):
            lr = last_results
            display_exact, display_partial = _parse_display_request(user_input, top_k)

            if verbose:
                print(f"\n🔍 Reusing last results — showing {display_exact} exact, {display_partial} partial")

            # Print results from Python directly
            _print_results(
                lr["exact_hits"], lr["partial_hits"],
                lr["exact_count"], lr["partial_count"],
                lr["intent"], display_exact, display_partial,
            )

            messages = build_prompt(
                lr["search_question"],
                lr["exact_hits"], lr["partial_hits"],
                lr["exact_count"], lr["partial_count"],
                lr["intent"], history,
                display_exact=display_exact,
                display_partial=display_partial,
            )
            try:
                response = ollama.chat(
                    model=current_model,
                    messages=messages,
                    options={"temperature": TEMPERATURE},
                )
                answer = response.message.content
            except Exception as e:
                print(f"❌ Ollama error: {e}")
                continue

            print(f"Assistant: {answer}\n")
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": answer})
            if len(history) > HISTORY_TURNS * 2:
                history = history[-(HISTORY_TURNS * 2):]
            continue
        else:
            search_question = user_input

        # ── RAG pipeline ───────────────────────────────────────────────────────
        exact_hits, partial_hits, exact_count, partial_count, intent, clarification = \
            retrieve_and_filter(search_question, collection, embed_model, top_k)

        # Clarification needed: ask and wait — do NOT call Ollama yet
        if clarification and search_question == user_input:
            print(f"\n❓ {clarification}\n   (type your answer, or 'skip' to search anyway)")
            pending_clarification = user_input
            continue

        # Print real counts from Python (not the LLM)
        total = exact_count + partial_count
        print(f"\nFound {exact_count:,} exact + {partial_count:,} partial matches.")

        if total == 0:
            print("   Try broadening your search (wider date range or remove location filter).\n")
            continue

        # Ask how many to display — store results, wait for reply
        print(f"   How many would you like to see? (1–{min(top_k, total)}, default {DEFAULT_DISPLAY})")
        pending_display = {
            "search_question": search_question,
            "exact_hits":      exact_hits,
            "partial_hits":    partial_hits,
            "exact_count":     exact_count,
            "partial_count":   partial_count,
            "intent":          intent,
        }
        continue


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copernicus RAG Chatbot (Ollama + ChromaDB)"
    )
    parser.add_argument("--model",       type=str,  default=None,
                        help="Ollama model name to use")
    parser.add_argument("--ask",         type=str,  default=None,
                        help="Single question mode (non-interactive)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    parser.add_argument("--verbose",     action="store_true",
                        help="Show detected filters and retrieval scores")
    parser.add_argument("--top",         type=int,  default=DEFAULT_TOP_K,
                        help=f"Retrieval pool size (default {DEFAULT_TOP_K})")

    run_chat(parser.parse_args())
