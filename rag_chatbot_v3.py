#!/usr/bin/env python3
"""
rag_chatbot_v3.py
=================
STAC-first satellite data assistant.

ARCHITECTURE OVERVIEW
─────────────────────
Every user message goes through this pipeline:

  User text
      │
      ▼
  [1] extract_intent()          ← Ollama/mistral at temperature=0 → JSON Intent object
      │                           (satellite, theme, location, dates, mode …)
      │
      ▼
  [2] _keyword_enhance()        ← Pure-Python post-processing on top of the LLM output.
      │                           Catches what Ollama missed: cloud cover regex,
      │                           year regex, CMR collection IDs, request_type derivation.
      │
      ▼
  [3] Routing (run_chat)        ← Inspects intent.request_type:
      │                           • "collection_discovery" → _handle_collection_discovery()
      │                           • "item_preview / count / links" → _handle_item_search()
      │                           • "item_by_id"  → _handle_item_by_id()
      │                           • "item_export" → _handle_item_search() then _handle_export()
      │
      ▼
  [4a] Mode A — collection_discovery:
      │   search_stac_collections() → ChromaDB cosine similarity on 5,999 collection
      │   documents → rerank by platform/spatial/temporal validation → display cards
      │   → synthesize_collection_results() → LLM summary.
      │
      ▼
  [4b] Mode B — item_search:
      │   _fresh_search_or_last() → ChromaDB (same as Mode A) → pick best collection
      │   → confirm with user if score is low / ambiguous
      │   → resolve_bbox() → build STAC POST /search payload
      │   → res.stac_searcher.search() → live STAC API call → display item cards
      │   → synthesize_item_results() → LLM summary.

TWO MODES in detail:
  Mode A  collection_discovery   "What data exists for flood detection?"
          → Returns metadata CARDS from ChromaDB (titles, providers, search URLs).
          → No live API call. No real counts.

  Mode B  item_search            "Show me Sentinel-2 over Toulouse in 2023"
          → Finds the right collection via ChromaDB, then calls the provider's
            STAC /search endpoint with bbox + datetime filters.
          → Returns REAL satellite products (scenes/granules) with actual dates,
            cloud cover, and asset download links.

DATA SOURCES (runtime):
  - chroma_db/stac_collections       5,999 real collections from 17 providers
  - kb/outputs/stac_providers.jsonl  search_url, search_method, CQL2 support per provider
  - kb/outputs/geo_index.json        location name → lat/lon (for bbox resolution)

NOT used (deprecated, kept on disk for reference):
  - kb/outputs/query_lookup.jsonl    synthetic — NOT real satellite acquisitions
  - chroma_db/copernicus_grouped     superseded by stac_collections

COUNTING RULES:
  - Only report counts from live STAC /search responses (context.matched / numberMatched)
  - If provider does not expose count: say so — never fabricate a number

USAGE:
  python3 rag_chatbot_v3.py
  python3 rag_chatbot_v3.py --model mistral
  python3 rag_chatbot_v3.py --top 5 --verbose
  python3 rag_chatbot_v3.py --ask "Show me Sentinel-2 over Toulouse summer 2025"

In-chat commands: more | export | /help | /model | /verbose | /top | /clear | quit
"""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Missing: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    sys.exit("Missing: pip install chromadb")

try:
    import ollama
except ImportError:
    sys.exit("Missing: pip install ollama")

from stac_item_search import (
    STACItemSearcher, SearchResult, ParsedItem,
    _fetch, make_client, get_next_link, extract_count, parse_item,
)


# ── Constants ──────────────────────────────────────────────────────────────────
# These are the only values you need to change if you move databases or switch models.

# Sentence-transformer model used to embed both the ChromaDB documents and the
# user's query at search time.  384-dimensional cosine-similarity vectors.
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"

# Local directory where ChromaDB stores its persistent collection files.
CHROMA_DIR         = "./chroma_db"

# Name of the ChromaDB collection that holds the 5,999 STAC collection documents.
STAC_COLLECTION    = "stac_collections"

# JSONL file with one record per provider: root URL, /search endpoint, HTTP method,
# CQL2-filter support flag.  Loaded at startup by STACItemSearcher.
STAC_PROVIDERS     = Path("kb/outputs/stac_providers.jsonl")

# JSON file mapping city/location names → {lat, lon, country}.
# Built by build_geo_index.py.  Used to turn "Toulouse" into a bbox for STAC queries.
GEO_INDEX          = Path("kb/outputs/geo_index.json")

# Directory where exported JSONL files are written (created on first export).
EXPORT_DIR         = Path("kb/exports")

# Number of items returned per STAC /search page (the `limit` parameter).
ITEM_PAGE_SIZE     = 10

# If export total exceeds this, the user is asked to confirm before proceeding.
EXPORT_WARN_THRESH = 500

# Absolute maximum items that can be fetched in a single export session.
EXPORT_MAX_ITEMS   = 5_000

# How many times the chatbot will ask for more location/date info before giving up.
MAX_CLARIFICATIONS = 1

# Bounding box buffer added around a city point (±0.5° ≈ ±55 km at the equator).
# Without a buffer a single lat/lon point would rarely intersect satellite footprints.
BBOX_BUFFER_DEG    = 0.5

# Redis session persistence (optional — chatbot works without Redis, falls back to in-memory).
# Sessions survive restarts and are shared across processes on the same machine.
REDIS_URL         = "redis://localhost:6379"
REDIS_SESSION_TTL = 7 * 24 * 3600   # 7 days in seconds — sessions expire after a week of inactivity


# ── Intent dataclass ───────────────────────────────────────────────────────────
# This is the structured output of the LLM intent-extraction step.
# The LLM fills most fields; _keyword_enhance() and resolve_bbox() fill the rest.
# All routing and search decisions downstream read from this object.

@dataclass
class Intent:
    # "en" or "fr" — detected by the LLM from the user's writing style.
    language:         str            = "en"

    # High-level routing flag set by the LLM:
    #   "collection_discovery" → user wants to explore what datasets exist
    #   "item_search"          → user wants actual satellite products
    mode:             str            = "collection_discovery"

    # Fine-grained routing set by _keyword_enhance() (pure Python — never from LLM):
    #   collection_discovery  → Mode A: ChromaDB cards only
    #   item_count            → Mode B: count only, no cards displayed
    #   item_preview          → Mode B: show items + synthesis
    #   item_links            → Mode B: show items + print asset URLs
    #   item_export           → Mode B: search then immediately export to JSONL
    #   item_by_id            → fetch a single specific item by ID
    request_type:     str            = "collection_discovery"

    # Satellite family code: "S1" | "S2" | "S3" | "S5P" | None.
    # Used for ChromaDB platform validation and the satellite-guard in _fresh_search_or_last.
    satellite:        Optional[str]  = None

    # Main subject of the query (e.g. "flood", "vegetation", "SST").
    # Used as part of the ChromaDB semantic query string.
    theme:            Optional[str]  = None

    # Sensor type: "radar" | "optical" | "ocean" | "atmospheric" | None.
    # Used in the ChromaDB query alongside theme.
    mission_type:     Optional[str]  = None

    # City or region name (e.g. "Toulouse", "North Sea") — not a country name.
    # Passed to resolve_bbox() to get a [W,S,E,N] bounding box.
    location_text:    Optional[str]  = None

    # Lowercase English country name (e.g. "france").
    # Used as fallback if location_text doesn't resolve in geo_index.
    country:          Optional[str]  = None

    # ISO 8601 date strings for the query time window.
    # None means "no constraint" (open range).
    date_start:       Optional[str]  = None
    date_end:         Optional[str]  = None

    # True when the user explicitly asked to SEE items (show me, give me, display …).
    wants_items:      bool           = False

    # True when the user asked for a COUNT (how many, total …).
    wants_count:      bool           = False

    # True when the user asked to save results to disk.
    export_requested: bool           = False

    # Maximum cloud cover percentage filter (0–100).
    # Used with CQL2 filter if supported, otherwise applied as local post-filter.
    cloud_cover_max:  Optional[float]= None

    # Specific item ID the user named (e.g. "S2B_32TLQ_20250901_0_L2A").
    # Set → routes to item_by_id handler.
    item_id:          Optional[str]  = None

    # Collection ID hint from the LLM or from _keyword_enhance's CMR-ID fix.
    # Verified against ChromaDB before use — LLM output is NOT trusted directly.
    collection_id:    Optional[str]  = None

    # STAC provider base URL (e.g. "https://planetarycomputer.microsoft.com/api/stac/v1").
    # Never set by the LLM — always resolved from ChromaDB metadata.
    provider_root:    Optional[str]  = None

    # [W, S, E, N] bounding box for spatial filtering.
    # Filled by resolve_bbox() after intent extraction, or by _parse_manual_bbox()
    # if the user typed explicit coordinates.
    bbox:             Optional[list] = None

    # LLM self-reported confidence (0.0 – 1.0).  Informational only — not used for routing.
    confidence:       float          = 0.5

    # List of assumptions the LLM made (e.g. "imagery → assumed optical").
    # Displayed in verbose mode.
    assumptions:      list           = field(default_factory=list)

    # List of things the LLM was unsure about.  Displayed in verbose mode.
    ambiguities:      list           = field(default_factory=list)

    # One-sentence explanation of the LLM's reasoning.  Displayed in verbose mode.
    reasoning:        str            = ""


# ── Resources ──────────────────────────────────────────────────────────────────
# All heavy objects (model, DB client, HTTP client) are loaded once at startup
# and stored in a single Resources instance passed everywhere — no globals.

class Resources:
    def __init__(self):
        # Sentence-transformer model for embedding queries at search time.
        self.embed_model:    SentenceTransformer = None

        # ChromaDB collection handle for the 5,999 STAC collection documents.
        self.stac_col:       chromadb.Collection = None

        # HTTP client that knows how to POST /search to each provider.
        # Also carries the ProviderIndex (search URLs, CQL2 flags, auth types).
        self.stac_searcher:  STACItemSearcher    = None

        # Raw geo_index dict: {"locations": {name: {lat, lon, country}}, "countries": {…}}
        self.geo_index:      dict                = {}

        # Pre-computed country → [W, S, E, N] bbox aggregated from all city points.
        # Built at startup by build_country_bboxes(); used when user types a country name.
        self.country_bboxes: dict                = {}


def build_country_bboxes(locations: dict) -> dict:
    """
    Derive country → [W, S, E, N] from all city/location lat/lon records.
    Used as fallback bbox when the user names a country instead of a city.

    Algorithm: iterate every city record, track the min/max lat and lon seen
    for each country key → that envelope becomes the country bbox.
    """
    bboxes: dict[str, list] = {}
    for rec in locations.values():
        country = rec.get("country", "")
        if not country:
            continue
        lat, lon = rec.get("lat"), rec.get("lon")
        if lat is None or lon is None:
            continue
        key = country.lower()
        if key not in bboxes:
            bboxes[key] = [lon, lat, lon, lat]   # [W, S, E, N]
        else:
            b = bboxes[key]
            b[0] = min(b[0], lon)   # W
            b[1] = min(b[1], lat)   # S
            b[2] = max(b[2], lon)   # E
            b[3] = max(b[3], lat)   # N
    return {k: [round(v, 4) for v in b] for k, b in bboxes.items()}


class SessionStore:
    """
    Optional Redis-backed store for conversation history.

    Stores state["history"] (compact turn summaries) under the key
    'rag_session:<session_id>:history' so the conversation survives restarts
    and multiple users can maintain separate sessions on the same machine.

    Falls back silently to in-memory-only mode when Redis is not reachable or
    not installed — the chatbot works exactly as before, just without persistence.

    Usage:
        store = SessionStore("alice")      # uses --session alice
        history = store.load()             # [] if no prior session
        store.save(state["history"])       # called after every turn
        store.delete()                     # called by /clear command
        SessionStore.list_all()            # lists all active session IDs
    """

    def __init__(self, session_id: str,
                 url: str = REDIS_URL, ttl: int = REDIS_SESSION_TTL) -> None:
        self.session_id = session_id
        self._key       = f"rag_session:{session_id}:history"
        self._ttl       = ttl
        self._client    = None
        try:
            import redis as _redis
            r = _redis.from_url(url, socket_connect_timeout=1, decode_responses=True)
            r.ping()
            self._client = r
        except Exception:
            pass   # Redis not available — in-memory only

    @property
    def available(self) -> bool:
        return self._client is not None

    def load(self) -> list:
        """Return saved history list, or [] if nothing saved / Redis unavailable."""
        if not self._client:
            return []
        try:
            raw = self._client.get(self._key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return []

    def save(self, history: list) -> None:
        """Persist history to Redis with TTL refresh."""
        if not self._client or not history:
            return
        try:
            self._client.setex(self._key, self._ttl, json.dumps(history))
        except Exception:
            pass

    def delete(self) -> None:
        """Remove this session's history from Redis (called by /clear)."""
        if not self._client:
            return
        try:
            self._client.delete(self._key)
        except Exception:
            pass

    @staticmethod
    def list_all(url: str = REDIS_URL) -> list:
        """Return a list of all active session IDs stored in Redis."""
        try:
            import redis as _redis
            r = _redis.from_url(url, socket_connect_timeout=1, decode_responses=True)
            keys = r.keys("rag_session:*:history")
            return sorted(k.split(":")[1] for k in keys)
        except Exception:
            return []


# ── Memory Managers ───────────────────────────────────────────────────────────
#
# Three strategies mapping directly to the article sections:
#
#   Section 2 — RawMessageMemory (local Ollama replaces OpenAI)
#     Stores full user messages and assistant responses as a bounded list.
#     When the limit is hit, the oldest turns are dropped.
#     Equivalent to the simple message_history list in the OpenAI example.
#
#   Section 3 — SummaryMemory (LangChain ConversationSummaryMemory equivalent)
#     After every turn Ollama generates/updates a single rolling summary.
#     The summary is what the intent extractor sees as "recent context".
#     History never grows — only the summary text changes.
#
#   Section 4 — BufferedSummaryMemory (Llama-Index ChatSummaryMemoryBuffer equivalent)
#     Keeps the last MAX_RAW_TURNS turns verbatim in a buffer.
#     When the buffer overflows, Ollama compresses the oldest half into the
#     persistent summary.  Context = summary + recent raw turns.
#
# All three share the same interface (MemoryManager ABC) and are serialised
# to/from the plain list-of-dicts format used by SessionStore / Redis.

from abc import ABC, abstractmethod as _abstractmethod


class MemoryManager(ABC):
    """Abstract base for conversation memory strategies."""

    @_abstractmethod
    def get_context(self) -> str:
        """Return the context string to prepend to intent-extraction prompts."""

    @_abstractmethod
    def add_exchange(self, user_msg: str, assistant_msg: str, model: str) -> None:
        """Record one completed turn (user message + assistant response)."""

    @_abstractmethod
    def clear(self) -> None:
        """Wipe all stored memory."""

    @_abstractmethod
    def to_list(self) -> list[dict]:
        """Serialise to a list-of-dicts for Redis persistence."""

    @_abstractmethod
    def last_user_message(self) -> Optional[str]:
        """Return the content of the most recent user message (for deduplication)."""

    @property
    @_abstractmethod
    def turn_count(self) -> int:
        """Number of completed turns stored."""


# ── Section 2 ─────────────────────────────────────────────────────────────────

class RawMessageMemory(MemoryManager):
    """
    Stores the full text of every turn in a bounded list.
    Equivalent to the article's OpenAI message_history approach, but using
    local Ollama — no external API or cost.

    When the list exceeds max_turns the oldest entries are dropped (not summarised).
    Use SummaryMemory or BufferedSummaryMemory to avoid losing old context.
    """

    def __init__(self, max_turns: int = 16, messages: list[dict] | None = None):
        self._msgs     = list(messages or [])
        self._max      = max_turns * 2   # 2 messages per turn

    def get_context(self) -> str:
        if not self._msgs:
            return ""
        lines = []
        for m in self._msgs[-16:]:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "Recent context:\n" + "\n".join(lines)

    def add_exchange(self, user_msg: str, assistant_msg: str, model: str) -> None:
        self._msgs.append({"role": "user",      "content": user_msg})
        self._msgs.append({"role": "assistant", "content": assistant_msg})
        if len(self._msgs) > self._max:
            self._msgs = self._msgs[-self._max:]

    def clear(self) -> None:
        self._msgs.clear()

    def to_list(self) -> list[dict]:
        return list(self._msgs)

    def last_user_message(self) -> Optional[str]:
        for m in reversed(self._msgs):
            if m["role"] == "user":
                return m["content"]
        return None

    @property
    def turn_count(self) -> int:
        return len(self._msgs) // 2


# ── Section 3 ─────────────────────────────────────────────────────────────────

class SummaryMemory(MemoryManager):
    """
    LangChain ConversationSummaryMemory equivalent — fully local via Ollama.

    After every turn, Ollama updates a single rolling natural-language summary
    of the entire conversation.  The intent extractor only ever sees this summary,
    so the context window size stays constant regardless of conversation length.

    Trade-off: the very latest exchange may not yet be in the summary if Ollama
    is slow — BufferedSummaryMemory avoids this by also keeping recent raw turns.
    """

    _UPDATE_SYS = (
        "You maintain a concise summary of a satellite Earth Observation search session. "
        "Update the existing summary to include the new exchange. "
        "Keep it under 120 words. Focus on: satellite/collection used, location, "
        "time period, what was found or returned. "
        "Return ONLY the updated summary — no labels, no quotes, no explanations."
    )

    def __init__(self, summary: str = "", turn_count: int = 0,
                 last_user: str = ""):
        self._summary    = summary
        self._turn_count = turn_count
        self._last_user  = last_user

    def get_context(self) -> str:
        if not self._summary:
            return ""
        return f"Conversation summary so far:\n{self._summary}"

    def add_exchange(self, user_msg: str, assistant_msg: str, model: str) -> None:
        self._turn_count += 1
        self._last_user  = user_msg
        prompt = (
            f"Existing summary:\n{self._summary or '(no history yet)'}\n\n"
            f"New exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n\n"
            "Return the updated summary."
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": self._UPDATE_SYS},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.0},
            )
            self._summary = resp.message.content.strip()
        except Exception:
            self._summary = (
                (self._summary + "\n" if self._summary else "")
                + f"Turn {self._turn_count}: {assistant_msg[:120]}"
            )

    def clear(self) -> None:
        self._summary    = ""
        self._turn_count = 0
        self._last_user  = ""

    def to_list(self) -> list[dict]:
        return [{"role": "_summary", "content": self._summary,
                 "_turns": self._turn_count, "_last_user": self._last_user}]

    def last_user_message(self) -> Optional[str]:
        return self._last_user or None

    @property
    def turn_count(self) -> int:
        return self._turn_count


# ── Section 4 ─────────────────────────────────────────────────────────────────

class BufferedSummaryMemory(MemoryManager):
    """
    Llama-Index ChatSummaryMemoryBuffer equivalent — fully local via Ollama.

    Maintains two layers:
      • A compressed LLM summary of all turns older than MAX_RAW_TURNS.
      • A verbatim buffer of the last MAX_RAW_TURNS turns.

    Context seen by the intent extractor = summary + recent raw turns.
    This gives precise recent context (no summarisation loss) while keeping
    very long histories compressed in the background.

    When the raw buffer exceeds MAX_RAW_TURNS, Ollama is called to fold the
    oldest half into the persistent summary.
    """

    MAX_RAW_TURNS = 6   # keep this many turns verbatim; compress the rest

    def __init__(self, summary: str = "", raw: list[dict] | None = None,
                 turn_count: int = 0):
        self._summary    = summary
        self._raw        = list(raw or [])
        self._turn_count = turn_count

    def get_context(self) -> str:
        parts = []
        if self._summary:
            parts.append(f"Earlier conversation (compressed):\n{self._summary}")
        if self._raw:
            lines = []
            for m in self._raw:
                role = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"{role}: {m['content']}")
            parts.append("Recent turns:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _compress_oldest(self, model: str) -> None:
        """LLM-compress the oldest half of _raw into _summary."""
        half       = len(self._raw) // 2
        to_fold    = self._raw[:half]
        self._raw  = self._raw[half:]
        lines = []
        for m in to_fold:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        block  = "\n".join(lines)
        prompt = (
            f"Existing summary:\n{self._summary or '(none)'}\n\n"
            f"Older turns to add:\n{block}\n\n"
            "Merge into a single updated summary under 120 words. "
            "Focus on satellite, collection, location, time period, results. "
            "Return ONLY the summary text."
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            self._summary = resp.message.content.strip()
        except Exception:
            tail = " | ".join(
                m["content"][:80] for m in to_fold if m["role"] == "assistant"
            )
            self._summary = (f"{self._summary} | {tail}" if self._summary else tail)

    def add_exchange(self, user_msg: str, assistant_msg: str, model: str) -> None:
        self._turn_count += 1
        self._raw.append({"role": "user",      "content": user_msg})
        self._raw.append({"role": "assistant", "content": assistant_msg})
        if len(self._raw) > self.MAX_RAW_TURNS * 2:
            self._compress_oldest(model)

    def clear(self) -> None:
        self._summary    = ""
        self._raw.clear()
        self._turn_count = 0

    def to_list(self) -> list[dict]:
        return [
            {"role": "_summary", "content": self._summary, "_turns": self._turn_count},
            *self._raw,
        ]

    def last_user_message(self) -> Optional[str]:
        for m in reversed(self._raw):
            if m["role"] == "user":
                return m["content"]
        return None

    @property
    def turn_count(self) -> int:
        return self._turn_count


def create_memory(memory_type: str, data: list[dict] | None = None) -> MemoryManager:
    """
    Factory: build the right MemoryManager from a type string + optional
    Redis-serialised data (from a previous session).

    memory_type: "raw" | "summary" | "buffer"
    data:        the list returned by store.load() — may be None or [] for a fresh session.
    """
    data = data or []
    if memory_type == "summary":
        if data and data[0].get("role") == "_summary":
            return SummaryMemory(
                summary    = data[0].get("content", ""),
                turn_count = data[0].get("_turns", 0),
                last_user  = data[0].get("_last_user", ""),
            )
        return SummaryMemory()
    if memory_type == "buffer":
        if data and data[0].get("role") == "_summary":
            return BufferedSummaryMemory(
                summary    = data[0].get("content", ""),
                raw        = data[1:],
                turn_count = data[0].get("_turns", 0),
            )
        return BufferedSummaryMemory(raw=data)
    # default: "raw"
    return RawMessageMemory(messages=data)


def load_resources() -> Resources:
    """Load all heavy resources once at startup.  Exits if critical ones are missing."""
    res = Resources()

    # Step 1: embedding model — needed to encode every ChromaDB query at search time.
    # First run downloads ~90 MB to ~/.cache/huggingface/.
    print("Loading embedding model …")
    res.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 2: ChromaDB — open the persistent on-disk store and get the collection handle.
    # The collection must already exist (built by embed_stac.py).
    print("Connecting to ChromaDB …")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        res.stac_col = client.get_collection(STAC_COLLECTION)
        print(f"  {STAC_COLLECTION}: {res.stac_col.count():,} collections")
    except Exception:
        sys.exit(
            f"Collection '{STAC_COLLECTION}' not found.\n"
            "Run: python3 embed_stac.py --force"
        )

    # Step 3: geo_index — location name → {lat, lon} mapping for bbox resolution.
    if GEO_INDEX.exists():
        raw = json.loads(GEO_INDEX.read_text(encoding="utf-8"))
        res.geo_index = raw
        locs = raw.get("locations", {})
        print(f"  geo_index: {len(locs):,} locations, "
              f"{len(raw.get('countries', {})):,} countries")
        # Pre-compute country envelopes from all city points so country lookups are O(1).
        res.country_bboxes = build_country_bboxes(locs)
    else:
        print(f"  [warn] {GEO_INDEX} not found — bbox resolution disabled")

    if not STAC_PROVIDERS.exists():
        print(f"  [warn] {STAC_PROVIDERS} not found — live search may fail")

    # Step 4: STACItemSearcher — wraps httpx, loads provider configs, manages pagination.
    res.stac_searcher = STACItemSearcher(
        providers_path=str(STAC_PROVIDERS),
        timeout=30,
        retries=3,
    )

    return res


# ── Utilities ──────────────────────────────────────────────────────────────────

def _now_ts() -> str:
    """Return current UTC time as an ISO 8601 string (used for export filename timestamps)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_text(text: str) -> str:
    """
    Lowercase + strip accents (NFKD normalization removes combining diacritics).
    Example: "Toulouse" → "toulouse", "Île-de-France" → "ile-de-france".
    Used for case/accent-insensitive geo_index and named-region lookups.
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip()


def _datetime_range(intent: Intent) -> Optional[str]:
    """
    Build a STAC-compatible datetime range string "start/end" from the intent.
    The STAC spec accepts open ranges like "2023-01-01/.." but many providers
    reject them, so we cap open-ended ranges to today's date instead.
    Returns None if no date was specified (means: no datetime filter in the search).
    """
    if intent.date_start and intent.date_end:
        return f"{intent.date_start}/{intent.date_end}"
    if intent.date_start:
        # Cap open-ended ranges to today so providers don't return unbounded results.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{intent.date_start}/{today}"
    return None


# Regex to extract an explicit [W, S, E, N] bounding box typed by the user.
# Supports three formats:
#   bbox=1.3,43.5,1.6,43.8   (equals sign, optional brackets)
#   bbox 1.3,43.5,1.6,43.8   (space separator)
#   [1.3,43.5,1.6,43.8]      (bare array literal)
# The regex has three alternatives × 4 capture groups = 12 groups total.
_MANUAL_BBOX_RE = re.compile(
    # bbox= or bbox: W,S,E,N
    r'\bbbox\s*[=:]\s*\[?(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]?'
    r'|'
    # bbox W,S,E,N (space, no equals) — matches "bbox 1.30,43.50,1.60,43.75"
    r'\bbbox\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
    r'|'
    # [W,S,E,N] array literal
    r'\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]',
    re.IGNORECASE,
)


def _parse_manual_bbox(text: str) -> Optional[list]:
    """
    Parse explicit bbox from user input: 'bbox=W,S,E,N', 'bbox W,S,E,N', or '[W,S,E,N]'.
    Returns [W, S, E, N] as floats, or None if no bbox pattern is found.
    Walks groups in chunks of 4 because the regex has three alternatives.
    """
    m = _MANUAL_BBOX_RE.search(text)
    if not m:
        return None
    groups = m.groups()
    # Three patterns × 4 groups each; find first non-None quartet
    for start in range(0, len(groups), 4):
        if groups[start] is not None:
            try:
                return [float(groups[start + i]) for i in range(4)]
            except (TypeError, ValueError):
                pass
    return None


# Ordinal word → 1-based index mapping.
# Used in _resolve_collection_ref() to handle "the second one", "3rd collection" etc.
_NUM_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}

# Cardinal aliases for bare number words like "give me two".
_ORD_ALIASES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}

# Matches "all these collections" / "search all these" / "all of them" / "the rest" etc.
# → search every displayed collection.
_ALL_THESE_RE = re.compile(
    r'\ball\s+(of\s+)?(these|them)\b'
    r'|search\s+all\s+these'
    r'|\bevery\s+(one|collection|result)\b'
    r'|\bsearch\s+them\s+all\b'
    r'|\bthe\s+rest\s+(of\s+(them|the\s+collections?))?\b',
    re.IGNORECASE,
)

# Matches "other 4" → search 4 collections that weren't the last one tried.
_OTHER_N_RE   = re.compile(r'\bother\s+(\d+)\b', re.IGNORECASE)

# Matches "other collections" / "other ones too" → search the remaining collections.
_OTHER_RE     = re.compile(
    r'\bother\s+(collections?|ones?|results?|those)\b'
    r'|\bother\b.*\b(collections?|too|also)\b',
    re.IGNORECASE,
)

# Matches "the same", "same but with cloud < 20%", "that one", etc.
# When this fires on the current input, using_last=True is forced so we reuse the
# last collection without triggering a confirmation prompt even if the score gap is small.
_SAME_AS_LAST_RE = re.compile(
    r'\b(the\s+same|same\s+one|same\s+(but|with|except|for|collection|data)'
    r'|that\s+one|this\s+one|those\s+ones?|keep\s+(the\s+)?same)\b',
    re.IGNORECASE,
)

# Matches "export 50" / "export 200" to extract the N from the export command.
_EXPORT_N_RE  = re.compile(r'\bexport\s+(\d+)\b', re.IGNORECASE)

# Matches questions like "what was exported?", "tell me about the export", "last export".
# When this fires and last_search_result exists, routes to _handle_export_summary()
# instead of triggering a new search.
_ABOUT_EXPORT_RE = re.compile(
    r'\b(about\s+(the\s+)?(export|exported|items?)'
    r'|what\s+(was|were|is|are)\s+(exported|in\s+the\s+export)'
    r'|the\s+\d+\s+items?\s+exported'
    r'|last\s+export'
    r'|export\s+(details?|summary|info|results?))\b',
    re.IGNORECASE,
)


def _resolve_collection_ref(user_text: str, hits: list[dict]) -> Optional[dict]:
    """
    Single-item reference helper: 'second one', 'collection 3', 'the 2nd', etc.
    Returns the matching hit dict, or None if no reference is found.
    """
    if not hits:
        return None
    t = user_text.lower()

    m = re.search(r'(?:collection\s*#?|#|\[)\s*(\d+)', t)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            return hits[idx]

    m = re.search(r'(?:^|\bthe\s+|number\s+)(\d)\b', t)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            return hits[idx]

    words = set(t.split())
    for word, n in {**_NUM_WORDS, **_ORD_ALIASES}.items():
        if word in words:
            idx = n - 1
            if 0 <= idx < len(hits):
                return hits[idx]

    return None


def _resolve_collection_refs_list(
    user_text: str,
    hits: list[dict],
    last_tried_id: Optional[str] = None,
) -> Optional[list[dict]]:
    """
    Extended reference resolver — returns a list of hits or None.

    Supports:
      "all these" / "search all these collections"   → all hits (capped at 5)
      "other 4" / "search the other 4 collections"   → hits excluding last_tried_id, up to 4
      "other collections" / "other ones too"          → remaining hits (up to 5)
      "second one" / "collection 3" / "2"             → [hits[n]]
      bare single digit "2"                           → [hits[1]]
    """
    if not hits:
        return None
    t = user_text.lower().strip()

    # "all these" / "search all these collections"
    if _ALL_THESE_RE.search(t):
        return hits[:5]

    # "other N"
    m = _OTHER_N_RE.search(t)
    if m:
        n = int(m.group(1))
        remaining = [h for h in hits if h["collection_id"] != last_tried_id] \
                    if last_tried_id else hits[1:]
        return remaining[:n] if remaining else None

    # "other collections" / "other ones too"
    if _OTHER_RE.search(t):
        remaining = [h for h in hits if h["collection_id"] != last_tried_id] \
                    if last_tried_id else hits[1:]
        return remaining[:5] if remaining else None

    # single-item ordinal/numeric references
    single = _resolve_collection_ref(user_text, hits)
    if single is not None:
        return [single]

    # bare single digit
    m = re.search(r'^\s*(\d)\s*$', t)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            return [hits[idx]]

    return None


# ── Intent extraction ──────────────────────────────────────────────────────────
# The LLM (Ollama/mistral) is called at temperature=0 to extract a structured JSON
# Intent from free-text user input.  The system prompt below is injected with today's
# date so seasonal/relative dates ("last summer", "this year") resolve correctly.
# The LLM output is parsed in extract_intent() and validated by _keyword_enhance().

_INTENT_SYSTEM = """\
You are an intent extraction assistant for a satellite Earth Observation search system.
Return ONLY a JSON object — no markdown, no explanation.

MODE SELECTION (critical):
- "collection_discovery": user wants to KNOW ABOUT or EXPLORE datasets/collections.
  Examples: "What data for ocean?", "Which SAR collections?", "I need flood data"
- "item_search": user wants ACTUAL SATELLITE PRODUCTS for a specific place/time.
  Examples: "Show me Sentinel-2 over Toulouse", "How many scenes in Paris 2023?",
            "Give me download links", "How many products?", "Find images last summer"

Item-search trigger keywords: show me, give me, find me, how many, count, images,
  scenes, products, items, files, acquisitions, download, links, available

SATELLITES:
  S1  -> SAR, radar, flood, deforestation, all-weather, backscatter, subsidence, ice
  S2  -> optical, multispectral, NDVI, vegetation, land cover, agriculture, burned area
  S3  -> ocean, SST, sea surface temperature, chlorophyll, marine, coastal, fire
  S5P -> atmosphere, NO2, methane, CH4, ozone, air quality, pollution

DATE RULES (strict):
  - If no date/year/season/month is mentioned: set date_start=null and date_end=null.
  - If a month/season is mentioned without a year, use TODAY_YEAR as the default year.
  - "from may 1" with no year -> date_start="TODAY_YEAR-05-01", date_end="TODAY_YEAR-05-01"
  - "2023" -> date_start="2023-01-01", date_end="2023-12-31"
  - "summer 2025" -> date_start="2025-06-01", date_end="2025-08-31"
  - "spring 2024" -> date_start="2024-03-01", date_end="2024-05-31"
  - "autumn 2023" -> date_start="2023-09-01", date_end="2023-11-30"
  - "winter 2024" -> date_start="2024-12-01", date_end="2025-02-28"

LOCATION RULES:
  - location_text: city, region, geographic area (NOT country name)
  - country: lowercase English country name only — ALWAYS translate to English
    ("brasil" → "brazil", "espana" → "spain", "deutschland" → "germany",
     "maroc" → "morocco", "algerie" → "algeria", "france" stays "france")
  - "Toulouse" -> location_text="Toulouse", country=null
  - "France"   -> location_text=null, country="france"
  - "brasil"   -> location_text=null, country="brazil"
  - "over Paris in France" -> location_text="Paris", country="france"

LANGUAGE: "fr" only if the user writes in French.

JSON SCHEMA:
{{
  "language":         "en" or "fr",
  "mode":             "collection_discovery" or "item_search",
  "satellite":        "S1" or "S2" or "S3" or "S5P" or null,
  "theme":            "<one main topic keyword or null>",
  "mission_type":     "radar" or "optical" or "ocean" or "atmospheric" or null,
  "location_text":    "<city or region name or null>",
  "country":          "<lowercase English country name or null>",
  "date_start":       "YYYY-MM-DD" or null,
  "date_end":         "YYYY-MM-DD" or null,
  "wants_items":      true or false,
  "wants_count":      true or false,
  "export_requested": true or false,
  "cloud_cover_max":  <number 0-100 or null>,
  "item_id":          "<specific item ID string or null>",
  "collection_id":    "<collection ID if user explicitly named it or null>",
  "confidence":       <0.0 to 1.0>,
  "assumptions":      ["<assumption made>"],
  "ambiguities":      ["<what is unclear>"],
  "reasoning":        "<one short sentence>"
}}

Respond with JSON ONLY.
"""


def _build_intent_system() -> str:
    """
    Build the system prompt with today's date injected.
    TODAY_YEAR is replaced in the template so the LLM can resolve
    "from may 1" (no year given) to the correct calendar year.
    We avoid an f-string over the whole JSON schema block to prevent
    accidental brace interpretation in the schema literals.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year  = today[:4]
    return (
        f"Today's date is {today}. "
        f"Use this as the reference for relative date expressions.\n\n"
        + _INTENT_SYSTEM.replace("TODAY_YEAR", year)
    )


def extract_intent(user_text: str, memory: "MemoryManager", model: str) -> Intent:
    """
    Call Ollama at temperature=0 to parse the user's free text into a structured Intent.

    Context is supplied by the MemoryManager — which strategy is active determines
    whether the LLM sees raw recent turns (RawMessageMemory), a rolling summary
    (SummaryMemory), or a compressed summary + recent verbatim buffer (BufferedSummaryMemory).

    On any failure (Ollama unreachable, bad JSON) returns a minimal Intent with
    confidence=0.1 so the chatbot can still give a graceful error.
    """
    ctx       = memory.get_context()
    ctx_block = f"{ctx}\n" if ctx else ""
    prompt    = f"{ctx_block}Current message: {user_text}"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _build_intent_system()},
                {"role": "user",   "content": prompt},
            ],
            # format="json" tells Ollama to constrain output to valid JSON.
            # Combined with temperature=0 this gives very deterministic results.
            format="json",
            options={"temperature": 0.0},
        )
        raw = response.message.content.strip()
        # Strip accidental markdown code fences that some models add despite format="json".
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.rstrip())
        data = json.loads(raw)
    except Exception as exc:
        # Graceful degradation: return a minimal intent so the rest of the pipeline
        # can still try to handle the query rather than crashing.
        return Intent(
            confidence=0.1,
            reasoning=f"Intent extraction failed: {exc}",
            ambiguities=["Could not parse intent — please rephrase"],
        )

    def _s(v) -> Optional[str]:
        return str(v).strip() if v not in (None, "", "null") else None

    def _f(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return Intent(
        language         = str(data.get("language", "en")),
        mode             = str(data.get("mode", "collection_discovery")),
        satellite        = _s(data.get("satellite")),
        theme            = _s(data.get("theme")),
        mission_type     = _s(data.get("mission_type")),
        location_text    = _s(data.get("location_text")),
        country          = (_s(data.get("country")) or "").lower() or None,
        date_start       = _s(data.get("date_start")),
        date_end         = _s(data.get("date_end")),
        wants_items      = bool(data.get("wants_items", False)),
        wants_count      = bool(data.get("wants_count", False)),
        export_requested = bool(data.get("export_requested", False)),
        cloud_cover_max  = _f(data.get("cloud_cover_max")),
        item_id          = _s(data.get("item_id")),
        collection_id    = _s(data.get("collection_id")),
        confidence       = float(data.get("confidence", 0.5)),
        assumptions      = list(data.get("assumptions", [])),
        ambiguities      = list(data.get("ambiguities", [])),
        reasoning        = str(data.get("reasoning", "")),
    )


# ── Keyword fallback ───────────────────────────────────────────────────────────
# _keyword_enhance() runs AFTER the LLM and corrects / supplements its output
# using deterministic Python rules.  This is the safety net for common LLM failures.

# Verbs that unambiguously mean "show me items" — override collection_discovery if present.
_DISPLAY_VERBS = {"show me", "give me", "get me", "find me", "display", "list", "fetch"}

# Product-type nouns — present in both count AND display queries; not display-only on their own.
_PRODUCT_NOUNS = {"products", "items", "scenes", "images", "files", "acquisitions", "datasets"}

# Words that trigger item_links request_type.
_LINK_WORDS    = {"download link", "download links", "links", "href"}

# Words that trigger wants_count=True.
_COUNT_TRIGGERS = {"how many", "how much", "count", "total"}

# Words that imply optical mission_type when the LLM didn't detect one.
_IMAGERY_WORDS  = {"image", "images", "imagery", "scene", "scenes", "picture",
                   "pictures", "photo", "photos"}

# Cloud-cover regex fallback: catches "cloud cover < 20%", "under 30%", "max 15% cloud" etc.
# The LLM sometimes forgets this when the query is complex or contains conversation history.
_CC_RE   = re.compile(
    r'\bcloud(?:\s+cover)?\s*(?:under|below|less\s+than|<|max\.?\s*of|≤)?\s*(\d+)\s*%',
    re.IGNORECASE,
)

# Detects "less clouds", "fewer clouds", "clearer", "cloud-free" etc. without an explicit %.
# When matched and no explicit cloud_cover_max, we default to 20 % as a reasonable threshold.
_LESS_CLOUD_RE = re.compile(
    r'\b(less|fewer|lower|minimal|minimum|reduce)\s+(cloud|clouds|cloud\s*cover)'
    r'|\b(clearer|cloud[-\s]?free|clear\s+sky|low\s+cloud)',
    re.IGNORECASE,
)

# Year regex fallback: catches bare "2023", "in 2024" etc. when the LLM drops the date.
# Broad by design — could match years in collection names, but that's a low-risk edge case.
_YEAR_RE = re.compile(r'\b(20\d{2})\b')

# Meta-question detector: routes to _handle_meta_question() instead of a new search.
# Fires on conversational follow-ups like "why are there so many?", "explain this",
# "how do these compare?" — only when prior results exist in state.
_META_Q_RE = re.compile(
    r'\b(why\s+(are|is|were|was|do|does|did)\s+(there|that|this|it|they|so\s+many|so\s+few)'
    r'|how\s+do\s+(these|this|they|that)\s+compare'
    r'|what\s+does\s+(this|that|it)\s+mean'
    r'|is\s+(this|that)\s+(normal|expected|a\s+lot|too\s+many|too\s+few)'
    r'|explain\s+(this|that|these|those|the\s+results?)'
    r'|tell\s+me\s+more\s+about\s+(this|that|these|those)'
    r'|what\s+(was|were|is|are)\s+(shown|found|returned|displayed))\b',
    re.IGNORECASE,
)


def _keyword_enhance(intent: Intent, user_text: str) -> Intent:
    """
    Post-process the LLM's Intent with deterministic keyword rules.
    This runs after extract_intent() and corrects common LLM failures.

    Rules applied in order:
      1. Imagery words → assume optical if mission_type not already set.
      2. Display verbs / product nouns → set wants_items + switch to item_search.
      3. Count triggers → set wants_count; "how many" alone → pure count (no display).
      4. Cloud cover regex fallback → fill cloud_cover_max the LLM missed.
      5. Year regex fallback → fill date_start/date_end the LLM missed.
      6. S3 generic ocean theme → expand to SST/SLSTR/OLCI for better ChromaDB recall.
      7. CMR-style collection IDs (colons, no slashes) → move from item_id → collection_id.
      8. Derive request_type (the fine-grained routing key) — never from the LLM.
    """
    t = user_text.lower()

    # Rule 1 — imagery words → assume optical if the LLM didn't set a mission_type.
    # "show me images" → optical, not radar or atmospheric.
    words = set(t.split())
    if _IMAGERY_WORDS & words and not intent.mission_type:
        intent.mission_type = "optical"
        intent.assumptions.append("imagery/images/scenes → assumed optical mission type")

    has_display_verb  = any(v in t for v in _DISPLAY_VERBS)
    has_count_trigger = any(c in t for c in _COUNT_TRIGGERS)
    has_product_noun  = any(n in words for n in _PRODUCT_NOUNS)
    has_link_word     = any(lw in t for lw in _LINK_WORDS)
    # Pure count = "how many" present with no display verb ("show me", "give me" …)
    is_pure_count     = has_count_trigger and not has_display_verb

    # Rule 2 — display verb OR (product noun without count) OR link request → show items.
    # "give me scenes" → item_preview; "how many scenes?" → item_count (not item_preview).
    if has_display_verb or (has_product_noun and not has_count_trigger) or has_link_word:
        intent.wants_items = True
        if intent.mode == "collection_discovery":
            intent.mode = "item_search"

    # Rule 3 — count trigger handling.
    # "how many products are there?" → item_count only (no card display).
    # "show me how many" → item_preview (both items AND count shown).
    if has_count_trigger:
        intent.wants_count = True
        if intent.mode == "collection_discovery":
            intent.mode = "item_search"
        if is_pure_count:
            intent.wants_items = False  # don't display cards, just report the number
        else:
            intent.wants_items = True   # "show me and how many" → display items too

    if "export" in t or "save all" in t:
        intent.export_requested = True

    # Rule 4 — cloud cover regex fallback.
    # Ollama drops this field when the query is complex or contains conversation history noise.
    if intent.cloud_cover_max is None:
        m = _CC_RE.search(user_text)
        if m:
            intent.cloud_cover_max = float(m.group(1))
        elif _LESS_CLOUD_RE.search(user_text):
            intent.cloud_cover_max = 20.0   # default threshold for vague "less clouds"

    # Rule 5 — year regex fallback.
    # Ollama sometimes omits the date when the history contains many prior turns.
    if not intent.date_start:
        m = _YEAR_RE.search(user_text)
        if m:
            yr = m.group(1)
            intent.date_start = f"{yr}-01-01"
            intent.date_end   = f"{yr}-12-31"

    # Rule 6 — S3 generic ocean theme expansion.
    # "ocean" / "ocean data" alone doesn't retrieve EUMETSAT SST collections well from
    # ChromaDB; expanding to the full vocabulary ("SST SLSTR OLCI …") raises their score.
    _S3_GENERIC_OCEAN = {"ocean", "ocean data", "marine data", "ocean monitoring", "sea data", "marine"}
    if intent.satellite == "S3" and intent.theme and intent.theme.lower() in _S3_GENERIC_OCEAN:
        intent.theme = "sea surface temperature SST SLSTR OLCI chlorophyll ocean colour"

    # Rule 7 — CMR-style collection IDs contain colons but no slashes.
    # Example: "EO:EUM:DAT:SENTINEL-3:SL_2_WST___NTC_2017-07-05"
    # Ollama confuses these for item IDs; we move them to collection_id instead.
    if intent.item_id and ":" in intent.item_id and "/" not in intent.item_id:
        intent.collection_id = intent.item_id
        intent.item_id       = None
        intent.mode          = "item_search"
        intent.wants_items   = True

    # Rule 8 — derive request_type (the fine-grained routing key used in run_chat).
    # This is computed from pure Python logic — the LLM never sets request_type directly.
    # Priority order: item_by_id > export > links > count > preview > collection_discovery.
    if intent.item_id:
        intent.request_type = "item_by_id"
    elif intent.export_requested:
        intent.request_type = "item_export"
    elif has_link_word or lower_is_links(t):
        intent.request_type = "item_links"
    elif intent.wants_count and not intent.wants_items:
        intent.request_type = "item_count"
    elif intent.wants_items or intent.mode == "item_search":
        intent.request_type = "item_preview"
    else:
        intent.request_type = "collection_discovery"

    return intent


def lower_is_links(t: str) -> bool:
    """Return True for bare 'links' commands that are not caught by _LINK_WORDS substring check."""
    return t.strip() in ("links", "download links", "show links", "asset links", "give links")


# ── Geo resolution ─────────────────────────────────────────────────────────────
# resolve_bbox() converts a location name into a [W, S, E, N] bounding box.
# It tries four sources in priority order:
#   1. _NAMED_REGION_BBOXES — hardcoded geographic regions (seas, continents)
#      that are not cities and therefore don't appear in the geo_index.
#   2. geo_index exact match — city/town name → lat/lon → ±0.5° box.
#   3. geo_index partial match — substring search in city names.
#   4. country_bboxes — aggregated envelope from all cities in that country.
# Returns (None, reason_string) if no bbox can be resolved.

# Hardcoded bboxes [W, S, E, N] for geographic regions too large or too non-city
# to appear in the geo_index (seas, continents, deserts, major basins).
# All values are in WGS-84 decimal degrees.
_NAMED_REGION_BBOXES: dict[str, list] = {
    "mediterranean":     [-6.0,  30.0,  36.5,  46.0],
    "mediterranean sea": [-6.0,  30.0,  36.5,  46.0],
    "north sea":         [-5.0,  51.0,  10.0,  62.0],
    "baltic sea":        [9.0,   53.0,  30.5,  65.5],
    "black sea":         [27.0,  40.5,  42.0,  46.8],
    "red sea":           [32.0,  12.0,  44.0,  30.0],
    "persian gulf":      [48.0,  22.0,  57.0,  30.5],
    "arctic":            [-180.0, 66.5, 180.0,  90.0],
    "antarctic":         [-180.0,-90.0, 180.0, -60.0],
    "sahara":            [-17.0,  15.0,  51.0,  37.0],
    "amazon":            [-73.0,  -9.0, -44.0,   5.0],
    "amazonia":          [-73.0,  -9.0, -44.0,   5.0],
    "europe":            [-31.0,  34.0,  40.0,  72.0],
    "africa":            [-18.0, -35.0,  52.0,  38.0],
    "asia":              [25.0,    0.0, 145.0,  55.0],
    "north america":     [-170.0, 15.0, -50.0,  85.0],
    "south america":     [-82.0, -56.0, -34.0,  13.0],
    "middle east":       [25.0,   12.0,  63.0,  42.0],
}


def resolve_bbox(
    location_text:  str,
    geo_index:      dict,
    country:        Optional[str]  = None,
    country_bboxes: Optional[dict] = None,
) -> tuple[Optional[list], str]:
    """
    Convert a location name into a [W, S, E, N] bounding box.

    Resolution priority (first match wins):
      1. _NAMED_REGION_BBOXES — hardcoded for seas, continents, deserts.
      2. geo_index exact key match — city name → lat/lon → ±BBOX_BUFFER_DEG box.
      3. geo_index partial/substring match — handles "Paris, France" → "paris".
      4. country bbox stored in geo_index.countries[country].bbox.
      5. country bbox aggregated from all city lat/lon records in country_bboxes.
      6. Country is in index but no bbox — returns (None, guidance message).
      7. Not found anywhere — returns (None, not-found message).

    Returns:
        (bbox, note) where bbox is [W, S, E, N] floats or None,
        and note is a human-readable string explaining the source.
    """
    locations = geo_index.get("locations", {}) if geo_index else {}

    if location_text:
        loc_key = normalize_text(location_text)

        # Priority 1 — named geographic regions (seas, continents, etc.) not in geo_index.
        # Try exact key first, then substring overlap for multi-word names.
        region_bbox = _NAMED_REGION_BBOXES.get(loc_key)
        if not region_bbox:
            for rk, rb in _NAMED_REGION_BBOXES.items():
                if rk in loc_key or loc_key in rk:
                    region_bbox = rb
                    break
        if region_bbox:
            return region_bbox, f"named region bbox for '{location_text}'"

        # Priority 2 — exact city match in geo_index.
        rec = locations.get(loc_key)
        if rec:
            lat, lon = rec["lat"], rec["lon"]
            return (
                [
                    round(lon - BBOX_BUFFER_DEG, 4),
                    round(lat - BBOX_BUFFER_DEG, 4),
                    round(lon + BBOX_BUFFER_DEG, 4),
                    round(lat + BBOX_BUFFER_DEG, 4),
                ],
                f"geo_index: {rec['name']} ({lat:.4f}N, {lon:.4f}E) ±{BBOX_BUFFER_DEG}°",
            )

        # Priority 3 — partial/substring city match (handles "toulouse, france", abbreviations).
        for key, rec in locations.items():
            if loc_key in key or key in loc_key:
                lat, lon = rec["lat"], rec["lon"]
                return (
                    [
                        round(lon - BBOX_BUFFER_DEG, 4),
                        round(lat - BBOX_BUFFER_DEG, 4),
                        round(lon + BBOX_BUFFER_DEG, 4),
                        round(lat + BBOX_BUFFER_DEG, 4),
                    ],
                    f"geo_index (partial → {rec['name']})",
                )

    # Priorities 4–6 — country-level bbox fallback.
    country_key = normalize_text(country or location_text or "")
    if country_key:
        countries = geo_index.get("countries", {}) if geo_index else {}
        country_rec = countries.get(country_key, {})

        # Priority 4 — explicit bbox stored in the geo_index country record.
        stored_cb = country_rec.get("bbox") or country_rec.get("bounding_box")
        if stored_cb and len(stored_cb) >= 4:
            return stored_cb, f"country bbox (stored) for '{country_key}'"

        # Priority 5 — bbox aggregated from city points at startup (build_country_bboxes).
        if country_bboxes:
            agg_cb = country_bboxes.get(country_key)
            if agg_cb:
                return (
                    agg_cb,
                    f"country bbox (aggregated from cities) for '{country_key}'"
                    f" [{agg_cb[0]},{agg_cb[1]}→{agg_cb[2]},{agg_cb[3]}]",
                )

        # Priority 6 — country known but no spatial data; return guidance message.
        if country_key in countries:
            return (
                None,
                f"country '{country_key}' is in the index but no bbox is available — "
                "please provide a city, region, or manual bbox (e.g. bbox=W,S,E,N)",
            )

    fallback = location_text or country or ""
    return None, f"'{fallback}' not found in geo_index"


# ── ChromaDB collection search ─────────────────────────────────────────────────
# search_stac_collections() encodes the query as a 384-dim vector and asks ChromaDB
# for the top candidates by cosine similarity.  Location terms are intentionally
# excluded from the query vector — "Toulouse" shouldn't bias which collection type
# is selected; it only matters for the STAC bbox filter later.
# After retrieval, _rerank_hits_for_intent() adjusts scores based on platform,
# spatial extent, and temporal extent metadata stored in ChromaDB.

# Expanded keyword strings for each satellite — enriches the semantic query so
# ChromaDB finds the right mission even when the user just says "S1" or "Sentinel-2".
_SAT_QUERY_EXPANSION = {
    "S1":  "SAR radar Sentinel-1 synthetic aperture radar backscatter flood deforestation",
    "S2":  "optical multispectral Sentinel-2 surface reflectance NDVI vegetation land cover",
    "S3":  "ocean Sentinel-3 sea surface temperature SST chlorophyll marine colour",
    "S5P": "atmosphere Sentinel-5P NO2 nitrogen dioxide methane ozone air quality",
}

# Satellite code → substrings that must appear in the `platforms` ChromaDB metadata field
# for a collection to be considered a "match" for that satellite.
# Used by validate_collection_against_intent() to score platform alignment.
_SAT_PLATFORM_TOKENS = {
    "S1":  ("sentinel-1",),
    "S2":  ("sentinel-2",),
    "S3":  ("sentinel-3",),
    "S5P": ("sentinel-5",),
}


def search_stac_collections(intent: Intent, res: Resources, top_k: int = 5) -> list[dict]:
    """
    Query ChromaDB for the most relevant STAC collections for the given intent.

    Two search strategies:
      A. Exact metadata lookup — when intent.collection_id is already known
         (user named it explicitly, or CMR ID was recognised).  Returns score=1.0.
      B. Semantic vector search — encodes satellite+theme+mission_type keywords as a
         384-dim vector and queries ChromaDB by cosine similarity.
         Location text is EXCLUDED from the query — "Toulouse" should not bias
         which collection type is returned.  Bbox is applied at STAC API call time.

    After retrieval, _rerank_hits_for_intent() adjusts scores:
      platform match    → +0.15   (collection is confirmed for the right satellite)
      platform mismatch → -0.50   (strong penalty: must drop below 0.45 confirm threshold)
      spatial overlap   → +0.10   (collection's extent overlaps the user's bbox)
      spatial no_overlap→ -0.30
      temporal overlap  → +0.10
      temporal no_overlap→ -0.30

    We fetch top_k*5 candidates (min 25) before reranking so the right-platform
    collection is not cut off before scoring — a satellite-specific collection ranked
    at position 10 raw might move to position 1 after +0.15 boost.
    """
    # Strategy A — exact lookup by collection_id (no LLM guessing needed).
    if intent.collection_id:
        try:
            exact = res.stac_col.get(
                where={"collection_id": intent.collection_id},
                limit=1,
                include=["documents", "metadatas"],
            )
            metas = exact.get("metadatas", [])
            docs  = exact.get("documents", [])
            if metas:
                m = metas[0]
                return [{
                    "collection_id": m.get("collection_id", intent.collection_id),
                    "provider_root":  m.get("provider_root", ""),
                    "title":          m.get("title", ""),
                    "score":          1.0,
                    "doc_text":       (docs[0] if docs else "")[:300],
                    "temporal_extent": m.get("extent_temporal", ""),
                    "spatial_extent":  m.get("extent_spatial", ""),
                    "platforms":       m.get("platforms", ""),
                    "keywords":        m.get("keywords", ""),
                }]
        except Exception:
            pass    # fall through to semantic search

    # Strategy B — build a location-free semantic query string.
    # Satellite → expanded synonym string (e.g. "S1" → "SAR radar Sentinel-1 …").
    # Theme → raw keyword from LLM (e.g. "flood", "sea surface temperature SST …").
    # mission_type → sensor class hint (e.g. "optical", "radar").
    parts = []
    if intent.satellite:
        parts.append(_SAT_QUERY_EXPANSION.get(intent.satellite, intent.satellite))
    if intent.theme:
        parts.append(intent.theme)
    if intent.mission_type:
        parts.append(intent.mission_type)
    if not parts:
        parts.append("satellite earth observation data")

    query = " ".join(parts)
    vec = res.embed_model.encode([query]).tolist()

    # Fetch a large candidate pool (5× requested, min 25) before reranking.
    # This ensures correct-platform collections that rank lower in raw cosine
    # similarity can still rise to the top after the +0.15 platform boost.
    results = res.stac_col.query(
        query_embeddings=vec,
        n_results=min(max(top_k * 5, 25), res.stac_col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    docs  = results.get("documents", [[]])[0]

    for i, meta in enumerate(metas):
        # ChromaDB returns L2 or cosine distance; convert to similarity: score = 1 - dist.
        dist = dists[i] if i < len(dists) else 1.0
        doc  = docs[i]  if i < len(docs)  else ""
        hits.append({
            "collection_id":   meta.get("collection_id", ""),
            "provider_root":   meta.get("provider_root", ""),
            "title":           meta.get("title", ""),
            "score":           round(1 - dist, 3),
            "doc_text":        doc[:300] if doc else "",
            "temporal_extent": meta.get("extent_temporal", ""),
            "spatial_extent":  meta.get("extent_spatial", ""),
            "platforms":       meta.get("platforms", ""),
            "keywords":        meta.get("keywords", ""),
        })

    # Rerank and return only the top_k after scoring adjustments.
    hits = _rerank_hits_for_intent(hits, intent)
    return hits[:top_k]


# ── Validation scaffolding (spatial / temporal / platform) ────────────────────
#
# validate_collection_against_intent() checks each ChromaDB hit against the
# intent's bbox, date range, and satellite code.
#
# Possible status values per dimension:
#   "overlaps"   → confirmed match   → score boosted
#   "no_overlap" → confirmed miss    → score penalised
#   "unknown"    → no metadata       → score unchanged (don't penalise missing data)
#
# NOTE: current indexed metadata may lack extents — extent_spatial and
# extent_temporal are often empty strings because the initial crawl did not
# store them.  Platform IS reliably stored.  Spatial/temporal validation becomes
# active automatically once the crawler stores extent.spatial.bbox and
# extent.temporal.interval in the ChromaDB records.


def parse_spatial_extent(meta: dict) -> list:
    """
    Parse 'extent_spatial' from ChromaDB metadata → list of [W, S, E, N] bboxes.
    Returns [] when the field is absent, empty, or unparseable.
    """
    raw = (meta.get("extent_spatial") or meta.get("spatial_extent") or "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list) and val:
            # [[W, S, E, N], ...] or [W, S, E, N]
            if isinstance(val[0], list):
                return [b for b in val if len(b) >= 4]
            if isinstance(val[0], (int, float)) and len(val) >= 4:
                return [val[:4]]
    except (json.JSONDecodeError, TypeError, IndexError):
        pass
    return []


def bbox_intersects(bbox1: list, bbox2: list) -> bool:
    """Return True if two [W, S, E, N] bboxes share any area."""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return False
    w1, s1, e1, n1 = bbox1[:4]
    w2, s2, e2, n2 = bbox2[:4]
    return not (e1 < w2 or e2 < w1 or n1 < s2 or n2 < s1)


def parse_temporal_extent(meta: dict) -> list:
    """
    Parse 'extent_temporal' from ChromaDB metadata → list of (start, end) tuples.
    Returns [] when the field is absent, empty, or unparseable.
    """
    raw = (meta.get("extent_temporal") or meta.get("temporal_extent") or "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list) and val:
            # [["2015-01-01T00:00:00Z", null], ...] or ["2015-01-01T00:00:00Z", null]
            if isinstance(val[0], list):
                return [(iv[0], iv[1]) for iv in val if len(iv) >= 2]
            if len(val) >= 2 and (val[0] is None or isinstance(val[0], str)):
                return [(val[0], val[1])]
    except (json.JSONDecodeError, TypeError, IndexError):
        pass
    return []


def date_overlaps(
    date_start: Optional[str],
    date_end:   Optional[str],
    interval:   tuple,
) -> bool:
    """
    Return True if the intent date range overlaps with a collection interval.
    Only returns False on a clear, unambiguous non-overlap.
    """
    iv_start, iv_end = interval

    def _dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.fromisoformat(s[:10])
            except ValueError:
                return None

    q_start = _dt(date_start)
    q_end   = _dt(date_end)
    c_start = _dt(iv_start)
    c_end   = _dt(iv_end)

    # Fully open ranges — assume overlap
    if (q_start is None and q_end is None) or (c_start is None and c_end is None):
        return True

    if q_end and c_start and q_end.replace(tzinfo=None) < c_start.replace(tzinfo=None):
        return False
    if q_start and c_end and q_start.replace(tzinfo=None) > c_end.replace(tzinfo=None):
        return False
    return True


def validate_collection_against_intent(
    hit:        dict,
    intent:     Intent,
    query_bbox: Optional[list] = None,
) -> dict:
    """
    Validate a ChromaDB hit against the user's spatial/temporal/satellite intent.

    Returns:
        {"spatial": "overlaps"|"no_overlap"|"unknown",
         "temporal":"overlaps"|"no_overlap"|"unknown",
         "platform":"match"   |"mismatch"  |"unknown"}

    Collections are NOT dropped when status is "unknown" — only boosted/penalised
    when status is known.  Because current indexed metadata commonly has empty
    extent fields, most results will be "unknown" until the crawler is updated.
    """
    result: dict = {"spatial": "unknown", "temporal": "unknown", "platform": "unknown"}

    # Spatial
    spatial_extents = parse_spatial_extent(hit)
    if spatial_extents and query_bbox and len(query_bbox) >= 4:
        result["spatial"] = (
            "overlaps" if any(bbox_intersects(query_bbox, e) for e in spatial_extents)
            else "no_overlap"
        )

    # Temporal
    temporal_extents = parse_temporal_extent(hit)
    if temporal_extents and (intent.date_start or intent.date_end):
        result["temporal"] = (
            "overlaps" if any(
                date_overlaps(intent.date_start, intent.date_end, iv)
                for iv in temporal_extents
            )
            else "no_overlap"
        )

    # Platform (this field IS populated in current metadata)
    plat = (hit.get("platforms") or "").lower()
    if plat and intent.satellite:
        wanted    = _SAT_PLATFORM_TOKENS.get(intent.satellite, ())
        all_sats  = {tok for toks in _SAT_PLATFORM_TOKENS.values() for tok in toks}
        if any(tok in plat for tok in wanted):
            result["platform"] = "match"
        elif any(tok in plat for tok in all_sats):
            result["platform"] = "mismatch"

    return result


def _rerank_hits_for_intent(
    hits:       list[dict],
    intent:     Intent,
    query_bbox: Optional[list] = None,
) -> list[dict]:
    """
    Rerank ChromaDB hits by adjusting their cosine similarity scores based on
    spatial, temporal, and platform validation results.

    Score adjustments (cumulative — all three dimensions are checked independently):
      spatial  overlaps    → +0.10
      spatial  no_overlap  → −0.30
      temporal overlaps    → +0.10
      temporal no_overlap  → −0.30
      platform match       → +0.15
      platform mismatch    → −0.50  (strong: forces score below 0.45 confirm threshold)
      "unknown" in any     →  0.00  (missing metadata must not penalise a collection)

    The −0.50 platform penalty is intentionally large: a wrong-satellite collection
    (e.g. S2 when the user asked for S1) should rank below the 0.45 confirmation
    threshold so the user is always asked to confirm it — or it is filtered out
    entirely by the non_mismatch filter in _handle_item_search().

    Scores are capped at 1.0 and written back into h["score"] so the display
    matches the sorted order.  The _validation dict is stored in h for verbose output.
    """
    scored = []
    for h in hits:
        v     = validate_collection_against_intent(h, intent, query_bbox)
        h["_validation"] = v   # retained so verbose mode can print spatial/temporal/platform
        score = h["score"]

        if v["spatial"] == "overlaps":
            score += 0.10
        elif v["spatial"] == "no_overlap":
            score -= 0.30

        if v["temporal"] == "overlaps":
            score += 0.10
        elif v["temporal"] == "no_overlap":
            score -= 0.30

        if v["platform"] == "match":
            score += 0.15
        elif v["platform"] == "mismatch":
            score -= 0.50

        h["score"] = round(min(score, 1.0), 3)   # cap at 1.0; write back for display
        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


# ── Display functions ──────────────────────────────────────────────────────────
# These functions print formatted results to the terminal.
# They never call any API — all data comes from ChromaDB hits or SearchResult objects.

_W = "═" * 64   # horizontal separator line used in all display blocks


def display_collection_cards(
    hits: list[dict], query_used: str, providers=None
) -> None:
    """
    Print one card per ChromaDB hit (Mode A output).
    Shows collection ID, relevance score, provider, collection URL, and description snippet.
    The search_url is looked up from the ProviderIndex (stac_providers.jsonl) so each
    provider gets its correct /search endpoint rather than a generic guess.
    """
    print(f"\n{_W}")
    print(f"  STAC COLLECTIONS  —  {len(hits)} results")
    print(f"  Query: {query_used}")
    print(_W)

    for i, h in enumerate(hits, 1):
        cid   = h["collection_id"] or "?"
        prov  = h["provider_root"].replace("https://", "").rstrip("/")[:50]
        title = (h["title"] or "")[:65]
        text  = (h["doc_text"] or "").replace("\n", " ")[:220]

        col_url    = f"{h['provider_root'].rstrip('/')}/collections/{h['collection_id']}"
        # Fix 3: use ProviderIndex lookup; falls back to /search if not known
        if providers is not None:
            search_url = providers.search_url(h["provider_root"])
        else:
            search_url = f"{h['provider_root'].rstrip('/')}/search"

        print(f"\n  [{i}]  {cid}")
        print(f"       Score      : {h['score']:.3f}")
        if title:
            print(f"       Title      : {title}")
        print(f"       Provider   : {prov}")
        col_url_clean = col_url
        print(f"       Collection : {col_url_clean}")
        print(f"       Search API : {search_url}")
        if text:
            print(f"       About      : {text}")

    print(f"\n{_W}")


def display_item_cards(
    result: SearchResult,
    intent: Intent,
    page_start: int = 1,
) -> None:
    """
    Print one card per satellite item from a live STAC /search response (Mode B output).
    page_start counts items across pagination (e.g. page 2 starts at 11).
    Only fields present in the item are shown — fields like cloud_cover and bbox
    are optional in the STAC spec and many providers omit them.
    Ends with 'more' / 'export' prompts when more pages or items exist.
    """
    cid  = result.collection_id
    prov = result.provider_root.replace("https://", "").rstrip("/")[:50]
    loc  = intent.location_text or intent.country or ""
    period = ""
    if intent.date_start:
        period = f"{intent.date_start} → {intent.date_end or '…'}"

    print(f"\n{_W}")
    print(f"  STAC PRODUCTS  —  {cid}")
    if loc:
        print(f"  Location  : {loc}")
    if period:
        print(f"  Period    : {period}")
    print(f"  Provider  : {prov}")

    if result.total_matched is not None:
        print(f"  Total     : {result.total_matched:,} items matched  (real count from provider)")
    else:
        print("  Total     : unknown — provider does not expose count")

    end_idx = page_start + result.returned - 1
    print(f"  Showing   : items {page_start}–{end_idx}")
    print(_W)

    _PAGE_DISPLAY = 5
    items_to_show = result.items[:_PAGE_DISPLAY]
    hidden        = len(result.items) - len(items_to_show)

    for i, item in enumerate(items_to_show, page_start):
        lines = [f"\n  [{i}] {item.item_id}"]

        dt = item.datetime or item.start_datetime
        if dt:
            _this_year = datetime.now(timezone.utc).year
            _future_flag = ""
            try:
                if int(dt[:4]) > _this_year + 1:
                    _future_flag = "  [!] provider date looks wrong"
            except (ValueError, IndexError):
                pass
            lines.append(f"      Date       : {dt[:19].replace('T', ' ')} UTC{_future_flag}")
        if item.end_datetime and item.start_datetime and item.end_datetime != item.start_datetime:
            lines.append(f"      Period     : {item.start_datetime[:10]} → {item.end_datetime[:10]}")
        if item.cloud_cover is not None:
            lines.append(f"      Cloud cover: {item.cloud_cover}%")
        if item.platform:
            lines.append(f"      Platform   : {item.platform}")
        if item.bbox and len(item.bbox) >= 4:
            w2, s2, e2, n2 = item.bbox[:4]
            lines.append(f"      BBox       : W={w2:.4f} S={s2:.4f} E={e2:.4f} N={n2:.4f}")
        if item.data_hrefs:
            keys = list(item.data_hrefs.keys())
            lines.append(f"      Data assets: {', '.join(keys[:8])}")
            for k, href in list(item.data_hrefs.items())[:2]:
                lines.append(f"        {k}: {href}")
        elif item.all_assets:
            lines.append(f"      Assets     : {', '.join(list(item.all_assets.keys())[:8])}")
        if item.thumbnail_href:
            lines.append(f"      Thumbnail  : {item.thumbnail_href}")

        print("\n".join(lines))

    print(f"\n{_W}")
    prompts = []
    if hidden:
        prompts.append(f"'all items' to see {hidden} more on this page")
    if result.has_more:
        prompts.append("'more' for next page")
    if result.total_matched or result.has_more:
        prompts.append("'export' to save all")
    if prompts:
        print(f"  Type: {' | '.join(prompts)}")


def display_single_item(item: ParsedItem) -> None:
    """
    Print full detail for a single item fetched by ID (item_by_id request_type).
    Shows all available metadata including all asset URLs and raw properties.
    """
    print(f"\n{_W}")
    print(f"  STAC ITEM  —  {item.item_id}")
    print(_W)
    print(f"\n  Collection  : {item.collection_id}")
    dt = item.datetime or item.start_datetime
    if dt:
        print(f"  Date        : {dt[:19].replace('T', ' ')} UTC")
    if item.cloud_cover is not None:
        print(f"  Cloud cover : {item.cloud_cover}%")
    if item.platform:
        print(f"  Platform    : {item.platform}")
    if item.instruments:
        print(f"  Instruments : {', '.join(item.instruments)}")
    if item.bbox and len(item.bbox) >= 4:
        w2, s2, e2, n2 = item.bbox[:4]
        print(f"  BBox        : W={w2:.4f} S={s2:.4f} E={e2:.4f} N={n2:.4f}")
    if item.data_hrefs:
        print(f"\n  Data assets ({len(item.data_hrefs)}):")
        for k, href in item.data_hrefs.items():
            print(f"    {k}: {href}")
    elif item.all_assets:
        print(f"\n  All assets ({len(item.all_assets)}):")
        for k, a in list(item.all_assets.items())[:10]:
            print(f"    {k}: {a.get('href', '')[:100]}")
    if item.thumbnail_href:
        print(f"\n  Thumbnail   : {item.thumbnail_href}")
    if item.properties:
        print(f"\n  Properties  : {json.dumps(item.properties, ensure_ascii=False, default=str)[:400]}")
    print(f"\n{_W}")


# ── LLM helper (single call-site for easy backend swap) ───────────────────────

def _llm_generate(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> str:
    """
    Single wrapper around ollama.chat() used by all synthesis and meta-question calls.
    Keeping this centralised makes it easy to swap Ollama for another backend later.
    temperature=0.1 for synthesis (slight creativity), 0.2 for meta-questions,
    0.0 for intent extraction (maximum determinism).
    """
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp.message.content.strip()


# ── Synthesis (Ollama) ─────────────────────────────────────────────────────────
# After a successful search the results are passed to a synthesis function which
# calls the LLM to write a short natural-language summary.
# System prompts are strict "only describe what you see" instructions so the LLM
# cannot hallucinate IDs, dates, or URLs that aren't in the data.

# System prompt for Mode A (collection cards).
_SYS_COLLECTION = (
    "You are a satellite Earth Observation assistant. "
    "Describe only what is in the provided list. "
    "Never invent collection IDs, provider names, or capabilities not listed."
)

# System prompt for Mode B (live STAC items).
_SYS_ITEMS = (
    "You are a satellite data assistant. "
    "Report only what the data shows. "
    "Never invent item IDs, dates, URLs, counts, or cloud cover values."
)


def _history_block(source, max_turns: int = 4) -> str:
    """
    Build a "Conversation so far:" prefix for synthesis prompts.

    `source` can be a MemoryManager (preferred — uses the active memory strategy)
    or a plain list[dict] (legacy fallback for callers that still pass state["history"]).
    """
    if isinstance(source, MemoryManager):
        ctx = source.get_context()
        return (f"Conversation so far:\n{ctx}\n\n") if ctx else ""
    # legacy: list[dict]
    lines = []
    for msg in (source or [])[-max_turns:]:
        if msg.get("role") in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
    return ("Conversation so far:\n" + "\n".join(lines) + "\n\n") if lines else ""


def synthesize_collection_results(
    intent: Intent, hits: list[dict], model: str,
    history: "MemoryManager | list[dict] | None" = None,
) -> str:
    """
    Ask the LLM to write a 3–5 sentence summary of the ChromaDB collection hits.
    Context passed: collection IDs, providers, titles, and relevance scores.
    The LLM is instructed to only describe what is listed — no invention.
    """
    if not hits:
        return "No matching collections found."

    context = "\n".join(
        f"- {h['collection_id']} | {h['provider_root']} | {h['title']} | score={h['score']:.3f}"
        for h in hits
    )
    topic = " + ".join(
        filter(None, [intent.satellite, intent.theme, intent.mission_type])
    ) or "satellite data"

    location = " + ".join(filter(None, [intent.location_text, intent.country])) or None
    period   = (f"{intent.date_start} to {intent.date_end}"
                if intent.date_start else None)
    context_line = " | ".join(filter(None, [location, period]))

    hist = _history_block(history) if history else ""
    prompt = (
        f"{hist}"
        f"User wants: {topic}"
        + (f" — location: {context_line}" if context_line else "") + "\n\n"
        f"Collections found:\n{context}\n\n"
        "Give a direct, practical answer in 2-3 sentences:\n"
        "1. Which collection is the BEST match and why.\n"
        "2. What type of data it provides (sensor, resolution if known).\n"
        "3. One concrete next step (e.g. 'Type 1 to search it, or 2 for the radar option').\n"
        "Be specific. Only describe the listed collections. No generic filler."
    )
    try:
        return _llm_generate(_SYS_COLLECTION, prompt, model, temperature=0.2, max_tokens=280)
    except Exception as exc:
        return f"[synthesis error: {exc}]"


def synthesize_item_results(
    intent: Intent, result: SearchResult, model: str,
    history: "MemoryManager | list[dict] | None" = None,
) -> str:
    """
    Ask the LLM to write a 2–4 sentence summary of the live STAC item results.
    Passes up to 6 sample items (ID, date, cloud cover, platform) as structured text.
    The "Describe ONLY the current search results" instruction prevents the LLM from
    referencing the previous query even when conversation history is prepended.
    Ends with a mandatory "(Source: collection_id @ provider)" citation line.
    """
    if not result.items:
        return "No products found for the given criteria."

    summaries = []
    for item in result.items[:6]:
        parts = [item.item_id]
        dt = item.datetime or item.start_datetime
        if dt:
            parts.append(f"date={dt[:10]}")
        if item.cloud_cover is not None:
            parts.append(f"cloud={item.cloud_cover}%")
        if item.platform:
            parts.append(f"platform={item.platform}")
        summaries.append(" | ".join(parts))

    if result.total_matched is not None:
        count_line = f"Total matched in provider database: {result.total_matched:,} items"
    else:
        count_line = (
            f"Provider does not expose a total count. "
            f"Returned {result.returned} items this page."
        )

    hist = _history_block(history) if history else ""
    prompt = (
        f"{hist}"
        f"Provider: {result.provider_root}\n"
        f"Collection: {result.collection_id}\n"
        f"{count_line}\n"
        f"Sample items (first {len(summaries)}):\n"
        + "\n".join(f"  {s}" for s in summaries)
        + "\n\nDescribe ONLY the current search results listed above (not the conversation history). "
        "In 2-4 sentences: state what was found — count, platform, date range, "
        "cloud cover range if available. Never invent IDs, dates, or URLs. "
        f"End your response with exactly: "
        f"(Source: {result.collection_id} @ "
        f"{result.provider_root.replace('https://','').rstrip('/')})"
    )
    try:
        return _llm_generate(_SYS_ITEMS, prompt, model, temperature=0.1, max_tokens=220)
    except Exception as exc:
        return f"[synthesis error: {exc}]"


# ── Mode handlers ──────────────────────────────────────────────────────────────
# Each handler corresponds to one branch of the routing table in run_chat().
# They read from `state` (the mutable session dict) and write results back to it.
# The `state` dict carries: model, top_k, verbose, history, last search/intent,
# last collection hits, pagination cursors, export paths, and raw user input.

# System prompt for the meta-question handler.
# Intentionally conservative: only answer from what was already shown, never search.
_SYS_META = (
    "You are a satellite data assistant answering a follow-up question about results "
    "that were already shown. Answer concisely using only the provided context. "
    "Never invent new data, IDs, or URLs."
)


def _handle_meta_question(user_input: str, state: dict) -> None:
    """
    Answer a conversational follow-up about prior results without triggering a new search.
    Detected by _META_Q_RE: "why so many?", "how do these compare?", "explain this", etc.

    Builds a context block from:
      - last_search_result: collection, provider, total matched, sample item dates/cloud
      - last_collection_hits: collection IDs and titles from the last Mode A search
    Then calls the LLM with that context + conversation history to answer the question.
    """
    ctx_parts: list[str] = []

    result = state.get("last_search_result")
    if result:
        ctx_parts.append(
            f"Last item search: {result.collection_id} @ "
            f"{result.provider_root.replace('https://','').rstrip('/')}\n"
            f"  Total matched: {result.total_matched or 'unknown'}\n"
            f"  Returned: {result.returned} items\n"
            f"  Items: " + ", ".join(
                (it.datetime or "")[:10] + f" cloud={it.cloud_cover}%"
                if it.cloud_cover is not None else (it.datetime or "")[:10]
                for it in result.items[:5]
            )
        )

    hits = state.get("last_collection_hits")
    if hits and not result:
        ctx_parts.append(
            "Last collection search results:\n" + "\n".join(
                f"  {h['collection_id']} | {h['title']} | score={h['score']:.3f}"
                for h in hits[:3]
            )
        )

    hist = _history_block(state.get("history", []), max_turns=4)
    context = "\n\n".join(ctx_parts) if ctx_parts else "No prior results available."

    prompt = (
        f"{hist}"
        f"Context from previous results:\n{context}\n\n"
        f"User follow-up question: {user_input}"
    )
    print("  [thinking …]", end="\r", flush=True)
    try:
        answer = _llm_generate(_SYS_META, prompt, state["model"], temperature=0.2, max_tokens=200)
    except Exception as exc:
        answer = f"[error: {exc}]"
    print("               ", end="\r")
    print(f"\nAssistant: {answer}\n")


def _handle_collection_discovery(intent: Intent, res: Resources, state: dict) -> None:
    """
    Mode A handler: search ChromaDB and show collection cards.

    Steps:
      1. Call search_stac_collections() → ChromaDB cosine search + rerank.
      2. display_collection_cards() → print one card per hit with provider URLs.
      3. synthesize_collection_results() → LLM writes a 3–5 sentence summary.
      4. Save hits to state["last_collection_hits"] so follow-up queries
         ("search the second one", "other 3") can reference them.
    """
    print("  [searching stac_collections …]", end="\r", flush=True)
    hits = search_stac_collections(intent, res, top_k=state["top_k"])
    print("                                  ", end="\r")

    if not hits:
        print("\nAssistant: No matching collections found. Try rephrasing your topic.\n")
        return

    parts = list(filter(None, [
        intent.satellite, intent.theme, intent.mission_type,
    ]))
    query_used = " + ".join(parts) if parts else "satellite data"

    display_collection_cards(hits, query_used, providers=res.stac_searcher.providers)

    print("  [synthesizing …]", end="\r", flush=True)
    answer = synthesize_collection_results(intent, hits, state["model"], state["memory"])
    print("                   ", end="\r")
    print(f"\nAssistant: {answer}\n")

    state["last_collection_hits"] = hits
    state["last_intent"]          = intent


def _handle_item_by_id(intent: Intent, res: Resources, state: dict) -> None:
    """
    Fetch and display a single STAC item by its explicit ID.
    Requires both item_id and collection_id to be known.
    If provider_root is missing, resolves it from ChromaDB first.
    """
    if not intent.item_id:
        print("\nAssistant: No item ID specified.\n")
        return

    # Need collection_id + provider_root to fetch item
    if not intent.collection_id:
        print(
            f"\nAssistant: Please also specify the collection ID for item '{intent.item_id}'.\n"
            "  Example: 'fetch item S2B_32TLQ_20250901_0_L2A from sentinel-2-l2a'\n"
        )
        return

    # Resolve provider_root from ChromaDB if not already set
    provider_root = intent.provider_root
    if not provider_root:
        print("  [resolving provider …]", end="\r", flush=True)
        hits = search_stac_collections(intent, res, top_k=1)
        print("                        ", end="\r")
        if hits:
            provider_root = hits[0]["provider_root"]
        else:
            print(
                f"\nAssistant: Cannot resolve provider for collection '{intent.collection_id}'.\n"
            )
            return

    print(f"  [fetching item {intent.item_id[:50]} …]", end="\r", flush=True)
    item = res.stac_searcher.get_item(provider_root, intent.collection_id, intent.item_id)
    print("                                                   ", end="\r")

    if item is None:
        print(
            f"\nAssistant: Item '{intent.item_id}' not found in "
            f"'{intent.collection_id}' at {provider_root}.\n"
        )
        return

    display_single_item(item)
    print(f"\nAssistant: Retrieved item '{item.item_id}' from {provider_root}.\n")


def _handle_item_search(intent: Intent, res: Resources, state: dict) -> None:
    """
    Mode B handler: find a collection, build a STAC /search payload, call the
    provider API, display real satellite items, and synthesize a summary.

    Seven logical steps:
      Step 1  Collection resolution — decide WHICH collection to search.
              Priority: explicit ref ("second one") > "same as last" > new ChromaDB search.
              Sets using_last=True when reusing a collection from the previous turn.

      Step 2a Pre-resolve bbox for reranking (before confirmation prompt).
      Step 2  Final bbox resolution — city/country/region → [W, S, E, N].
              Also inherits bbox/dates from last_intent when using_last=True.

      Step 3  Clarification — if no bbox AND no date, ask the user once (max 1 time).

      Step 4  Count-only — if request_type=="item_count", call .count() and return.
              No item cards are shown.

      Step 5  CQL2 cloud-cover filter — use server-side CQL2 if the provider supports it,
              otherwise post-filter client-side after fetching more items.

      Step 6  Item search loop — try up to 3 candidate collections in order.
              Stop at first collection that returns items (or display all if search_all=True).
              Auth errors (401/403) are surfaced clearly instead of silently ignored.

      Step 7  Display + synthesize — show item cards and ask LLM for summary.
              Update state for 'more', 'export', and next-turn context.
    """
    last_hits   = state.get("last_collection_hits") or []
    interactive = state.get("_interactive", True)

    # Short-circuit: item_by_id is a different handler entirely.
    if intent.request_type == "item_by_id" or intent.item_id:
        _handle_item_by_id(intent, res, state)
        return

    # ── Step 1: collection resolution ──────────────────────────────────────────
    # Decide which collection(s) to search.
    # using_last=True means we reuse last_collection_hits without a new ChromaDB query.
    # This is set when:
    #   a) The LLM's reasoning text mentions reference phrases ("from the list", etc.)
    #   b) The raw input matches _SAME_AS_LAST_RE ("the same", "that one", etc.)
    #   c) _resolve_collection_refs_list() matched an explicit ordinal/numeric reference.
    #   d) No topic is specified at all (user only added location/date to a prior query).
    has_topic = bool(intent.satellite or intent.theme or intent.mission_type)

    _REF_PHRASES = ("these collection", "those collection", "from these", "from those",
                    "from the list", "same collection", "those results", "these results")
    using_last = any(p in (intent.reasoning or "").lower() for p in _REF_PHRASES)

    # Fix 14: multi-ref resolver (all these / other N / second one / etc.)
    raw_input      = state.get("_raw_input", "")
    last_tried_id  = state.get("_last_used_collection_id")

    # Fix 16: "the same / same but / that one" → reuse last collection without re-confirming
    if not using_last and raw_input and last_hits and _SAME_AS_LAST_RE.search(raw_input):
        using_last = True

    ref_list: Optional[list] = None
    if last_hits and raw_input:
        ref_list = _resolve_collection_refs_list(raw_input, last_hits, last_tried_id)

    if ref_list:
        hits       = ref_list
        using_last = True
    elif not has_topic and last_hits:
        hits       = last_hits
        using_last = True
    elif using_last and last_hits:
        hits = last_hits
    else:
        hits = _fresh_search_or_last(intent, res, last_hits)
        using_last = hits is last_hits
        # Keep last_collection_hits current so "other N" in the next turn refers
        # to the collections that were actually searched.
        if not using_last and hits:
            state["last_collection_hits"] = hits

    if not hits:
        print(
            "\nAssistant: I couldn't find a matching collection for your query.\n"
            "  Try 'What data is available for [topic]?' to explore first.\n"
        )
        return

    # ── Step 2a: resolve bbox early so validation can use it ──────────────────
    # (We resolve again below if needed; this pre-resolution is for reranking only)
    pre_bbox: Optional[list] = intent.bbox
    if not pre_bbox and (intent.location_text or intent.country):
        pre_bbox, _ = resolve_bbox(
            location_text  = intent.location_text,
            geo_index      = res.geo_index,
            country        = intent.country,
            country_bboxes = res.country_bboxes,
        )

    # Rerank with full validation now that bbox is known
    hits = _rerank_hits_for_intent(hits, intent, query_bbox=pre_bbox)

    # ── Step 2: resolve bbox (final, authoritative) ────────────────────────────
    bbox, bbox_note = None, ""
    if intent.bbox:
        bbox      = intent.bbox
        bbox_note = "manual bbox from user input"
    elif intent.location_text or intent.country:
        bbox, bbox_note = resolve_bbox(
            location_text  = intent.location_text,
            geo_index      = res.geo_index,
            country        = intent.country,
            country_bboxes = res.country_bboxes,
        )

    if state["verbose"] and bbox_note:
        print(f"  [geo] {bbox_note}")

    # Surface the "no bbox" guidance message from resolve_bbox when country is known
    if not bbox and bbox_note and "please provide" in bbox_note:
        print(f"\n  Note: {bbox_note}\n")

    # ── Step 3: clarification ──────────────────────────────────────────────────
    # Inherit bbox/dates from last_intent when reusing a collection ("the same but …").
    last_intent = state.get("last_intent")
    if using_last and last_intent:
        if not bbox and last_intent.bbox:
            bbox = last_intent.bbox
        if not intent.date_start and last_intent.date_start:
            intent.date_start = last_intent.date_start
            intent.date_end   = last_intent.date_end

    # Clarification fires when EITHER bbox OR date is missing (not just both absent).
    # We save the current intent so the next turn can restore topic context automatically.
    if (not bbox or not intent.date_start) and state["clarification_count"] < MAX_CLARIFICATIONS:
        missing = []
        if not bbox:
            missing.append("a location (e.g. 'over Toulouse', 'over Europe')")
        if not intent.date_start:
            missing.append("a time period (e.g. 'in 2023', 'summer 2025')")
        print(
            f"\nAssistant: For product search I need {' and '.join(missing)}.\n"
        )
        state["_pending_intent"] = intent   # topic preserved for next turn
        state["clarification_count"] += 1
        return

    state["clarification_count"] = 0
    state.pop("_pending_intent", None)   # clarification satisfied — clear saved state
    datetime_range = _datetime_range(intent)

    # ── Confirmation logic ─────────────────────────────────────────────────────
    # We ask the user to confirm the top collection when the score is low or ambiguous.
    # This prevents silently searching the wrong collection when the top result is uncertain.
    best = hits[0]
    # score_gap: difference between #1 and #2 — small gap means two collections are nearly tied.
    score_gap = (hits[0]["score"] - hits[1]["score"]) if len(hits) >= 2 else 1.0
    # most_unknown: if 2+ of the top 3 have no spatial/temporal metadata,
    # we can't trust the ranking and should be conservative.
    most_unknown = (
        sum(1 for h in hits[:3]
            if h.get("_validation", {}).get("spatial")  == "unknown"
            and h.get("_validation", {}).get("temporal") == "unknown")
        >= 2
    )

    # Remove confirmed-mismatch collections from the candidate list when user named a satellite.
    # This ensures a wrong-satellite collection is never silently used even after reranking.
    if intent.satellite and not using_last:
        non_mismatch = [h for h in hits if h.get("_validation", {}).get("platform") != "mismatch"]
        if non_mismatch:
            hits = non_mismatch
            best = hits[0]
            score_gap = (hits[0]["score"] - hits[1]["score"]) if len(hits) >= 2 else 1.0

    platform_confirmed = best.get("_validation", {}).get("platform") == "match"
    platform_mismatch  = best.get("_validation", {}).get("platform") == "mismatch"

    # needs_confirm is True when all of these hold:
    #   - interactive session (not --ask mode)
    #   - not explicitly reusing last collection (using_last=False)
    #   - satellite is not confirmed as a platform match
    # AND at least one of:
    #   - score < 0.45 (too low to be confident)
    #   - score_gap ≤ 0.03 (top two collections nearly tied)
    #   - most metadata unknown AND score < 0.60 (can't rank reliably)
    #   - named satellite but platform is a confirmed mismatch
    needs_confirm = (
        interactive
        and not using_last
        and not (intent.satellite and platform_confirmed)
        and (
            best["score"] < 0.45
            or score_gap <= 0.03
            or (most_unknown and best["score"] < 0.60)
            or (intent.satellite and platform_mismatch)
        )
    )
    # "all these" in the raw input means the user explicitly named all candidates
    # → force search_all and skip the interactive confirmation prompt.
    search_all = bool(ref_list and _ALL_THESE_RE.search(raw_input))
    if needs_confirm and not search_all:
        print(f"\n  Similar collections found:")
        for i, h in enumerate(hits[:3], 1):
            v = h.get("_validation", {})
            print(f"    [{i}] {h['collection_id']}")
            print(f"         provider : {h['provider_root'].replace('https://','')[:50]}")
            print(f"         score    : {h['score']:.3f}")
            print(f"         spatial  : {v.get('spatial','unknown')}  "
                  f"temporal: {v.get('temporal','unknown')}  "
                  f"platform: {v.get('platform','unknown')}")
        try:
            choice = input(
                "  Proceed with [1]? Enter 1/2/3, a collection_id, 'all', or Enter: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""
        if choice == "all":
            search_all = True   # search every candidate, display each separately
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(hits):
                hits = [hits[idx]] + [h for j, h in enumerate(hits) if j != idx]
                best = hits[0]
        elif choice:
            # try to match a collection_id substring
            matched = [h for h in hits if choice in h["collection_id"].lower()]
            if matched:
                hits = matched + [h for h in hits if h not in matched]
                best = hits[0]
        state["pending_candidates"] = hits[:3]

    collection_id = best["collection_id"]
    provider_root = best["provider_root"]

    if state["verbose"]:
        src = "last_hits" if using_last else "chromadb"
        print(f"  [collection/{src}] {collection_id} @ {provider_root} "
              f"(score={best['score']:.3f})")
        # Fix 2 verbose: print validation notes for each candidate
        for i, h in enumerate(hits[:3], 1):
            v = h.get("_validation", {})
            print(f"    [{i}] {h['collection_id']} | "
                  f"spatial={v.get('spatial','?')} "
                  f"temporal={v.get('temporal','?')} "
                  f"platform={v.get('platform','?')}")

    # ── Step 4: count-only ─────────────────────────────────────────────────────
    # "How many Sentinel-2 scenes over Paris in 2023?" → just the number, no cards.
    if intent.request_type == "item_count":
        print("  [counting items …]", end="\r", flush=True)
        try:
            count = res.stac_searcher.count(
                provider_root, collection_id, bbox, datetime_range
            )
        except RuntimeError as exc:
            print(f"\nAssistant: Could not reach the search endpoint — {exc}\n")
            return
        print("                    ", end="\r")

        if count is not None:
            print(
                f"\nAssistant: There are {count:,} real items in '{collection_id}' "
                f"for the given criteria (source: live STAC /search).\n"
            )
        else:
            print(
                f"\nAssistant: The provider ({provider_root.replace('https://', '')}) "
                f"does not expose a total count for '{collection_id}'.\n"
                "  Type 'show me items' to retrieve the first page of results.\n"
            )
        return

    # ── Step 5: build CQL2 cloud cover filter if supported ────────────────────
    # Providers that support CQL2 (e.g. Microsoft Planetary Computer) accept a
    # server-side filter in the POST body: {"op": "<=", "args": [{"property": "eo:cloud_cover"}, N]}
    # Providers that don't support CQL2 (e.g. Copernicus Data Space) can't filter server-side,
    # so we fetch 5× more items and drop those that exceed the threshold after the fact.
    cql2_filter   = None
    local_cc_max  = None
    if intent.cloud_cover_max is not None:
        if res.stac_searcher.providers.supports_cql2(provider_root):
            cql2_filter = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, intent.cloud_cover_max],
            }
        else:
            local_cc_max = intent.cloud_cover_max   # applied after fetching

    # ── Step 6: item search ───────────────────────────────────────────────────
    # Normal mode: iterate candidates in score order; stop at the first one that
    # returns at least 1 item (so the user always gets something).
    # search_all mode (user typed "all" at confirmation): search every candidate
    # and print each result block inline — useful when comparing collections.
    candidates  = [best] + [h for h in hits[1:] if h["collection_id"] != collection_id]
    result      = None
    tried       = []
    all_results = []   # (collection_id, returned_count, SearchResult, provider_root) for recap+export

    for candidate in candidates[:3]:
        c_cid  = candidate["collection_id"]
        c_prov = candidate["provider_root"]
        prov_s = c_prov.replace("https://", "").rstrip("/")[:40]
        print(f"  [querying {prov_s}/{c_cid[:25]} …]", end="\r", flush=True)
        tried.append(c_cid)

        limit = ITEM_PAGE_SIZE
        if local_cc_max is not None:
            limit = min(ITEM_PAGE_SIZE * 5, 50)    # fetch more to post-filter

        try:
            r = res.stac_searcher.search(
                provider_root  = c_prov,
                collection_id  = c_cid,
                bbox           = bbox,
                datetime_range = datetime_range,
                filters        = cql2_filter,
                limit          = limit,
            )
        except RuntimeError as exc:
            err_str = str(exc)
            is_auth = "401" in err_str or "403" in err_str
            print(f"                                                            ", end="\r")
            if is_auth:
                print(f"\n  [{c_cid}] — authentication required (HTTP 401/403). "
                      f"This provider needs a token.\n")
            elif search_all:
                print(f"\n  [{c_cid}] — unreachable ({exc})\n")
            else:
                print(f"\n  Note: could not reach {c_cid} ({exc})\n")
            continue

        # Local cloud-cover post-filter
        if local_cc_max is not None and r.items:
            r.items   = [it for it in r.items if
                         it.cloud_cover is None or it.cloud_cover <= local_cc_max]
            r.returned = len(r.items)

        if r.items:
            result        = r
            collection_id = c_cid
            provider_root = c_prov
            if search_all:
                # Display each collection's result block immediately.
                # Temporarily set intent fields for display, then restore them so
                # the main intent is not left pointing at an arbitrary loop iteration.
                print("                                                            ", end="\r")
                _save_cid, _save_prov = intent.collection_id, intent.provider_root
                intent.bbox          = bbox
                intent.collection_id = c_cid
                intent.provider_root = c_prov
                display_item_cards(r, intent, page_start=1)
                print("  [synthesizing …]", end="\r", flush=True)
                syn = synthesize_item_results(intent, r, state["model"], state["memory"])
                intent.collection_id = _save_cid   # restore — don't leak loop state onto intent
                intent.provider_root = _save_prov
                print("                   ", end="\r")
                print(f"\nAssistant [{c_cid}]: {syn}\n")
                all_results.append((c_cid, r.returned, r, c_prov))
            else:
                break
        elif search_all:
            print("                                                            ", end="\r")
            print(f"\n  [{c_cid}] — no items found for these criteria.\n")

    print("                                                            ", end="\r")

    # ── search_all: recap + pick best collection for export/more ─────────────
    if search_all and all_results:
        if len(all_results) > 1:
            counts = ", ".join(f"{e[0]}: {e[1]}" for e in all_results)
            print(f"\n  Comparison: {counts}.")
        # Point result/collection_id/provider_root at the collection with most items
        # so that 'more' and 'export' work on the most useful result, not a random
        # last-iterated one.
        best_entry    = max(all_results, key=lambda x: x[1])
        result        = best_entry[2]
        collection_id = best_entry[0]
        provider_root = best_entry[3]
        if len(all_results) > 1:
            print(f"  'more' and 'export' will use '{collection_id}' (most items).\n")

    if result is None or not result.items:
        tried_str = ", ".join(tried)
        print(f"\nAssistant: No items found in any of the tried collections ({tried_str}).")
        if not bbox:
            print("  Tip: add a location (e.g. 'over Toulouse') to narrow the search.")
        if not datetime_range:
            print("  Tip: add a time period (e.g. 'in 2023') to narrow the search.")
        if intent.cloud_cover_max is not None:
            print(f"  Tip: cloud cover ≤{intent.cloud_cover_max}% may be filtering everything out.")
        print()
        return

    # ── Step 7: display + synthesize ──────────────────────────────────────────
    intent.bbox          = bbox
    intent.collection_id = collection_id
    intent.provider_root = provider_root

    if not search_all:
        # search_all already displayed each block inline; skip here
        display_item_cards(result, intent, page_start=1)
        print("  [synthesizing …]", end="\r", flush=True)
        answer = synthesize_item_results(intent, result, state["model"], state["memory"])
        print("                   ", end="\r")
        print(f"\nAssistant: {answer}\n")

    # State always tracks the last result (for 'more' / 'export')
    state["last_search_result"]        = result
    state["last_page_start"]           = 1 + result.returned
    state["last_intent"]               = intent
    state["_last_used_collection_id"]  = collection_id

    # Fix 9: if request was item_links, show them right after the search
    if intent.request_type == "item_links":
        _show_item_links(result)


def _fresh_search_or_last(intent: Intent, res: Resources, last_hits: list) -> list:
    """
    Search ChromaDB for a fresh set of collection hits.
    Falls back to last_hits if the top ChromaDB result is below the 0.35 confidence floor.

    Satellite guard (runs only when score < 0.35 AND a satellite was explicitly named):
      If last_hits are from a different satellite family than the current intent,
      return [] instead of silently reusing them — the caller will print "not found"
      rather than searching S2 collections when the user asked for S1.

      The guard checks whether "sentinel-1" / "sentinel-2" / "sentinel-3" / "sentinel-5"
      appears in the concatenated collection_id+title of the last hits.  We use the full
      "sentinel-N" string (not the short code "s3") because "s3" is not a substring of
      "sentinel-3-slstr-wst-l2-netcdf" but "sentinel-3" is.
    """
    print("  [searching stac_collections …]", end="\r", flush=True)
    hits = search_stac_collections(intent, res, top_k=3)
    print("                                  ", end="\r")

    # Good result — use it directly.
    if hits and hits[0]["score"] >= 0.35:
        return hits

    # Score too low — consider falling back to last_hits.
    # But first: if the user named a satellite, verify last_hits are from the same family.
    if intent.satellite and last_hits:
        _SAT_TEXT_TOKENS = {
            "S1": "sentinel-1", "S2": "sentinel-2",
            "S3": "sentinel-3", "S5P": "sentinel-5",
        }
        sat_text = _SAT_TEXT_TOKENS.get(intent.satellite, intent.satellite.lower())
        last_text = " ".join(
            (h.get("collection_id", "") + " " + h.get("title", "")).lower()
            for h in last_hits[:2]
        )
        if sat_text not in last_text:
            # last_hits are for a different satellite — don't silently reuse them.
            return []

    if last_hits:
        return last_hits
    return []


def _show_item_links(result: SearchResult) -> None:
    """Print all data asset hrefs from the current search result."""
    print(f"\n  Download links ({result.collection_id}):\n")
    any_links = False
    for item in result.items:
        if item.data_hrefs:
            any_links = True
            print(f"  [{item.item_id[:50]}]")
            for k, href in item.data_hrefs.items():
                print(f"    {k}: {href}")
            print()
    if not any_links:
        print("  (No data-role assets found — check all_assets in the item cards above.)\n")


def _parse_export_command(
    user_input: str, state: dict, model: str
) -> tuple[Optional[int], Optional[str]]:
    """
    Parse 'export [N] [for <date phrase>]' into (max_items, datetime_override).
    Examples:
      "export 50"                → (50, None)
      "export 100 for summer 2025" → (100, "2025-06-21/2025-09-22")
      "export"                   → (None, None)
    """
    max_items = None
    m = _EXPORT_N_RE.search(user_input)
    if m:
        max_items = int(m.group(1))

    # Extract any date context that follows
    bare = re.sub(r'\b(export|save\s+all?|items?)\s*\d*\s*', '', user_input, flags=re.IGNORECASE).strip()
    bare = re.sub(r'^(for|in|from|during)\s+', '', bare, flags=re.IGNORECASE).strip()
    datetime_override = None
    if len(bare) > 3:
        try:
            date_intent = extract_intent(bare, state.get("history", []), model)
            if date_intent.date_start:
                datetime_override = _datetime_range(date_intent)
        except Exception:
            pass

    return max_items, datetime_override


def _handle_export_summary(state: dict) -> None:
    """Print a local summary of the last search/export without calling any API."""
    result      = state.get("last_search_result")
    intent      = state.get("last_intent")
    export_path = state.get("last_export_path")
    export_count= state.get("last_export_count")

    if result is None:
        print("\nAssistant: No export has been run yet in this session. "
              "Run a search then type 'export'.\n")
        return

    print(f"\n{_W}")
    print("  LAST EXPORT SUMMARY")
    print(_W)
    print(f"\n  Collection  : {result.collection_id}")
    print(f"  Provider    : {result.provider_root.replace('https://', '').rstrip('/')}")
    if intent:
        loc = intent.location_text or intent.country or ""
        if loc:
            print(f"  Location    : {loc}")
        if intent.bbox and len(intent.bbox) >= 4:
            w, s, e, n = intent.bbox[:4]
            print(f"  BBox        : W={w} S={s} E={e} N={n}")
        if intent.date_start:
            print(f"  Date range  : {intent.date_start} → {intent.date_end or 'open'}")
    if result.total_matched is not None:
        print(f"  Matched     : {result.total_matched:,} items in provider")
    if export_count is not None:
        print(f"  Exported    : {export_count:,} items written")
    else:
        print("  Exported    : (no export file written yet — type 'export' to save)")
    if export_path:
        print(f"  File        : {export_path}")
    print(f"\n{_W}\n")


def _handle_more(res: Resources, state: dict) -> None:
    """
    Fetch the next page of items from the last search when the user types 'more'.

    Uses the STAC standard next_href link from the previous response.
    Some providers expire their GET pagination tokens (e.g. Planetary Computer SAS tokens),
    so if the GET fails we fall back to a POST /search with an offset parameter.
    A second guard corrects a Terrascope-specific bug where next_href omits the
    collection filter, causing page 2+ to return items from a different collection.
    """
    result: Optional[SearchResult] = state.get("last_search_result")
    if result is None:
        print("\nAssistant: No active search to continue.\n")
        return
    if not result.has_more or not result.next_href:
        print("\nAssistant: No more pages available for this search.\n")
        return

    print("  [fetching next page …]", end="\r", flush=True)
    access = res.stac_searcher.providers.access_type(result.provider_root)

    with make_client(30, access) as client:
        response, err = _fetch(client, result.next_href, method="GET")
        # Fallback: GET pagination token expired (common on Planetary Computer).
        # Retry as POST /search with an explicit offset.
        _used_post_fallback = False
        if response is None and result.search_url:
            intent = state.get("last_intent")
            offset = state.get("last_page_start", 1) - 1
            fallback_payload = {
                "collections": [result.collection_id],
                "limit":       ITEM_PAGE_SIZE,
                "offset":      offset,
            }
            if intent and intent.bbox:
                fallback_payload["bbox"] = intent.bbox
            if intent:
                dt = _datetime_range(intent)
                if dt:
                    fallback_payload["datetime"] = dt
            response, err2 = _fetch(
                client, result.search_url, method="POST", body=fallback_payload
            )
            if response is None:
                err = f"{err} / POST fallback: {err2}"
            else:
                _used_post_fallback = True

    if response is None:
        print(f"\nAssistant: Failed to fetch next page — {err}\n")
        return

    # Detect duplicate page: some providers (e.g. Planetary Computer) use cursor-based
    # pagination and do not support the `offset` parameter.  When the GET token expires
    # and the POST fallback ignores offset, the response is page 1 again.
    if _used_post_fallback and result.items:
        prev_ids       = {it.item_id for it in result.items}
        first_new_id   = (response.get("features") or [{}])[0].get("id", "")
        if first_new_id and first_new_id in prev_ids:
            print(
                "\nAssistant: Pagination token expired and the offset fallback returned "
                "duplicate results — this provider uses cursor-based pagination that "
                "does not support offset.\n"
                "  Please re-run your original search to continue from the beginning.\n"
            )
            return

    # Guard: some providers (e.g. Terrascope) drop the collection filter in
    # the next_href, returning a different collection on page 2+.
    features = response.get("features", [])
    if features:
        leaked = features[0].get("collection", "")
        if leaked and leaked != result.collection_id:
            intent_pg = state.get("last_intent")
            offset_pg = state.get("last_page_start", 1) - 1

            # Fallback 1: POST /search with explicit collection + offset.
            response2 = None
            if result.search_url:
                fallback_pg = {
                    "collections": [result.collection_id],
                    "limit":       ITEM_PAGE_SIZE,
                    "offset":      offset_pg,
                }
                if intent_pg and intent_pg.bbox:
                    fallback_pg["bbox"] = intent_pg.bbox
                if intent_pg:
                    dt_pg = _datetime_range(intent_pg)
                    if dt_pg:
                        fallback_pg["datetime"] = dt_pg
                with make_client(30, access) as client2:
                    response2, _ = _fetch(
                        client2, result.search_url, method="POST", body=fallback_pg
                    )

            # Fallback 2: standard OGC /collections/{id}/items endpoint.
            # Every conformant STAC API exposes this regardless of POST /search support.
            if response2 is None:
                items_url = (
                    f"{result.provider_root.rstrip('/')}"
                    f"/collections/{result.collection_id}/items"
                    f"?limit={ITEM_PAGE_SIZE}&offset={offset_pg}"
                )
                if intent_pg and intent_pg.bbox:
                    w, s, e, n = intent_pg.bbox[:4]
                    items_url += f"&bbox={w},{s},{e},{n}"
                with make_client(30, access) as client3:
                    response2, err3 = _fetch(client3, items_url, method="GET")

            if response2 is not None:
                response = response2
                features = response.get("features", [])
            else:
                print(f"\nAssistant: Could not retrieve page 2 for {result.collection_id} "
                      f"(provider dropped collection filter and both fallbacks failed). "
                      f"Re-run the search to try again.\n")
                return
    total     = extract_count(response)
    next_href = get_next_link(response)
    items     = [parse_item(f, result.provider_root, result.collection_id) for f in features]
    print("                        ", end="\r")

    new_result = SearchResult(
        items         = items,
        total_matched = total if total is not None else result.total_matched,
        returned      = len(items),
        has_more      = next_href is not None,
        next_href     = next_href,
        provider_root = result.provider_root,
        collection_id = result.collection_id,
        search_url    = result.search_url,
        query_ts      = _now_ts(),
    )

    intent     = state.get("last_intent")
    page_start = state.get("last_page_start", 1)

    display_item_cards(new_result, intent, page_start=page_start)

    print("  [synthesizing …]", end="\r", flush=True)
    answer = synthesize_item_results(intent, new_result, state["model"], state["memory"])
    print("                   ", end="\r")
    print(f"\nAssistant: {answer}\n")

    state["last_search_result"] = new_result
    state["last_page_start"]    = page_start + new_result.returned


def _handle_export(
    res: Resources,
    state: dict,
    max_items: Optional[int] = None,
    datetime_override: Optional[str] = None,
) -> None:
    result: Optional[SearchResult] = state.get("last_search_result")
    intent: Optional[Intent]       = state.get("last_intent")
    interactive = state.get("_interactive", True)

    if result is None or intent is None:
        print("\nAssistant: No active search to export. Run a search first.\n")
        return

    total          = result.total_matched
    effective_range = datetime_override or _datetime_range(intent)

    # ── Step 1: show available count ──────────────────────────────────────────
    print()
    print(f"  Collection : {result.collection_id}")
    print(f"  Provider   : {result.provider_root.replace('https://','').rstrip('/')}")
    if intent:
        loc = intent.location_text or intent.country or ""
        if loc:
            print(f"  Location   : {loc}")
        if intent.date_start:
            print(f"  Period     : {intent.date_start} → {intent.date_end or 'open'}")
    if total is not None:
        print(f"  Available  : {total:,} items matched in provider")
    else:
        print("  Available  : unknown total (provider does not expose count)")
    if datetime_override:
        print(f"  Date override: {effective_range}")
    print()

    # ── Step 2: ask how many to export (skip if already provided via command) ─
    if max_items is None and interactive:
        cap_str = f"max {EXPORT_MAX_ITEMS:,}"
        avail_str = f"{total:,} available" if total is not None else "unknown total"
        try:
            ans = input(
                f"  How many items to export? ({avail_str}, {cap_str})\n"
                "  Enter a number, 'all', or press Enter to cancel: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Export cancelled.\n")
            return

        if not ans:
            print("  Export cancelled.\n")
            return
        elif ans in ("all", "a"):
            max_items = EXPORT_MAX_ITEMS
        else:
            try:
                max_items = int(ans)
                if max_items <= 0:
                    print("  Export cancelled — must be a positive number.\n")
                    return
            except ValueError:
                print(f"  '{ans}' is not a valid number — export cancelled.\n")
                return
    elif max_items is None:
        max_items = EXPORT_MAX_ITEMS   # non-interactive: use hard cap without asking

    # ── Step 3: warn if the requested amount is large ─────────────────────────
    effective_max   = min(max_items, EXPORT_MAX_ITEMS)
    effective_total = min(total, effective_max) if total is not None else effective_max

    if effective_total > EXPORT_WARN_THRESH and interactive:
        try:
            confirm = input(
                f"  Warning: exporting {effective_total:,} items. Continue? (yes/no): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Export cancelled.\n")
            return
        if confirm not in ("yes", "y"):
            print("  Export cancelled.\n")
            return

    # ── Step 4: run export ────────────────────────────────────────────────────
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"stac_export_{result.collection_id}_{ts}.jsonl"
    out_path = EXPORT_DIR / filename

    print(f"  Exporting {effective_max:,} items to {out_path} …")

    try:
        n = res.stac_searcher.export(
            provider_root  = result.provider_root,
            collection_id  = result.collection_id,
            output_path    = str(out_path),
            bbox           = intent.bbox,
            datetime_range = effective_range,
            max_items      = effective_max,
        )
        state["last_export_path"]  = str(out_path)
        state["last_export_count"] = n
        print(f"\nAssistant: Exported {n:,} items to {out_path}\n")
    except Exception as exc:
        print(f"\nAssistant: Export failed — {exc}\n")


# ── Debug / commands ───────────────────────────────────────────────────────────

def _print_intent_debug(intent: Intent) -> None:
    print("\n  [intent]")
    print(f"    mode           : {intent.mode}")
    print(f"    request_type   : {intent.request_type}")
    print(f"    satellite      : {intent.satellite}")
    print(f"    theme          : {intent.theme}")
    print(f"    mission_type   : {intent.mission_type}")
    print(f"    location_text  : {intent.location_text}")
    print(f"    country        : {intent.country}")
    print(f"    date           : {intent.date_start} → {intent.date_end}")
    print(f"    wants_items    : {intent.wants_items}")
    print(f"    wants_count    : {intent.wants_count}")
    print(f"    cloud_cover_max: {intent.cloud_cover_max}")
    print(f"    item_id        : {intent.item_id}")
    print(f"    collection_id  : {intent.collection_id}")
    print(f"    bbox           : {intent.bbox}")
    print(f"    confidence     : {intent.confidence:.2f}")
    if intent.ambiguities:
        print(f"    ambiguities    : {intent.ambiguities}")
    if intent.assumptions:
        print(f"    assumptions    : {intent.assumptions}")
    print()


def _handle_command(cmd: str, state: dict) -> None:
    parts = cmd.strip().split()
    name  = parts[0].lower()

    if name == "/help":
        print("""
Commands:
  /help                      show this help
  /model <name>              switch Ollama model (e.g. /model llama3)
  /models                    list available Ollama models
  /verbose                   toggle verbose debug output
  /top <n>                   set number of collections to show (default 5)
  /clear                     clear conversation history (also clears Redis session)
  /sessions                  list all active sessions stored in Redis
  /token <keyword> <token>   set a bearer token for a provider (e.g. /token planetary mytoken)
                             keyword is matched against known provider URLs

Special inputs:
  more           fetch next page of items from last search
  export         export all items from last search to kb/exports/
  links          show download links from last search result
  quit / exit    quit
""")
    elif name == "/verbose":
        state["verbose"] = not state["verbose"]
        print(f"  Verbose: {'on' if state['verbose'] else 'off'}")
    elif name == "/model":
        if len(parts) > 1:
            state["model"] = parts[1]
            print(f"  Model: {state['model']}")
        else:
            print(f"  Current model: {state['model']}")
    elif name == "/models":
        try:
            result = ollama.list()
            for m in (result.get("models") or []):
                name_m = m.get("name") or m.get("model", "?")
                print(f"  {name_m}")
        except Exception as exc:
            print(f"  Error listing models: {exc}")
    elif name == "/top":
        if len(parts) > 1:
            try:
                state["top_k"] = max(1, int(parts[1]))
                print(f"  top_k: {state['top_k']}")
            except ValueError:
                print("  Usage: /top <integer>")
        else:
            print(f"  Current top_k: {state['top_k']}")
    elif name == "/clear":
        state["memory"].clear()
        state["history"] = []
        state["clarification_count"] = 0
        _store = state.get("_session_store")
        if _store:
            _store.delete()
            print("  Conversation history cleared (Redis session deleted).")
        else:
            print("  Conversation history cleared.")
    elif name == "/sessions":
        sessions = SessionStore.list_all()
        if sessions:
            current = state.get("_session_id", "default")
            print(f"  Active sessions in Redis ({len(sessions)}):")
            for s in sessions:
                marker = "  ← current" if s == current else ""
                print(f"    {s}{marker}")
        else:
            print("  No sessions found in Redis (Redis may not be running).")
    elif name == "/token":
        if len(parts) < 3:
            print("  Usage: /token <provider_keyword> <bearer_token>")
            print("  Example: /token planetary ghp_abc123")
        else:
            keyword  = parts[1].lower()
            token    = parts[2]
            _res     = state.get("_res")
            if _res is None:
                print("  Resources not initialised yet.")
            else:
                providers = _res.stac_searcher.providers
                matched   = [p for p in providers.list_providers() if keyword in p.lower()]
                if not matched:
                    print(f"  No provider matching '{keyword}'. Known providers:")
                    for p in providers.list_providers():
                        print(f"    {p}")
                elif len(matched) > 1:
                    print(f"  Ambiguous — '{keyword}' matched {len(matched)} providers:")
                    for m in matched:
                        print(f"    {m}")
                    print("  Use a more specific keyword.")
                else:
                    providers.set_token(matched[0], token)
                    print(f"  Bearer token set for {matched[0]} (session only — not persisted).")
    else:
        print(f"  Unknown command: {name}  (try /help)")


# ── Main chat loop ─────────────────────────────────────────────────────────────

def run_chat(
    model:       str  = "mistral",
    top_k:       int  = 5,
    verbose:     bool = False,
    ask:         str  = None,
    session_id:  str  = "default",
    memory_type: str  = "buffer",
) -> None:
    """
    Main REPL loop.  Handles one user message per iteration.

    The full routing table (in order — first match wins):
      1. quit / exit / q / bye         → exit the loop
      2. /command                      → _handle_command() (slash commands)
      3. "more" / "next page"          → _handle_more() (pagination)
      4. "export …" / "save all"       → _handle_export() (save to JSONL)
      5. "links" / "download links"    → _show_item_links() from last result
      6. "about the export?" etc.      → _handle_export_summary()
      7. Meta-questions ("why so many?") → _handle_meta_question()
      8. Everything else               → extract_intent() → _keyword_enhance()
                                         → pre-routing fixes → handler dispatch:
                                           collection_discovery → _handle_collection_discovery()
                                           item_preview/count/links → _handle_item_search()
                                           item_by_id → _handle_item_by_id()
                                           item_export → _handle_item_search() + _handle_export()

    `state` is the mutable session dictionary shared by all handlers.
    """
    res = load_resources()

    # ── Redis session setup ────────────────────────────────────────────────────
    # Try to connect to Redis and load any prior history for this session.
    # If Redis is not running or not installed the store silently becomes a no-op
    # and the chatbot behaves exactly as before (in-memory only).
    store         = SessionStore(session_id)
    prior_data    = store.load()
    memory        = create_memory(memory_type, prior_data)

    _mem_labels = {
        "raw":     "RawMessage  (full turns, bounded)",
        "summary": "Summary     (LLM rolling summary)",
        "buffer":  "Buffer+Summary (recent raw + compressed old)",
    }
    print(f"\nSTAC RAG v3  —  model: {model}  |  top_k: {top_k}  |  session: {session_id}")
    print(f"  Memory: {_mem_labels.get(memory_type, memory_type)}")
    if store.available:
        print(f"  Redis connected — session '{session_id}' persisted across restarts.")
        if memory.turn_count:
            print(f"  Resumed: {memory.turn_count} prior turn(s) loaded from Redis.")
    else:
        print("  Redis not available — history in-memory only (install redis + run redis-server).")
    print("Sources: stac_collections (ChromaDB) + live STAC /search APIs")
    print("No synthetic query_lookup.jsonl — all counts are real.\n")
    print("Type your question, or /help for commands.\n")

    # interactive=False when called via --ask or when stdin is a pipe (non-TTY).
    # In non-interactive mode, confirmation prompts are skipped.
    interactive = ask is None and sys.stdin.isatty()

    # Session state — persists for the entire conversation.
    state = {
        "model":                    model,
        "top_k":                    top_k,
        "verbose":                  verbose,
        "memory":                   memory,      # active MemoryManager (raw/summary/buffer)
        "history":                  memory.to_list(),  # kept for callers still using the raw list
        "clarification_count":      0,
        "last_search_result":       None,
        "last_page_start":          1,
        "last_intent":              None,
        "last_collection_hits":     None,
        "pending_candidates":       None,
        "_last_used_collection_id": None,
        "_interactive":             interactive,
        "_raw_input":               None,
        "last_export_path":         None,
        "last_export_count":        None,
        "_pending_intent":          None,
        "_res":                     res,
        "_last_topic_key":          None,
        "_session_store":           store,
        "_session_id":              session_id,
    }

    while True:
        # ── Input ──────────────────────────────────────────────────────────────
        if ask is not None:
            user_input = ask
        else:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

        if not user_input:
            continue

        lower = user_input.lower().strip()
        state["_raw_input"] = user_input

        # ── Quit ───────────────────────────────────────────────────────────────
        if lower in ("quit", "exit", "q", "bye"):
            print("Bye.")
            break

        # ── Commands ───────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            _handle_command(user_input, state)
            if ask is not None:
                break
            continue

        # ── Pagination ─────────────────────────────────────────────────────────
        if lower in ("all items", "show all", "show all items", "see all", "see more items"):
            result = state.get("last_search_result")
            intent = state.get("last_intent")
            if result and result.items:
                for i, item in enumerate(result.items, 1):
                    lines = [f"\n  [{i}] {item.item_id}"]
                    dt = item.datetime or item.start_datetime
                    if dt:
                        lines.append(f"      Date       : {dt[:19].replace('T', ' ')} UTC")
                    if item.cloud_cover is not None:
                        lines.append(f"      Cloud cover: {item.cloud_cover}%")
                    if item.platform:
                        lines.append(f"      Platform   : {item.platform}")
                    print("\n".join(lines))
                print(f"\n{_W}")
            else:
                print("\nAssistant: No items loaded. Run a search first.\n")
            if ask is not None:
                break
            continue

        if lower in ("more", "show more", "next", "next page"):
            _handle_more(res, state)
            if ask is not None:
                break
            continue

        # ── Export ─────────────────────────────────────────────────────────────
        if lower.startswith("export") or lower == "save all":
            max_items, dt_override = _parse_export_command(user_input, state, state["model"])
            _handle_export(res, state, max_items=max_items, datetime_override=dt_override)
            if ask is not None:
                break
            continue

        # ── Links shortcut ─────────────────────────────────────────────────────
        _is_link_request = (
            "download link" in lower or "give me link" in lower
            or lower in ("links", "download links", "give download links",
                         "show links", "asset links")
        )
        if _is_link_request:
            if state["last_search_result"]:
                _show_item_links(state["last_search_result"])
            else:
                print(
                    "\nAssistant: No item search done yet — I need to fetch real products first.\n"
                    "  Ask something like: 'Show me SAR data over Paris in 2023'\n"
                    "  Then type 'links' to see the asset URLs.\n"
                )
            if ask is not None:
                break
            continue

        # ── E4: post-export summary query ──────────────────────────────────────
        if (_ABOUT_EXPORT_RE.search(lower)
                and state.get("last_search_result")
                and state.get("last_export_path")):
            _handle_export_summary(state)
            if ask is not None:
                break
            continue

        # ── Meta-questions about prior results ─────────────────────────────────
        # "why are there so many?", "how do these compare?", "explain this" etc.
        if (_META_Q_RE.search(lower)
                and (state.get("last_search_result") or state.get("last_collection_hits"))):
            _handle_meta_question(user_input, state)
            if ask is not None:
                break
            continue

        # ── Pre-intent: extract manual bbox from raw text ─────────────────────
        # Done before intent extraction so we can inject it into the intent after.
        manual_bbox = _parse_manual_bbox(user_input)

        # ── Intent extraction ──────────────────────────────────────────────────
        # Two-phase: LLM → deterministic keyword post-processor.
        print("  [extracting intent …]", end="\r", flush=True)
        intent = extract_intent(user_input, state["memory"], state["model"])
        intent = _keyword_enhance(intent, user_input)
        print("                        ", end="\r")

        # Inject manual bbox if the LLM didn't pick one up from the text.
        if manual_bbox and intent.bbox is None:
            intent.bbox = manual_bbox
            intent.assumptions.append(f"manual bbox from input: {manual_bbox}")

        # Restore topic from a pending clarification: if the previous turn fired a
        # clarification prompt (user was asked for location/date), the saved intent
        # holds the topic (satellite/theme/mission_type) that was missing then.
        # Merge it into the new intent which now has the location/date answer.
        _pending = state.get("_pending_intent")
        if _pending and not (intent.satellite or intent.theme or intent.mission_type):
            intent.satellite    = _pending.satellite    or intent.satellite
            intent.theme        = _pending.theme        or intent.theme
            intent.mission_type = _pending.mission_type or intent.mission_type
            if intent.mode == "collection_discovery":
                intent.mode         = "item_search"
                intent.request_type = "item_preview"
                intent.wants_items  = True

        # Correction: if the LLM put a collection ID (from last_collection_hits) into
        # item_id, move it to collection_id — it's a collection reference, not an item.
        if intent.item_id and state.get("last_collection_hits"):
            known_cids = {h["collection_id"] for h in state["last_collection_hits"]}
            if intent.item_id in known_cids:
                intent.collection_id = intent.item_id
                intent.item_id       = None
                intent.request_type  = "item_preview"
                intent.mode          = "item_search"
                intent.wants_items   = True

        # ── Early topic-change detection ───────────────────────────────────────────
        # Detect when the user switches topic BEFORE the pre-routing blocks below
        # act on stale collection state, and clear that state first.
        #
        # Two cases that clear stale state:
        #   _sat_changed:      user explicitly names a different satellite (S1→S2, SAR→optical).
        #   _discovery_reset:  after item searches on satellite X, user asks a fresh
        #                      open-ended discovery question with a different theme and
        #                      no specific satellite ("what data for deforestation?").
        #
        # In both cases we first inherit location/date from the prior intent so that
        # "show me optical data for the SAME LOCATION" still resolves the bbox.
        _old_topic = state.get("_last_topic_key") or ""
        _parts     = _old_topic.split("|", 1) if _old_topic else ["", ""]
        _old_sat   = _parts[0] if _parts[0] not in ("None", "") else ""
        _old_theme = (_parts[1] if len(_parts) > 1 and _parts[1] not in ("None", "") else "")
        _new_sat   = intent.satellite or ""
        _new_theme = intent.theme or ""

        _sat_changed = bool(_old_sat and _new_sat and _old_sat != _new_sat)
        _discovery_reset = bool(
            _old_sat                          # prior context had a specific satellite
            and not _new_sat                  # new query has no specific satellite
            and _new_theme                    # but does specify a theme
            and _new_theme != _old_theme      # and it differs from the old one
            and intent.mode == "collection_discovery"  # LLM classified as discovery
        )
        # LLM returned mission_type="optical"/"radar" instead of satellite="S2"/"S1"
        # — treat as satellite change when the inferred satellite differs from the prior.
        _MISSION_TO_SAT = {"optical": "S2", "radar": "S1", "ocean": "S3", "atmospheric": "S5P"}
        _mission_sat = _MISSION_TO_SAT.get(intent.mission_type or "", "")
        _mission_type_changed = bool(
            _old_sat
            and not _new_sat               # LLM returned no satellite code
            and _mission_sat               # but mission_type maps to one
            and _mission_sat != _old_sat   # and it differs from the prior satellite
        )

        if _sat_changed or _discovery_reset or _mission_type_changed:
            # Inherit location/date before wiping last_intent.
            _prior = state.get("last_intent")
            if _prior:
                if not intent.location_text and _prior.location_text:
                    intent.location_text = _prior.location_text
                if not intent.country and _prior.country:
                    intent.country = _prior.country
                if not intent.bbox and _prior.bbox:
                    intent.bbox = _prior.bbox
                if not intent.date_start and _prior.date_start:
                    intent.date_start = _prior.date_start
                    intent.date_end   = _prior.date_end
            # Wipe stale collection context so routing and _fresh_search_or_last
            # start fresh instead of falling back to old satellite collections.
            state["last_collection_hits"]     = None
            state["last_search_result"]       = None
            state["last_intent"]              = None
            state["_last_used_collection_id"] = None
            # For re-discovery: enforce collection_discovery even if the LLM
            # was misled by memory context into returning item_preview.
            if _discovery_reset:
                intent.request_type = "collection_discovery"
                intent.mode         = "collection_discovery"
                intent.wants_items  = False

        # Pre-routing: "other N" / "all these" / "other collections" with prior hits
        # → force item_search and carry over spatial/temporal context from last_intent
        # so the user doesn't need to repeat location and date.
        if state.get("last_collection_hits"):
            _rq = state.get("_raw_input", user_input)
            if _ALL_THESE_RE.search(_rq) or _OTHER_N_RE.search(_rq) or _OTHER_RE.search(_rq):
                intent.mode         = "item_search"
                intent.request_type = "item_preview"
                intent.wants_items  = True
                _li = state.get("last_intent")
                if _li:
                    if not intent.location_text and _li.location_text:
                        intent.location_text = _li.location_text
                    if not intent.country and _li.country:
                        intent.country = _li.country
                    if not intent.bbox and _li.bbox:
                        intent.bbox = _li.bbox
                    if not intent.date_start and _li.date_start:
                        intent.date_start = _li.date_start
                    if not intent.date_end and _li.date_end:
                        intent.date_end = _li.date_end

        # Pre-routing: "over Toulouse in 2023" with no topic but prior collection hits
        # → force item_search on those collections (the user is refining their last query).
        has_topic = bool(intent.satellite or intent.theme or intent.mission_type)
        if (not has_topic
                and (intent.location_text or intent.country or intent.bbox or intent.date_start)
                and state.get("last_collection_hits")
                and intent.mode == "collection_discovery"):
            intent.mode         = "item_search"
            intent.request_type = "item_preview"
            intent.wants_items  = True

        if state["verbose"]:
            _print_intent_debug(intent)

        # ── Route via request_type ─────────────────────────────────────────────
        # request_type is set deterministically by _keyword_enhance() and never by the LLM.
        rt = intent.request_type
        if rt == "collection_discovery":
            _handle_collection_discovery(intent, res, state)
        elif rt == "item_by_id":
            _handle_item_by_id(intent, res, state)
        elif rt == "item_export":
            # Run the search first (to get the collection and bbox), then export.
            if state["last_search_result"] is None:
                _handle_item_search(intent, res, state)
            _handle_export(res, state)
        elif rt in ("item_preview", "item_count", "item_links"):
            _handle_item_search(intent, res, state)
        else:
            # Fallback for request_types not covered above — use the mode flag.
            if intent.mode == "item_search" or intent.wants_items or intent.wants_count:
                _handle_item_search(intent, res, state)
            else:
                _handle_collection_discovery(intent, res, state)

        # ── History update ─────────────────────────────────────────────────────
        # We store a compact structured summary as the "assistant" turn rather than
        # the full synthesised text.  This keeps the context window small while still
        # letting the next extract_intent() call know which satellite/collection was last used.
        # The summary includes which collections were shown so the LLM can resolve
        # "that one" or "the second collection" in the next turn.
        col_ctx = ""
        if state.get("last_collection_hits"):
            col_ids = [h["collection_id"][:35] for h in state["last_collection_hits"][:3]]
            col_ctx = f" | shown_collections=[{', '.join(col_ids)}]"
        elif state.get("last_search_result"):
            col_ctx = f" | searched_collection={state['last_search_result'].collection_id}"

        summary = (
            f"mode={intent.mode} sat={intent.satellite} theme={intent.theme} "
            f"loc={intent.location_text or intent.country or '?'} "
            f"date={intent.date_start or 'none'}{col_ctx}"
        )
        mem: MemoryManager = state["memory"]

        # Topic-change detection: when the user switches satellite, wipe memory so
        # the new topic starts clean without interference from the old one.
        new_topic_key = f"{intent.satellite}|{intent.theme}"
        old_topic_key = state["_last_topic_key"]
        if (old_topic_key and new_topic_key != old_topic_key
                and intent.satellite and old_topic_key.split("|")[0] != intent.satellite):
            mem.clear()
        state["_last_topic_key"] = new_topic_key

        # Deduplication: skip if the user sent the exact same message as last time.
        if mem.last_user_message() != user_input:
            mem.add_exchange(user_input, summary, state["model"])

        # Keep the legacy state["history"] list in sync for any callers still using it.
        state["history"] = mem.to_list()

        # Persist to Redis after every turn so restarts resume correctly.
        store.save(mem.to_list())

        if ask is not None:
            break


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STAC-first satellite data chatbot — real collections and real items only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default="mistral", help="Ollama model name")
    p.add_argument("--top",     type=int, default=5,
                   help="Number of collections to show in collection discovery mode")
    p.add_argument("--verbose", action="store_true", help="Show intent debug info")
    p.add_argument("--ask",     default=None, metavar="QUESTION",
                   help="Run one question non-interactively and exit")
    p.add_argument("--session", default="default", metavar="NAME",
                   help="Session name for Redis history persistence (default: 'default'). "
                        "Use different names for separate users: --session alice")
    p.add_argument("--memory", default="buffer",
                   choices=["raw", "summary", "buffer"],
                   help=(
                       "Memory strategy: "
                       "'raw' = full messages bounded list (Section 2 / local Ollama); "
                       "'summary' = rolling LLM summary after each turn (Section 3 / LangChain style); "
                       "'buffer' = recent turns verbatim + LLM summary of older ones (Section 4 / Llama-Index style, default)"
                   ))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_chat(
        model       = args.model,
        top_k       = args.top,
        verbose     = args.verbose,
        ask         = args.ask,
        session_id  = args.session,
        memory_type = args.memory,
    )
