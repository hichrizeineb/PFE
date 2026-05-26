#!/usr/bin/env python3
"""
stac_item_search.py
===================
Standalone module for searching real STAC items/products from live provider APIs.

Core mechanism (from supervisor):
    provider_root + "/search"  →  POST JSON payload  →  items/products

Payload structure:
    {
        "collections": ["collection_id"],   ← what
        "bbox":        [W, S, E, N],        ← where
        "datetime":    "start/end",         ← when
        "limit":       10
    }

Usage (standalone CLI test):
    python3 stac_item_search.py \\
        --provider  https://stac.terrascope.be \\
        --collection sentinel-2-l2a \\
        --bbox      "1.30,43.50,1.60,43.75" \\
        --datetime  "2025-06-21/2025-09-22" \\
        --limit     10

    python3 stac_item_search.py \\
        --provider   https://cmr.earthdata.nasa.gov/stac/JAXA \\
        --collection ALOS_PALSAR_RTC_HIGH_RES \\
        --datetime   "2020-01-01/2020-12-31" \\
        --count-only

    python3 stac_item_search.py \\
        --provider   https://stac.terrascope.be \\
        --collection sentinel-2-l2a \\
        --item-id    S2B_32TLQ_20250901_0_L2A

    python3 stac_item_search.py \\
        --provider   https://stac.terrascope.be \\
        --collection sentinel-2-l2a \\
        --bbox       "1.30,43.50,1.60,43.75" \\
        --datetime   "2025-06-21/2025-09-22" \\
        --export     results.jsonl

Requirements:
    pip install httpx
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")


# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger("stac_item_search")


# ── Defaults ───────────────────────────────────────────────────────────────────

PROVIDERS_PATH         = "kb/outputs/stac_providers.jsonl"
DEFAULT_LIMIT          = 10
DEFAULT_MAX_PAGES      = 10
DEFAULT_MAX_ITEMS      = 5_000
EXPORT_CONFIRM_THRESH  = 1_000
DEFAULT_TIMEOUT        = 30
DEFAULT_RETRIES        = 3
PAGE_SIZE_EXPORT       = 100


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class ParsedItem:
    """One real satellite product/observation extracted from a STAC feature."""
    item_id:        str
    collection_id:  str
    provider_root:  str
    datetime:       str
    start_datetime: str
    end_datetime:   str
    bbox:           list
    geometry_type:  str
    cloud_cover:    Optional[float]
    platform:       str
    instruments:    list
    data_hrefs:     dict    # key → href for assets with role=data
    thumbnail_href: str
    metadata_href:  str
    all_assets:     dict    # key → {href, roles, type}
    properties:     dict    # first 15 properties (no datetime/created/updated)


@dataclass
class SearchResult:
    """Result of one /search call — first page + metadata."""
    items:          list[ParsedItem]
    total_matched:  Optional[int]   # None if provider does not expose count
    returned:       int
    has_more:       bool
    next_href:      Optional[str]
    provider_root:  str
    collection_id:  str
    search_url:     str
    query_ts:       str


# ── Provider index ─────────────────────────────────────────────────────────────

class ProviderIndex:
    """
    In-memory index of provider records loaded from stac_providers.jsonl.
    Keyed by canonical provider_root (trailing slash stripped).
    """

    def __init__(self, path: str = PROVIDERS_PATH) -> None:
        self._index: dict[str, dict] = {}
        p = Path(path)
        if not p.exists():
            log.warning("Providers file not found: %s — fallback URLs will be constructed.", path)
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("provider_root", "").rstrip("/")
                    if key:
                        self._index[key] = rec
                except json.JSONDecodeError:
                    continue
        log.info("ProviderIndex loaded: %d providers", len(self._index))

    def get(self, provider_root: str) -> Optional[dict]:
        return self._index.get(provider_root.rstrip("/"))

    def search_url(self, provider_root: str) -> str:
        """
        Return the discovered search_url for this provider.
        Falls back to provider_root.rstrip('/') + '/search' if not found.
        This fallback is safe because STAC spec requires /search at the root path.
        """
        rec = self.get(provider_root)
        if rec and rec.get("search_url"):
            return rec["search_url"]
        return provider_root.rstrip("/") + "/search"

    def search_method(self, provider_root: str) -> str:
        rec = self.get(provider_root)
        if rec and rec.get("search_method"):
            return rec["search_method"]
        return "POST"

    def conforms_to(self, provider_root: str) -> list:
        rec = self.get(provider_root)
        if rec:
            return rec.get("conforms_to", [])
        return []

    def supports_cql2(self, provider_root: str) -> bool:
        conforms = " ".join(self.conforms_to(provider_root)).lower()
        return "cql2" in conforms or "filter" in conforms

    def access_type(self, provider_root: str) -> str:
        rec = self.get(provider_root)
        if rec:
            return rec.get("access_type", "")
        return ""

    def list_providers(self) -> list:
        """Return all known provider_root URLs."""
        return list(self._index.keys())

    def set_token(self, provider_root: str, token: str) -> bool:
        """Set a bearer token for a provider at runtime (in-memory only, not persisted)."""
        key = provider_root.rstrip("/")
        if key in self._index:
            self._index[key]["access_type"] = token
            return True
        return False

    def items_url(self, provider_root: str, collection_id: str) -> str:
        """Construct the /collections/{id}/items URL for fallback use."""
        return f"{provider_root.rstrip('/')}/collections/{collection_id}/items"


# ── HTTP client ────────────────────────────────────────────────────────────────

def make_client(timeout: int = DEFAULT_TIMEOUT, access: str = "") -> httpx.Client:
    """Build httpx.Client with JSON Accept headers and optional Bearer auth."""
    headers: dict[str, str] = {
        "Accept":     "application/json, application/geo+json",
        "User-Agent": "stac-item-search/1.0 (dataionics-rag)",
    }
    stripped = access.strip().lower()
    if access.strip() and stripped not in ("open", "public", "free", "none", ""):
        headers["Authorization"] = f"Bearer {access.strip()}"
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers=headers,
    )


# ── Payload builder ────────────────────────────────────────────────────────────

def _normalize_datetime(dt: str) -> str:
    """
    Ensure datetime string is full ISO 8601.
    Some providers (e.g. Terrascope) reject bare dates like "2025-06-21"
    and require "2025-06-21T00:00:00Z".

    Handles:
        "2025-06-21"                         → "2025-06-21T00:00:00Z"
        "2025-06-21/2025-09-22"              → "2025-06-21T00:00:00Z/2025-09-22T23:59:59Z"
        "2025-06-21T00:00:00Z/..."           → unchanged (already full)
        "../2025-09-22"                      → "../2025-09-22T23:59:59Z"
    """
    def _pad(part: str, is_end: bool) -> str:
        part = part.strip()
        if part == ".." or not part:
            return part
        if "T" not in part:
            suffix = "T23:59:59Z" if is_end else "T00:00:00Z"
            return part + suffix
        if not part.endswith("Z") and "+" not in part:
            return part + "Z"
        return part

    if "/" in dt:
        start, end = dt.split("/", 1)
        return f"{_pad(start, False)}/{_pad(end, True)}"
    return _pad(dt, False)


def build_search_payload(
    collection_id:  str,
    bbox:           Optional[list[float]] = None,
    datetime_range: Optional[str]         = None,
    limit:          int                   = DEFAULT_LIMIT,
    filters:        Optional[dict]        = None,
    sortby:         Optional[list]        = None,
    item_ids:       Optional[list[str]]   = None,
) -> dict:
    """
    Build the JSON payload to POST to /search.

    Fields are omitted entirely when None — never sent as null.
    datetime_range is normalized to full ISO 8601 automatically.

    Args:
        collection_id:  STAC collection id (what)
        bbox:           [west, south, east, north] (where)
        datetime_range: ISO 8601 interval "2023-01-01/2023-12-31" or single date (when)
        limit:          max items per page
        filters:        CQL2-JSON filter dict (only for providers that support it)
        sortby:         list of sort dicts e.g. [{"field": "datetime", "direction": "desc"}]
        item_ids:       list of specific item IDs to fetch
    """
    payload: dict[str, Any] = {
        "collections": [collection_id],
        "limit":       limit,
    }
    if bbox:
        payload["bbox"] = bbox
    if datetime_range:
        payload["datetime"] = _normalize_datetime(datetime_range)
    if item_ids:
        payload["ids"] = item_ids
    if filters:
        payload["filter-lang"] = "cql2-json"
        payload["filter"]      = filters
    if sortby:
        payload["sortby"] = sortby
    return payload


# ── Low-level HTTP fetch ───────────────────────────────────────────────────────

def _fetch(
    client:    httpx.Client,
    url:       str,
    method:    str  = "POST",
    body:      Optional[dict] = None,
    params:    Optional[dict] = None,
    retries:   int  = DEFAULT_RETRIES,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Send one HTTP request and return (response_dict, error_str).
    - POST with body, or GET with params.
    - Rejects HTML responses immediately.
    - Retries 5xx with exponential backoff (1s → 2s → 4s).
    - Does not retry 4xx (structural errors).
    - Handles 429 Retry-After.
    - Returns (None, error_str) on failure — never raises.
    """
    delay   = 1.0
    last_err: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            if method == "POST":
                r = client.post(url, json=body or {}, params=params)
            else:
                r = client.get(url, params=params)

            ct = r.headers.get("content-type", "")
            if "text/html" in ct and r.status_code == 200:
                return None, "HTML response — not a STAC JSON endpoint"

            if r.status_code == 200:
                try:
                    return r.json(), None
                except Exception as exc:
                    return None, f"JSON parse error: {exc}"

            last_err = f"HTTP {r.status_code}"

            # Structural errors — no retry
            if r.status_code in (400, 401, 403, 404, 410):
                break

            # Method not allowed — caller should switch to GET
            if r.status_code == 405:
                return None, "HTTP 405"

            # Rate limit — respect Retry-After
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", str(int(delay))))
                log.debug("  rate-limited, waiting %ds", wait)
                time.sleep(min(wait, 60))
                continue

        except httpx.TimeoutException as exc:
            last_err = f"Timeout: {exc}"
        except httpx.RequestError as exc:
            last_err = f"Request error: {exc}"
        except Exception as exc:
            last_err = f"Unexpected error: {exc}"

        if attempt < retries:
            log.debug("  retry %d/%d for %s (%s)", attempt + 1, retries, url, last_err)
            time.sleep(delay)
            delay = min(delay * 2, 4.0)

    return None, last_err


# ── Search first page ──────────────────────────────────────────────────────────

def search_first_page(
    client:  httpx.Client,
    url:     str,
    payload: dict,
    method:  str = "POST",
    retries: int = DEFAULT_RETRIES,
) -> tuple[Optional[dict], Optional[str]]:
    """
    POST (or GET) the search payload to the search URL.
    Fallback chain:
        1. POST /search with JSON body  (preferred)
        2. GET  /search with URL params (if POST returns 405)
        3. Returns (None, error_str)    (if both fail)

    Returns (response_dict, error_str).
    """
    # Step 1 — try configured method (usually POST)
    data, err = _fetch(client, url, method=method, body=payload, retries=retries)
    if data is not None:
        return data, None

    # Step 2 — fallback to GET if POST was rejected
    if method == "POST" and err == "HTTP 405":
        log.debug("  POST rejected (405), falling back to GET for %s", url)
        get_params: dict[str, Any] = {}
        if "collections" in payload:
            get_params["collections"] = ",".join(payload["collections"])
        if "bbox" in payload:
            get_params["bbox"] = ",".join(str(x) for x in payload["bbox"])
        if "datetime" in payload:
            get_params["datetime"] = payload["datetime"]
        if "limit" in payload:
            get_params["limit"] = payload["limit"]
        if "ids" in payload:
            get_params["ids"] = ",".join(payload["ids"])

        data, err = _fetch(client, url, method="GET", params=get_params, retries=retries)
        if data is not None:
            return data, None

    return None, err


# ── Count extraction ───────────────────────────────────────────────────────────

def extract_count(response: dict) -> Optional[int]:
    """
    Extract the total matching item count from a /search response.

    Tries in order:
        1. context.matched        (STAC Context Extension — 14/17 providers)
        2. numberMatched          (OGC API Features variant)
        3. Returns None           (provider does not expose total count)
    """
    ctx = response.get("context", {})
    if isinstance(ctx, dict) and "matched" in ctx:
        return int(ctx["matched"])

    nm = response.get("numberMatched")
    if nm is not None:
        try:
            return int(nm)
        except (TypeError, ValueError):
            pass

    return None


def get_next_link(response: dict) -> Optional[str]:
    """Extract the rel='next' href from a response links array, or None."""
    for link in response.get("links", []):
        if link.get("rel") == "next" and link.get("href"):
            return link["href"]
    return None


# ── Item parser ────────────────────────────────────────────────────────────────

_SKIP_PROPS = {"datetime", "created", "updated", "start_datetime", "end_datetime"}

def parse_item(
    feature:       dict,
    provider_root: str,
    collection_id: str,
) -> ParsedItem:
    """
    Flatten one GeoJSON feature (STAC Item) into a ParsedItem.
    Extracts metadata and asset hrefs — never downloads binary files.
    """
    props  = feature.get("properties", {}) or {}
    assets = feature.get("assets", {})     or {}

    # Classify assets by role
    data_hrefs: dict[str, str] = {}
    thumbnail_href = ""
    metadata_href  = ""
    all_assets: dict[str, dict] = {}

    for key, asset in assets.items():
        href  = asset.get("href", "")
        roles = asset.get("roles", []) or []
        atype = asset.get("type", "")
        all_assets[key] = {"href": href, "roles": roles, "type": atype}
        if "data" in roles:
            data_hrefs[key] = href
        if "thumbnail" in roles and not thumbnail_href:
            thumbnail_href = href
        if "metadata" in roles and not metadata_href:
            metadata_href = href

    # Geometry
    geom = feature.get("geometry") or {}
    geom_type = geom.get("type", "")

    # Instruments
    instruments = props.get("instruments", []) or []
    if isinstance(instruments, str):
        instruments = [instruments]

    # Properties sample (exclude datetime-like fields)
    props_sample = {
        k: v for k, v in list(props.items())[:20]
        if k not in _SKIP_PROPS
    }

    return ParsedItem(
        item_id        = feature.get("id", ""),
        collection_id  = feature.get("collection", collection_id),
        provider_root  = provider_root,
        datetime       = props.get("datetime", ""),
        start_datetime = props.get("start_datetime", ""),
        end_datetime   = props.get("end_datetime", ""),
        bbox           = feature.get("bbox", []),
        geometry_type  = geom_type,
        cloud_cover    = props.get("eo:cloud_cover"),
        platform       = props.get("platform", ""),
        instruments    = instruments,
        data_hrefs     = data_hrefs,
        thumbnail_href = thumbnail_href,
        metadata_href  = metadata_href,
        all_assets     = all_assets,
        properties     = props_sample,
    )


# ── Paginated search ───────────────────────────────────────────────────────────

def search_items_paginated(
    client:        httpx.Client,
    search_url:    str,
    payload:       dict,
    provider_root: str,
    collection_id: str,
    method:        str = "POST",
    max_pages:     int = DEFAULT_MAX_PAGES,
    max_items:     int = DEFAULT_MAX_ITEMS,
    retries:       int = DEFAULT_RETRIES,
) -> Iterator[ParsedItem]:
    """
    Yield parsed items across pages, following rel='next' links.
    Stops at max_pages pages or max_items items — whichever comes first.
    Never raises — logs errors and stops iteration on failure.
    """
    pages   = 0
    yielded = 0

    response, err = search_first_page(client, search_url, payload, method, retries)
    if response is None:
        log.error("  search failed: %s", err)
        return

    while True:
        pages += 1
        for feature in response.get("features", []):
            yield parse_item(feature, provider_root, collection_id)
            yielded += 1
            if yielded >= max_items:
                log.info("  reached max_items=%d — stopping pagination", max_items)
                return

        next_href = get_next_link(response)
        if not next_href or pages >= max_pages:
            if pages >= max_pages and next_href:
                log.info("  reached max_pages=%d — stopping pagination", max_pages)
            break

        # Follow rel="next" directly — provider constructs the correct next URL
        response, err = _fetch(client, next_href, method="GET", retries=retries)
        if response is None:
            log.error("  pagination error on page %d: %s", pages + 1, err)
            break


# ── Single item fetch ──────────────────────────────────────────────────────────

def get_item(
    client:        httpx.Client,
    provider_root: str,
    collection_id: str,
    item_id:       str,
    retries:       int = DEFAULT_RETRIES,
) -> Optional[ParsedItem]:
    """
    Fetch one specific item by its ID.
    GET {provider_root}/collections/{collection_id}/items/{item_id}
    Returns ParsedItem or None if not found.
    """
    url = f"{provider_root.rstrip('/')}/collections/{collection_id}/items/{item_id}"
    log.info("Fetching item: %s", url)
    data, err = _fetch(client, url, method="GET", retries=retries)
    if data is None:
        log.error("  item fetch failed: %s", err)
        return None
    return parse_item(data, provider_root, collection_id)


# ── Export ─────────────────────────────────────────────────────────────────────

class ExportTooLargeError(Exception):
    pass


def export_items(
    client:        httpx.Client,
    search_url:    str,
    payload:       dict,
    provider_root: str,
    collection_id: str,
    output_path:   str,
    method:        str = "POST",
    max_items:     int = DEFAULT_MAX_ITEMS,
    retries:       int = DEFAULT_RETRIES,
) -> int:
    """
    Export all matching items to a JSONL file safely.

    Steps:
        1. Probe count with limit=1 — get context.matched if available.
        2. If count > max_items: raise ExportTooLargeError (caller asks user).
        3. Paginate with page_size=100 and write each item immediately.
        4. Returns the number of items written.

    Never downloads binary assets — only metadata and hrefs.
    """
    # Step 1 — probe count
    probe_payload = dict(payload)
    probe_payload["limit"] = 1
    probe, err = search_first_page(client, search_url, probe_payload, method, retries)
    if probe is None:
        raise RuntimeError(f"Cannot reach search endpoint: {err}")

    total = extract_count(probe)
    if total is not None:
        log.info("  export: %d total items matched (cap=%d)", total, max_items)

    # Step 2 — paginate and write
    export_payload = dict(payload)
    export_payload["limit"] = PAGE_SIZE_EXPORT

    written = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for item in search_items_paginated(
            client, search_url, export_payload,
            provider_root, collection_id,
            method=method, max_pages=9999, max_items=max_items,
            retries=retries,
        ):
            fh.write(json.dumps(asdict(item), ensure_ascii=False, default=str) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  Exported {written:,} items...", end="\r", flush=True)

    print()
    return written


# ── Main class (public interface) ──────────────────────────────────────────────

class STACItemSearcher:
    """
    Public interface for STAC item search.
    Loads provider index at init — lightweight (~17 records).
    """

    def __init__(
        self,
        providers_path: str = PROVIDERS_PATH,
        timeout:        int = DEFAULT_TIMEOUT,
        retries:        int = DEFAULT_RETRIES,
    ) -> None:
        self.providers = ProviderIndex(providers_path)
        self.timeout   = timeout
        self.retries   = retries

    def search(
        self,
        provider_root:  str,
        collection_id:  str,
        bbox:           Optional[list[float]] = None,
        datetime_range: Optional[str]         = None,
        filters:        Optional[dict]        = None,
        limit:          int                   = DEFAULT_LIMIT,
        sortby:         Optional[list]        = None,
    ) -> SearchResult:
        """
        Search for items — first page only.
        Returns a SearchResult with items, total count (if available), and pagination state.
        """
        s_url   = self.providers.search_url(provider_root)
        s_meth  = self.providers.search_method(provider_root)
        access  = self.providers.access_type(provider_root)

        payload = build_search_payload(
            collection_id, bbox, datetime_range, limit, filters, sortby
        )

        log.info("Searching: %s", s_url)
        log.info("Payload:   %s", json.dumps(payload))

        with make_client(self.timeout, access) as client:
            response, err = search_first_page(
                client, s_url, payload, method=s_meth, retries=self.retries
            )

        if response is None:
            raise RuntimeError(f"Search failed for {s_url}: {err}")

        features  = response.get("features", [])
        total     = extract_count(response)
        next_href = get_next_link(response)
        items     = [parse_item(f, provider_root, collection_id) for f in features]

        return SearchResult(
            items         = items,
            total_matched = total,
            returned      = len(items),
            has_more      = next_href is not None,
            next_href     = next_href,
            provider_root = provider_root,
            collection_id = collection_id,
            search_url    = s_url,
            query_ts      = _now_ts(),
        )

    def count(
        self,
        provider_root:  str,
        collection_id:  str,
        bbox:           Optional[list[float]] = None,
        datetime_range: Optional[str]         = None,
    ) -> Optional[int]:
        """
        Get the total count of matching items with limit=1 (cheapest probe).
        Returns None if the provider does not expose total counts.
        """
        result = self.search(
            provider_root, collection_id,
            bbox=bbox, datetime_range=datetime_range, limit=1
        )
        return result.total_matched

    def get_item(
        self,
        provider_root: str,
        collection_id: str,
        item_id:       str,
    ) -> Optional[ParsedItem]:
        """Fetch one specific item by ID."""
        access = self.providers.access_type(provider_root)
        with make_client(self.timeout, access) as client:
            return get_item(client, provider_root, collection_id, item_id, self.retries)

    def export(
        self,
        provider_root:  str,
        collection_id:  str,
        output_path:    str,
        bbox:           Optional[list[float]] = None,
        datetime_range: Optional[str]         = None,
        filters:        Optional[dict]        = None,
        max_items:      int                   = DEFAULT_MAX_ITEMS,
    ) -> int:
        """Export all matching items to a JSONL file. Returns item count."""
        s_url  = self.providers.search_url(provider_root)
        s_meth = self.providers.search_method(provider_root)
        access = self.providers.access_type(provider_root)
        payload = build_search_payload(
            collection_id, bbox, datetime_range, PAGE_SIZE_EXPORT, filters
        )
        with make_client(self.timeout, access) as client:
            return export_items(
                client, s_url, payload, provider_root, collection_id,
                output_path, method=s_meth, max_items=max_items, retries=self.retries,
            )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _now_ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_item(item: ParsedItem, idx: int) -> str:
    """Format one item for terminal display."""
    lines = [f"\n[{idx}] {item.item_id}"]
    lines.append(f"    Collection : {item.collection_id}")
    if item.datetime:
        lines.append(f"    Date       : {item.datetime}")
    elif item.start_datetime:
        lines.append(f"    Period     : {item.start_datetime} → {item.end_datetime}")
    if item.cloud_cover is not None:
        lines.append(f"    Cloud cover: {item.cloud_cover}%")
    if item.platform:
        lines.append(f"    Platform   : {item.platform}")
    if item.bbox:
        w, s, e, n = item.bbox[:4]
        lines.append(f"    BBox       : W={w:.4f} S={s:.4f} E={e:.4f} N={n:.4f}")
    if item.data_hrefs:
        lines.append(f"    Data assets: {', '.join(item.data_hrefs.keys())}")
        for k, href in list(item.data_hrefs.items())[:2]:
            lines.append(f"      {k}: {href}")
    if item.thumbnail_href:
        lines.append(f"    Thumbnail  : {item.thumbnail_href}")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search real STAC items/products from a live provider API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider",    required=True,
                   help="Provider root URL  e.g. https://stac.terrascope.be")
    p.add_argument("--collection",  required=True,
                   help="Collection ID  e.g. sentinel-2-l2a")
    p.add_argument("--bbox",        default=None,
                   help="Bounding box as 'W,S,E,N'  e.g. '1.30,43.50,1.60,43.75'")
    p.add_argument("--datetime",    default=None,
                   help="Date range  e.g. '2025-06-21/2025-09-22' or single date")
    p.add_argument("--limit",       type=int, default=DEFAULT_LIMIT,
                   help="Max items to display in chat mode")
    p.add_argument("--item-id",     default=None,
                   help="Fetch one specific item by ID")
    p.add_argument("--count-only",  action="store_true",
                   help="Only report the total item count, do not display items")
    p.add_argument("--export",      default=None, metavar="FILE",
                   help="Export all matching items to a JSONL file")
    p.add_argument("--max-items",   type=int, default=DEFAULT_MAX_ITEMS,
                   help="Maximum items to export")
    p.add_argument("--providers-path", default=PROVIDERS_PATH,
                   help="Path to stac_providers.jsonl")
    p.add_argument("--timeout",     type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--retries",     type=int, default=DEFAULT_RETRIES)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    searcher = STACItemSearcher(
        providers_path=args.providers_path,
        timeout=args.timeout,
        retries=args.retries,
    )

    bbox: Optional[list[float]] = None
    if args.bbox:
        try:
            parts = [float(x) for x in args.bbox.split(",")]
            if len(parts) != 4:
                sys.exit("--bbox must have 4 values: W,S,E,N")
            bbox = parts
        except ValueError:
            sys.exit("--bbox values must be numbers")

    # ── Fetch one item by ID ────────────────────────────────────────────────
    if args.item_id:
        print(f"\nFetching item: {args.item_id}")
        item = searcher.get_item(args.provider, args.collection, args.item_id)
        if item is None:
            print("  Item not found.")
        else:
            print(_fmt_item(item, 1))
        return

    # ── Count only ──────────────────────────────────────────────────────────
    if args.count_only:
        print(f"\nCounting items in '{args.collection}' at {args.provider}")
        if bbox:
            print(f"  bbox:     {bbox}")
        if args.datetime:
            print(f"  datetime: {args.datetime}")
        total = searcher.count(args.provider, args.collection, bbox, args.datetime)
        if total is not None:
            print(f"\n  Total matching items: {total:,}")
        else:
            print("\n  This provider does not expose a total count.")
            print("  Run without --count-only to see the first page of results.")
        return

    # ── Export ──────────────────────────────────────────────────────────────
    if args.export:
        print(f"\nExporting items from '{args.collection}' to {args.export}")
        if bbox:
            print(f"  bbox:      {bbox}")
        if args.datetime:
            print(f"  datetime:  {args.datetime}")
        try:
            n = searcher.export(
                args.provider, args.collection, args.export,
                bbox=bbox, datetime_range=args.datetime, max_items=args.max_items,
            )
            print(f"\n  Exported {n:,} items to {args.export}")
        except ExportTooLargeError as e:
            print(f"\n  Export blocked: {e}")
        except RuntimeError as e:
            print(f"\n  Error: {e}")
        return

    # ── Search and display ──────────────────────────────────────────────────
    print(f"\nSearching '{args.collection}' at {args.provider}")
    s_url = searcher.providers.search_url(args.provider)
    print(f"  Search URL : {s_url}")
    if bbox:
        print(f"  bbox       : {bbox}")
    if args.datetime:
        print(f"  datetime   : {args.datetime}")
    print(f"  limit      : {args.limit}")

    try:
        result = searcher.search(
            args.provider, args.collection,
            bbox=bbox, datetime_range=args.datetime, limit=args.limit,
        )
    except RuntimeError as e:
        print(f"\n  Search failed: {e}")
        sys.exit(1)

    print()
    if result.total_matched is not None:
        print(f"Total matching items : {result.total_matched:,}")
    else:
        print("Total matching items : unknown (provider does not expose count)")
    print(f"Returned this page   : {result.returned}")
    print(f"More pages available : {'Yes' if result.has_more else 'No'}")

    if not result.items:
        print("\nNo items found for this query.")
        return

    for i, item in enumerate(result.items, 1):
        print(_fmt_item(item, i))

    if result.has_more:
        print(f"\n  ... more items available.")
        print(f"  Use --limit to increase page size, or --export FILE to save all.")


if __name__ == "__main__":
    main()
