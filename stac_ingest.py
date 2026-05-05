#!/usr/bin/env python3
"""
stac_ingest.py — STAC Provider Crawler for RAG Pipelines
=========================================================
Reads a CSV of STAC provider root URLs, crawls each provider via its
STAC root catalog links, extracts normalized provider + collection
metadata, optionally samples a few items per collection, and writes
clean JSONL outputs ready for a RAG pipeline.

No HTML scraping. No pystac. No imports from project code. Standalone.
Uses httpx to call STAC endpoints directly and parses raw JSON payloads.

Dependencies: httpx  (pip install httpx)

Outputs (in --output-dir, default kb/outputs/):
  stac_providers.jsonl      — one record per crawled provider root
  stac_collections.jsonl    — one record per unique collection
  stac_items_sample.jsonl   — sampled item metadata per collection
  stac_rag_documents.jsonl  — text + metadata ready for vector indexing
  stac_ingest_errors.jsonl  — structured per-request errors
  stac_ingest_summary.json  — run statistics

Auth note: httpx can pass a Bearer token via --access-column for providers
that require authentication. Assets behind auth will still expose public
metadata; protected binary assets are not downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stac_ingest")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl STAC providers from a CSV and write JSONL outputs for RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, metavar="FILE",
                   help="CSV file with STAC provider root URLs")
    p.add_argument("--output-dir", default="kb/outputs", metavar="DIR",
                   help="Directory where JSONL outputs are written")
    p.add_argument("--url-column", default=None, metavar="COL",
                   help="CSV column name containing provider root URLs "
                        "(auto-detected when omitted)")
    p.add_argument("--access-column", default=None, metavar="COL",
                   help="CSV column for access type / Bearer token (optional)")
    p.add_argument("--sample-items", type=int, default=3, metavar="N",
                   help="Number of example items to sample per collection (0 = skip)")
    p.add_argument("--max-collections", type=int, default=None, metavar="N",
                   help="Stop after N collections per provider (None = all)")
    p.add_argument("--timeout", type=int, default=60, metavar="S",
                   help="HTTP request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, metavar="N",
                   help="Retry attempts per failed request (exponential backoff)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate CSV and URLs only — no HTTP crawling")
    p.add_argument("--resume", action="store_true",
                   help="Skip providers already present in stac_providers.jsonl "
                        "and append to existing output files instead of overwriting")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging")
    return p.parse_args()


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class ProviderRecord:
    """Root catalog metadata for one STAC provider."""
    provider_root: str
    catalog_id: str = ""
    title: str = ""
    description: str = ""
    stac_version: str = ""
    conforms_to: list = field(default_factory=list)
    collections_url: str = ""
    search_url: str = ""
    search_method: str = ""       # "GET" or "POST"
    queryables_url: str = ""
    access_type: str = ""
    crawl_ts: str = ""


@dataclass
class CollectionRecord:
    """Normalized metadata for one STAC collection."""
    provider_root: str
    collection_id: str
    title: str = ""
    description: str = ""
    keywords: list = field(default_factory=list)
    license: str = ""
    extent_spatial: list = field(default_factory=list)    # [[W,S,E,N], …]
    extent_temporal: list = field(default_factory=list)   # [[start, end], …]
    providers: list = field(default_factory=list)
    item_asset_names: list = field(default_factory=list)
    item_asset_roles: dict = field(default_factory=dict)  # name → roles list
    item_asset_types: dict = field(default_factory=dict)  # name → media type
    collection_asset_names: list = field(default_factory=list)
    summaries: dict = field(default_factory=dict)
    queryables: dict = field(default_factory=dict)
    platforms: list = field(default_factory=list)
    instruments: list = field(default_factory=list)
    bands: list = field(default_factory=list)
    stac_version: str = ""
    stac_extensions: list = field(default_factory=list)
    raw_collection_hash: str = ""
    crawl_ts: str = ""


@dataclass
class ItemSample:
    """Sampled item metadata — asset hrefs, roles, types and scalar properties."""
    provider_root: str
    collection_id: str
    item_id: str
    datetime: str = ""
    geometry_type: str = ""
    bbox: list = field(default_factory=list)
    asset_keys: list = field(default_factory=list)
    asset_hrefs: dict = field(default_factory=dict)   # key → href
    asset_roles: dict = field(default_factory=dict)   # key → [roles]
    asset_types: dict = field(default_factory=dict)   # key → media_type
    properties_sample: dict = field(default_factory=dict)
    crawl_ts: str = ""


@dataclass
class RagDocument:
    """Text document built from collection metadata for vector indexing."""
    doc_id: str
    provider_root: str
    collection_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestError:
    """Structured error record written per failed request."""
    stage: str
    url: str
    provider_root: str
    collection_id: str
    exception: str
    ts: str = ""


# ── JSONL writer ───────────────────────────────────────────────────────────────

class JsonlWriter:
    """Incremental JSONL writer — flushes after every record."""

    def __init__(self, path: Path, mode: str = "w") -> None:
        self.path = path
        self._fh = open(path, mode, encoding="utf-8")
        self.count = 0

    def write(self, obj: Any) -> None:
        if hasattr(obj, "__dataclass_fields__"):
            obj = asdict(obj)
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── CSV parsing ────────────────────────────────────────────────────────────────

_URL_COLUMN_HINTS = [
    "url", "root_url", "stac_url", "stac_root", "catalog_url",
    "provider_url", "api_url", "endpoint", "link", "href", "root",
]


def _clean_url(raw: Any) -> str:
    """
    Normalize a raw URL cell.
    Per spec: keep substring before first comma (handles 'https://…, Notes'),
    strip whitespace/quotes, then validate scheme + netloc via urllib.parse.
    Returns "" for anything invalid (including None/empty).
    """
    if raw is None:
        return ""
    url = str(raw).strip().strip("\"'")
    if not url:
        return ""
    if "," in url:
        url = url.split(",", 1)[0].strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return url


def _detect_url_column(fieldnames: list[str]) -> Optional[str]:
    lower_map = {f.lower(): f for f in fieldnames}
    for hint in _URL_COLUMN_HINTS:
        if hint in lower_map:
            return lower_map[hint]
    for f in fieldnames:
        if any(kw in f.lower() for kw in ("url", "link", "href", "endpoint", "stac")):
            return f
    return None


def load_csv(
    path: Path,
    url_column: Optional[str],
    access_column: Optional[str],
) -> list[dict]:
    """
    Parse CSV safely using csv.DictReader (header present) or csv.reader
    (no header). Never uses line.split(',').
    Returns list of {"url", "access", "row"} dicts.
    """
    records: list[dict] = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True

        if has_header:
            reader = csv.DictReader(fh, dialect=dialect)
            fieldnames: list[str] = list(reader.fieldnames or [])

            ucol = url_column
            if ucol and ucol not in fieldnames:
                log.warning("--url-column %r not in CSV headers; auto-detecting", ucol)
                ucol = None
            if not ucol:
                ucol = _detect_url_column(fieldnames)
            if not ucol and fieldnames:
                ucol = fieldnames[0]
                log.warning("URL column not detected; using first column: %r", ucol)
            if not ucol:
                log.error("Cannot determine URL column (empty headers)")
                return records

            acol = access_column
            if acol and acol not in fieldnames:
                log.warning("--access-column %r not found; ignoring", acol)
                acol = None

            log.info("CSV  url_column=%r  access_column=%r  headers=%s",
                     ucol, acol, fieldnames)

            for row in reader:
                raw_url = row.get(ucol, "")
                url = _clean_url(raw_url)
                if not url:
                    log.debug("Skipping row — invalid URL: %r", raw_url)
                    continue
                records.append({
                    "url":    url,
                    "access": row.get(acol, "").strip() if acol else "",
                    "row":    dict(row),
                })

        else:
            # No header — first column = URL, second (optional) = access
            reader = csv.reader(fh, dialect=dialect)
            for row in reader:
                if not row:
                    continue
                url = _clean_url(row[0])
                if not url:
                    continue
                records.append({
                    "url":    url,
                    "access": row[1].strip() if len(row) > 1 else "",
                    "row":    {},
                })

    log.info("Loaded %d valid provider URLs from %s", len(records), path)
    return records


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def make_client(timeout: int, access: str = "") -> httpx.Client:
    """
    Build an httpx.Client with JSON accept headers and optional Bearer auth.
    follow_redirects=True handles providers that redirect / to /stac etc.
    Non-trivial access values (not 'open'/'public'/'free') are sent as Bearer.
    """
    headers: dict[str, str] = {
        "Accept": "application/json, application/geo+json",
        "User-Agent": "stac-ingest/1.0 (+github.com/dataionics)",
    }
    stripped = access.strip().lower()
    if access.strip() and stripped not in ("open", "public", "free", "none", ""):
        headers["Authorization"] = f"Bearer {access.strip()}"
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers=headers,
    )


def fetch_json(
    client: httpx.Client,
    url: str,
    retries: int,
    method: str = "GET",
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> tuple[Optional[Any], Optional[str]]:
    """
    Fetch a URL and return (parsed_json, error_str).
    - Retries with exponential backoff (2s → 4s → 8s … capped at 30s).
    - 4xx (except 429) are not retried — they indicate a structural problem.
    - HTML responses are rejected immediately (not a STAC endpoint).
    - Never raises; caller decides how to handle (None, error_str).
    """
    delay = 2.0
    last_err: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            if method == "POST":
                r = client.post(url, json=json_body or {}, params=params)
            else:
                r = client.get(url, params=params)

            ct = r.headers.get("content-type", "")
            if "text/html" in ct and r.status_code == 200:
                return None, "HTML response — not a STAC JSON endpoint"

            if r.status_code == 200:
                try:
                    return r.json(), None
                except Exception as exc:
                    return None, f"JSON decode error: {exc}"

            last_err = f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 404, 405, 410):
                break  # structural — no retry
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", str(int(delay))))
                time.sleep(min(wait, 60))
                continue

        except httpx.TimeoutException as exc:
            last_err = f"Timeout: {exc}"
        except httpx.RequestError as exc:
            last_err = f"Request error: {exc}"
        except Exception as exc:
            last_err = f"Unexpected: {exc}"

        if attempt < retries:
            log.debug("  retry %d/%d for %s (%s)", attempt + 1, retries, url, last_err)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    return None, last_err


# ── Link helpers ───────────────────────────────────────────────────────────────

def _links_for_rels(links: Any, rels: set[str]) -> list[dict]:
    """Return all link dicts whose 'rel' is in the given set."""
    if not isinstance(links, list):
        return []
    return [lk for lk in links if isinstance(lk, dict) and lk.get("rel", "") in rels]


def _first_href(links: Any, rels: set[str], base: str = "") -> str:
    """Return the href of the first matching link, resolved against base."""
    for lk in _links_for_rels(links, rels):
        href = lk.get("href", "")
        if href:
            return urllib.parse.urljoin(base, href) if base else href
    return ""


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _write_err(
    writer: JsonlWriter,
    stage: str,
    url: str,
    provider_root: str,
    collection_id: str,
    exception: str,
) -> None:
    writer.write(IngestError(
        stage=stage, url=url, provider_root=provider_root,
        collection_id=collection_id, exception=str(exception), ts=_now_ts(),
    ))


# ── Provider root crawling ─────────────────────────────────────────────────────

_QUERYABLES_RELS: set[str] = {
    "queryables",
    "http://www.opengis.net/def/rel/ogc/1.0/queryables",
}


def crawl_provider(
    client: httpx.Client,
    provider_url: str,
    access: str,
    retries: int,
    err_writer: JsonlWriter,
) -> Optional[ProviderRecord]:
    """
    GET the STAC root catalog JSON and extract provider-level metadata.

    Discovers:
      - collections endpoint via rel="data" | "collections"
      - search endpoint via rel="search" (prefers POST if available)
      - queryables endpoint via rel="queryables"
      - Fallback: append /collections to root if no collections link found

    NASA CMR-STAC: each provider-scoped URL (e.g. .../stac/LPCLOUD) is
    treated as the root. We never navigate up to a global search endpoint
    outside the provider's path scope.
    """
    data, err = fetch_json(client, provider_url, retries)
    if err or not isinstance(data, dict):
        _write_err(err_writer, "provider_root", provider_url,
                   provider_url, "", err or "non-dict response")
        log.warning("Provider root failed %s: %s", provider_url, err)
        return None

    links: list = data.get("links", []) if isinstance(data.get("links"), list) else []
    parsed_root = urllib.parse.urlparse(provider_url)

    # ── Collections URL ────────────────────────────────────────────────────
    collections_url = ""
    for lk in _links_for_rels(links, {"data", "collections", "child"}):
        href = lk.get("href", "")
        if not href:
            continue
        full = urllib.parse.urljoin(provider_url, href)
        if any(seg in full.lower() for seg in ["/collections", "/datasets", "/data"]):
            collections_url = full
            break
    if not collections_url:
        collections_url = provider_url.rstrip("/") + "/collections"

    # ── Search URL (scoped to this provider's origin) ──────────────────────
    search_url = ""
    search_method = "GET"
    for lk in _links_for_rels(links, {"search"}):
        href = lk.get("href", "")
        if not href:
            continue
        full = urllib.parse.urljoin(provider_url, href)
        # Reject cross-origin search (would leave the provider scope)
        if urllib.parse.urlparse(full).netloc != parsed_root.netloc:
            continue
        method = lk.get("method", "GET").upper()
        if method == "POST":
            search_url = full
            search_method = "POST"
            break
        elif not search_url:
            search_url = full
            search_method = method

    # ── Queryables URL ─────────────────────────────────────────────────────
    queryables_url = _first_href(links, _QUERYABLES_RELS, provider_url)

    return ProviderRecord(
        provider_root=provider_url,
        catalog_id=str(data.get("id", "")),
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        stac_version=str(data.get("stac_version", "")),
        conforms_to=(data.get("conformsTo", [])
                     if isinstance(data.get("conformsTo"), list) else []),
        collections_url=collections_url,
        search_url=search_url,
        search_method=search_method,
        queryables_url=queryables_url,
        access_type=access,
        crawl_ts=_now_ts(),
    )


# ── Queryables ─────────────────────────────────────────────────────────────────

def fetch_queryables(
    client: httpx.Client,
    url: str,
    retries: int,
    err_writer: JsonlWriter,
    provider_root: str,
    collection_id: str = "",
) -> dict:
    """
    Fetch a /queryables endpoint. Returns {} on any failure and continues.
    Never aborts the run.
    """
    if not url:
        return {}
    data, err = fetch_json(client, url, retries)
    if err or not isinstance(data, dict):
        if err:
            _write_err(err_writer, "queryables", url, provider_root, collection_id, err)
        log.debug("  queryables unavailable (%s): %s", collection_id or "provider", err)
        return {}
    return data


def _compact_queryables(raw: dict) -> dict:
    """
    Build a compact {property_name: label} map from a JSON-Schema queryables
    document. The full schema can be huge — we keep names + titles only,
    capped at 60 properties to bound memory.
    """
    props = raw.get("properties", raw)
    if not isinstance(props, dict):
        return {}
    result: dict[str, str] = {}
    for k, v in list(props.items())[:60]:
        if isinstance(v, dict):
            label = v.get("title") or v.get("type") or k
        else:
            label = str(v)
        result[str(k)] = str(label)
    return result


# ── Collection normalization ───────────────────────────────────────────────────

def _as_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if val is not None:
        return [val]
    return []


def _extract_platforms_instruments(summaries: dict) -> tuple[list[str], list[str]]:
    platforms: list[str] = []
    instruments: list[str] = []
    for key in ("platform", "platforms", "eo:platform",
                "sat:platform_international_designator"):
        v = summaries.get(key)
        if isinstance(v, list):
            platforms.extend(str(x) for x in v)
        elif isinstance(v, str):
            platforms.append(v)
    for key in ("instrument", "instruments", "eo:instrument"):
        v = summaries.get(key)
        if isinstance(v, list):
            instruments.extend(str(x) for x in v)
        elif isinstance(v, str):
            instruments.append(v)
    return list(dict.fromkeys(platforms)), list(dict.fromkeys(instruments))


def _extract_bands(summaries: dict, item_assets: dict) -> list[str]:
    """Extract band names from summaries['eo:bands'] or item_assets."""
    bands: list[str] = []
    for key in ("eo:bands", "raster:bands"):
        for entry in summaries.get(key, []):
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("common_name") or ""
                if name:
                    bands.append(str(name))
            elif isinstance(entry, str):
                bands.append(entry)
    if not bands and isinstance(item_assets, dict):
        for asset_val in list(item_assets.values())[:20]:
            if not isinstance(asset_val, dict):
                continue
            for b in asset_val.get("eo:bands", []):
                if isinstance(b, dict):
                    name = b.get("name") or b.get("common_name") or ""
                    if name:
                        bands.append(str(name))
    return list(dict.fromkeys(bands))


def _extract_item_asset_meta(
    item_assets: dict,
) -> tuple[list[str], dict[str, list], dict[str, str]]:
    """
    From item_assets dict return:
      names  — ordered list of asset keys
      roles  — {key: [role, …]}
      types  — {key: media_type}
    """
    names: list[str] = []
    roles: dict[str, list] = {}
    types: dict[str, str] = {}
    if not isinstance(item_assets, dict):
        return names, roles, types
    for key, val in list(item_assets.items())[:60]:
        names.append(str(key))
        if isinstance(val, dict):
            r = val.get("roles", [])
            roles[key] = r if isinstance(r, list) else [r] if r else []
            t = val.get("type", "") or val.get("media_type", "")
            if t:
                types[key] = str(t)
    return names, roles, types


_EXCLUDE_SUMMARY_KEYS = {
    "eo:bands", "raster:bands", "bands",
    "platform", "platforms", "eo:platform",
    "instrument", "instruments", "eo:instrument",
}


def _normalize_collection(
    data: dict,
    provider_root: str,
    provider_queryables: dict,
    collection_queryables: dict,
) -> CollectionRecord:
    """Build a CollectionRecord from a raw /collections/{id} JSON payload."""
    cid = str(data.get("id", ""))

    extent = data.get("extent") or {}
    if not isinstance(extent, dict):
        extent = {}
    spatial = extent.get("spatial") or {}
    temporal = extent.get("temporal") or {}
    bbox = spatial.get("bbox", []) if isinstance(spatial, dict) else []
    interval = temporal.get("interval", []) if isinstance(temporal, dict) else []

    summaries = data.get("summaries") or {}
    if not isinstance(summaries, dict):
        summaries = {}

    platforms, instruments = _extract_platforms_instruments(summaries)

    raw_item_assets = data.get("item_assets") or {}
    if not isinstance(raw_item_assets, dict):
        raw_item_assets = {}
    bands = _extract_bands(summaries, raw_item_assets)
    ia_names, ia_roles, ia_types = _extract_item_asset_meta(raw_item_assets)

    raw_assets = data.get("assets") or {}
    col_asset_names = list(raw_assets.keys())[:30] if isinstance(raw_assets, dict) else []

    # Compact summaries — exclude already-extracted fields
    compact_summaries: dict[str, Any] = {}
    for k, v in list(summaries.items())[:30]:
        if k in _EXCLUDE_SUMMARY_KEYS:
            continue
        if isinstance(v, list) and len(v) <= 20:
            compact_summaries[k] = v
        elif isinstance(v, (str, int, float, bool)):
            compact_summaries[k] = v

    # Queryables: collection-specific wins; fall back to provider level
    effective_q = collection_queryables or provider_queryables or {}
    compact_q = _compact_queryables(effective_q)

    providers_raw = data.get("providers") or []
    if not isinstance(providers_raw, list):
        providers_raw = []

    keywords = _as_list(data.get("keywords") or [])

    raw_hash = _md5(json.dumps(data, sort_keys=True, ensure_ascii=False))

    return CollectionRecord(
        provider_root=provider_root,
        collection_id=cid,
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        keywords=[str(k) for k in keywords[:60]],
        license=str(data.get("license", "")),
        extent_spatial=bbox if isinstance(bbox, list) else [],
        extent_temporal=interval if isinstance(interval, list) else [],
        providers=providers_raw[:20],
        item_asset_names=ia_names,
        item_asset_roles=ia_roles,
        item_asset_types=ia_types,
        collection_asset_names=col_asset_names,
        summaries=compact_summaries,
        queryables=compact_q,
        platforms=platforms[:20],
        instruments=instruments[:20],
        bands=bands[:60],
        stac_version=str(data.get("stac_version", "")),
        stac_extensions=_as_list(data.get("stac_extensions") or [])[:20],
        raw_collection_hash=raw_hash,
        crawl_ts=_now_ts(),
    )


# ── Collection crawling ────────────────────────────────────────────────────────

def crawl_collections(
    client: httpx.Client,
    provider: ProviderRecord,
    retries: int,
    max_collections: Optional[int],
    err_writer: JsonlWriter,
) -> Iterator[CollectionRecord]:
    """
    Paginate /collections and yield normalized CollectionRecord objects.

    Flow per collection:
      1. List page   → get summary rows + per-row links
      2. rel="self"  → fetch full collection detail (fallback: collections_url/id)
      3. queryables  → fetch collection-specific queryables (fallback: provider level)
      4. Normalize   → CollectionRecord

    Pagination follows rel="next" links. Stops when no next link is present
    or when max_collections is reached. Memory is bounded — yields one at a time.
    """
    # Fetch provider-level queryables once; reused as fallback for all collections
    provider_queryables: dict = {}
    if provider.queryables_url:
        provider_queryables = fetch_queryables(
            client, provider.queryables_url, retries,
            err_writer, provider.provider_root,
        )
        if provider_queryables:
            log.debug("  provider queryables: %d properties",
                      len(_compact_queryables(provider_queryables)))

    seen_ids: set[str] = set()
    total = 0
    page_url: Optional[str] = provider.collections_url

    while page_url:
        data, err = fetch_json(client, page_url, retries)
        if err or not isinstance(data, dict):
            _write_err(err_writer, "collections_page", page_url,
                       provider.provider_root, "", err or "non-dict response")
            log.warning("  collections page error %s: %s", page_url, err)
            break

        # STAC /collections wraps items under "collections"; some APIs return a list
        raw_items = data.get("collections") or data.get("items") or []
        if not isinstance(raw_items, list):
            log.warning("  unexpected /collections shape at %s", page_url)
            break

        if not raw_items:
            log.debug("  empty page at %s", page_url)

        for raw_col in raw_items:
            if not isinstance(raw_col, dict):
                continue
            cid = str(raw_col.get("id", ""))
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)

            col_links = (raw_col.get("links", [])
                         if isinstance(raw_col.get("links"), list) else [])

            # ── Full collection detail ─────────────────────────────────────
            self_href = _first_href(col_links, {"self", "canonical"}, page_url)
            if not self_href:
                self_href = provider.collections_url.rstrip("/") + "/" + cid

            full_data, err = fetch_json(client, self_href, retries)
            if err or not isinstance(full_data, dict):
                if err:
                    _write_err(err_writer, "collection_detail", self_href,
                               provider.provider_root, cid, err)
                    log.debug("  collection detail fallback for %s: %s", cid, err)
                full_data = raw_col   # use summary row as fallback

            detail_links = (full_data.get("links", [])
                            if isinstance(full_data.get("links"), list) else [])

            # ── Collection-level queryables ────────────────────────────────
            col_q_url = _first_href(detail_links, _QUERYABLES_RELS, self_href)
            col_queryables: dict = {}
            if col_q_url:
                col_queryables = fetch_queryables(
                    client, col_q_url, retries,
                    err_writer, provider.provider_root, cid,
                )

            rec = _normalize_collection(
                full_data, provider.provider_root,
                provider_queryables, col_queryables,
            )
            yield rec
            total += 1

            if max_collections and total >= max_collections:
                log.info("  reached --max-collections %d; stopping", max_collections)
                return

        # ── Pagination via rel="next" ──────────────────────────────────────
        page_links = data.get("links", []) if isinstance(data.get("links"), list) else []
        next_href = _first_href(page_links, {"next"}, page_url)
        if next_href and next_href != page_url:
            log.debug("  next page → %s", next_href)
            page_url = next_href
        else:
            page_url = None


# ── Item sampling ──────────────────────────────────────────────────────────────

def sample_items(
    client: httpx.Client,
    provider: ProviderRecord,
    collection_id: str,
    n: int,
    retries: int,
    err_writer: JsonlWriter,
    collection_detail_links: Optional[list] = None,
) -> list[ItemSample]:
    """
    Fetch up to n item examples for a collection.  Strategy in order:
      1. POST /search  {"collections":[id], "limit":n}   — preferred
      2. GET  /search  ?collections=id&limit=n
      3. GET  /collections/{id}/items  ?limit=n          — always tried last

    Stops at n items — never pages through a full collection.
    Captures: item id, datetime, geometry type, bbox, asset keys/hrefs/
    roles/media_types, and scalar properties (capped at 25 fields).
    """
    def _parse_features(data: Any) -> list[ItemSample]:
        if not isinstance(data, dict):
            return []
        return [s for feat in data.get("features", [])[:n]
                if (s := _parse_item(feat, provider.provider_root, collection_id))]

    # Attempt 1 — POST /search
    if provider.search_url and provider.search_method == "POST":
        d, err = fetch_json(
            client, provider.search_url, retries,
            method="POST",
            json_body={"collections": [collection_id], "limit": n},
        )
        if not err:
            items = _parse_features(d)
            if items:
                return items

    # Attempt 2 — GET /search
    if provider.search_url:
        d, err = fetch_json(
            client, provider.search_url, retries,
            params={"collections": collection_id, "limit": n},
        )
        if not err:
            items = _parse_features(d)
            if items:
                return items

    # Attempt 3 — GET /collections/{id}/items
    items_url = ""
    if collection_detail_links:
        items_url = _first_href(collection_detail_links, {"items"},
                                provider.collections_url)
    if not items_url:
        items_url = provider.collections_url.rstrip("/") + "/" + collection_id + "/items"

    d, err = fetch_json(client, items_url, retries, params={"limit": n})
    if err:
        _write_err(err_writer, "items_sample", items_url,
                   provider.provider_root, collection_id, err)
        log.debug("  items sample failed %s: %s", collection_id, err)
        return []

    return _parse_features(d)


def _parse_item(
    feat: dict, provider_root: str, collection_id: str,
) -> Optional[ItemSample]:
    """
    Parse one GeoJSON Feature (STAC Item) into an ItemSample.
    Captures: datetime, geometry type, bbox, asset hrefs/roles/types,
    and scalar properties. Never downloads binary asset content.
    """
    if not isinstance(feat, dict):
        return None
    item_id = str(feat.get("id", ""))
    if not item_id:
        return None

    props = feat.get("properties") or {}
    if not isinstance(props, dict):
        props = {}

    dt = (props.get("datetime") or props.get("start_datetime")
          or props.get("end_datetime") or "")

    geom = feat.get("geometry") or {}
    geom_type = geom.get("type", "") if isinstance(geom, dict) else ""
    bbox = feat.get("bbox", [])
    if not isinstance(bbox, list):
        bbox = []

    # Asset metadata: href, roles, media type — no binary download
    raw_assets = feat.get("assets") or {}
    if not isinstance(raw_assets, dict):
        raw_assets = {}
    asset_keys: list[str] = []
    asset_hrefs: dict[str, str] = {}
    asset_roles: dict[str, list] = {}
    asset_types: dict[str, str] = {}
    for key, val in list(raw_assets.items())[:40]:
        asset_keys.append(str(key))
        if not isinstance(val, dict):
            continue
        href = val.get("href", "") or val.get("url", "")
        if href:
            asset_hrefs[key] = str(href)
        r = val.get("roles", [])
        asset_roles[key] = r if isinstance(r, list) else [r] if r else []
        t = val.get("type", "") or val.get("media_type", "")
        if t:
            asset_types[key] = str(t)

    # Scalar properties only (no nested objects or large arrays)
    safe_props: dict[str, Any] = {}
    skip_keys = {"datetime", "created", "updated", "start_datetime", "end_datetime"}
    for k, v in list(props.items())[:30]:
        if k in skip_keys:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe_props[str(k)] = v
        if len(safe_props) >= 25:
            break

    return ItemSample(
        provider_root=provider_root,
        collection_id=collection_id,
        item_id=item_id,
        datetime=str(dt) if dt else "",
        geometry_type=geom_type,
        bbox=bbox[:6],
        asset_keys=asset_keys,
        asset_hrefs=asset_hrefs,
        asset_roles=asset_roles,
        asset_types=asset_types,
        properties_sample=safe_props,
        crawl_ts=_now_ts(),
    )


# ── RAG document builder ───────────────────────────────────────────────────────

def build_rag_document(col: CollectionRecord) -> RagDocument:
    """
    Build a human-readable text document from collection metadata.
    Every sentence is sourced directly from a structured field — no invented facts.
    Text is structured for dense retrieval (BM25 + vector).
    """
    lines: list[str] = []

    if col.title:
        lines.append(f"Collection: {col.title}")
    if col.collection_id:
        lines.append(f"ID: {col.collection_id}")
    if col.description:
        lines.append(f"Description: {col.description}")
    if col.keywords:
        lines.append(f"Keywords: {', '.join(col.keywords)}")
    if col.license:
        lines.append(f"License: {col.license}")

    if col.platforms:
        lines.append(f"Platforms / missions: {', '.join(col.platforms)}")
    if col.instruments:
        lines.append(f"Instruments / sensors: {', '.join(col.instruments)}")
    if col.bands:
        lines.append(f"Spectral bands: {', '.join(col.bands)}")

    # Provider organization names + roles
    if col.providers:
        pnames: list[str] = []
        for pv in col.providers:
            if not isinstance(pv, dict):
                continue
            name = pv.get("name", "")
            roles = pv.get("roles") or []
            url = pv.get("url", "")
            if name:
                suffix = f" [{', '.join(roles)}]" if roles else ""
                pnames.append(f"{name}{suffix}")
        if pnames:
            lines.append(f"Data providers: {'; '.join(pnames)}")

    # Temporal extent
    for interval in col.extent_temporal:
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            start = interval[0] or "open"
            end   = interval[1] or "present"
            lines.append(f"Temporal coverage: {start} to {end}")

    # Spatial extent
    for bbox in col.extent_spatial:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            w, s, e, n = bbox[0], bbox[1], bbox[2], bbox[3]
            lines.append(f"Spatial extent (W S E N): {w} {s} {e} {n}")

    # Item assets — names, roles, and media types
    if col.item_asset_names:
        # Compact one-liner per asset with role and type if known
        asset_descs: list[str] = []
        for name in col.item_asset_names[:30]:
            parts = [name]
            roles = col.item_asset_roles.get(name, [])
            if roles:
                parts.append(f"roles=[{', '.join(roles)}]")
            mtype = col.item_asset_types.get(name, "")
            if mtype:
                parts.append(f"type={mtype}")
            asset_descs.append(" ".join(parts))
        lines.append(f"Item assets: {'; '.join(asset_descs)}")

    if col.collection_asset_names:
        lines.append(f"Collection assets: {', '.join(col.collection_asset_names[:15])}")

    # Queryable property labels
    if col.queryables:
        q_labels = list(col.queryables.values())[:20]
        lines.append(f"Queryable properties: {', '.join(str(q) for q in q_labels)}")

    # STAC extensions (signal supported feature sets)
    if col.stac_extensions:
        lines.append(f"STAC extensions: {', '.join(col.stac_extensions)}")

    # Compact summaries (scalar and short-list of scalars only — no nested dicts)
    for k, v in list(col.summaries.items())[:10]:
        if isinstance(v, list) and 1 <= len(v) <= 10:
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                lines.append(f"{k}: {', '.join(str(x) for x in v)}")
        elif isinstance(v, (str, int, float, bool)):
            lines.append(f"{k}: {v}")

    lines.append(f"Provider root: {col.provider_root}")

    doc_id = _md5(f"{col.provider_root}::{col.collection_id}")
    text = "\n".join(lines)

    metadata = {
        "provider_root":       col.provider_root,
        "collection_id":       col.collection_id,
        "title":               col.title,
        "keywords":            col.keywords,
        "platforms":           col.platforms,
        "instruments":         col.instruments,
        "bands":               col.bands,
        "license":             col.license,
        "extent_temporal":     col.extent_temporal,
        "extent_spatial":      col.extent_spatial,
        "stac_extensions":     col.stac_extensions,
        "item_asset_names":    col.item_asset_names,
        "raw_collection_hash": col.raw_collection_hash,
        "crawl_ts":            col.crawl_ts,
    }

    return RagDocument(
        doc_id=doc_id,
        provider_root=col.provider_root,
        collection_id=col.collection_id,
        text=text,
        metadata=metadata,
    )


# ── Main orchestration ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path, args.url_column, args.access_column)
    if not rows:
        sys.exit("No valid provider URLs found in CSV.")

    # Deduplicate provider root URLs — the CSV may have one row per collection
    # (same root URL repeated for each collection it contains).  We crawl each
    # unique root exactly once and discover collections via the API.
    #
    # Canonical form: strip trailing slash so that
    #   https://stac.terrascope.be   and
    #   https://stac.terrascope.be/
    # are treated as the same provider.  The canonical (no-slash) form is used
    # for crawling so httpx doesn't waste a redirect round-trip.
    def _canonical(url: str) -> str:
        return url.rstrip("/")

    seen_roots: dict[str, str] = {}   # canonical_url → access
    for r in rows:
        key = _canonical(r["url"])
        if key not in seen_roots:
            seen_roots[key] = r["access"]
    unique_rows = [{"url": u, "access": a} for u, a in seen_roots.items()]
    if len(unique_rows) < len(rows):
        log.info("Deduplicated %d CSV rows → %d unique provider roots",
                 len(rows), len(unique_rows))
    rows = unique_rows

    # ── Dry-run ────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("Dry run — %d unique provider roots validated, no crawling", len(rows))
        for r in rows:
            log.info("  ✓  %s  [access=%r]", r["url"], r["access"] or "open")
        log.info("Pass without --dry-run to begin crawling.")
        return

    # ── Resume: read already-crawled providers ─────────────────────────────
    done_providers: set[str] = set()
    if args.resume:
        prov_file = output_dir / "stac_providers.jsonl"
        if prov_file.exists():
            with open(prov_file, encoding="utf-8") as _pf:
                for _line in _pf:
                    _line = _line.strip()
                    if _line:
                        _rec = json.loads(_line)
                        done_providers.add(_canonical(_rec.get("provider_root", "")))
            log.info("--resume: %d providers already done, will skip them.",
                     len(done_providers))
        file_mode = "a"
    else:
        file_mode = "w"

    # ── Open output writers ────────────────────────────────────────────────
    prov_w = JsonlWriter(output_dir / "stac_providers.jsonl", file_mode)
    col_w  = JsonlWriter(output_dir / "stac_collections.jsonl", file_mode)
    item_w = JsonlWriter(output_dir / "stac_items_sample.jsonl", file_mode)
    rag_w  = JsonlWriter(output_dir / "stac_rag_documents.jsonl", file_mode)
    err_w  = JsonlWriter(output_dir / "stac_ingest_errors.jsonl", file_mode)

    summary: dict[str, Any] = {
        "providers_attempted": len(rows),
        "providers_crawled":   0,
        "providers_failed":    0,
        "collections_total":   0,
        "items_sampled":       0,
        "rag_docs_written":    0,
        "errors_total":        0,
        "start_ts":            _now_ts(),
        "end_ts":              "",
        "output_dir":          str(output_dir.resolve()),
    }

    # Global endpoints that aggregate all collections from many sub-providers.
    # Crawling them without a cap would return tens of thousands of collections
    # that are already covered by the individual provider-scoped URLs in the CSV.
    _SKIP_ROOTS: set[str] = {
        "https://cmr.earthdata.nasa.gov/stac/ALL",
    }

    # Global dedup: same collection ID from the same provider root
    global_seen: set[str] = set()

    try:
        for idx, row in enumerate(rows):
            provider_url: str = row["url"]
            access: str       = row["access"]

            if provider_url in _SKIP_ROOTS:
                log.info("[%d/%d] SKIP (global aggregator): %s",
                         idx + 1, len(rows), provider_url)
                summary["providers_attempted"] -= 1
                continue

            if _canonical(provider_url) in done_providers:
                log.info("[%d/%d] SKIP (already done): %s",
                         idx + 1, len(rows), provider_url)
                summary["providers_attempted"] -= 1
                continue

            log.info("[%d/%d] %s", idx + 1, len(rows), provider_url)

            with make_client(args.timeout, access) as client:
                provider = crawl_provider(
                    client, provider_url, access, args.retries, err_w,
                )
                if not provider:
                    summary["providers_failed"] += 1
                    continue

                prov_w.write(provider)
                summary["providers_crawled"] += 1
                log.info("  collections_url : %s", provider.collections_url)
                if provider.search_url:
                    log.info("  search          : %s (%s)",
                             provider.search_url, provider.search_method)
                if provider.queryables_url:
                    log.info("  queryables      : %s", provider.queryables_url)

                col_count = 0
                item_count = 0

                for col in crawl_collections(
                    client, provider, args.retries,
                    args.max_collections, err_w,
                ):
                    dedup_key = f"{provider_url}::{col.collection_id}"
                    if dedup_key in global_seen:
                        log.debug("  skip duplicate: %s", col.collection_id)
                        continue
                    global_seen.add(dedup_key)

                    col_w.write(col)
                    rag_w.write(build_rag_document(col))
                    col_count += 1

                    if args.sample_items > 0:
                        sampled = sample_items(
                            client, provider, col.collection_id,
                            args.sample_items, args.retries, err_w,
                        )
                        for item in sampled:
                            item_w.write(item)
                        item_count += len(sampled)

                    if col_count % 25 == 0:
                        log.info("  … %d collections", col_count)

                summary["collections_total"] += col_count
                summary["items_sampled"]     += item_count
                log.info("  Done: %d collections, %d items", col_count, item_count)

    finally:
        prov_w.close()
        col_w.close()
        item_w.close()
        rag_w.close()
        err_w.close()

        summary["rag_docs_written"] = rag_w.count
        summary["errors_total"]     = err_w.count
        summary["end_ts"]           = _now_ts()

        summary_path = output_dir / "stac_ingest_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        log.info("")
        log.info("══════════════════════════════════════════════")
        log.info("  Providers crawled  : %d / %d",
                 summary["providers_crawled"], summary["providers_attempted"])
        log.info("  Collections total  : %d", summary["collections_total"])
        log.info("  Items sampled      : %d", summary["items_sampled"])
        log.info("  RAG docs written   : %d", summary["rag_docs_written"])
        log.info("  Errors logged      : %d", summary["errors_total"])
        log.info("  Output             : %s", summary["output_dir"])
        log.info("══════════════════════════════════════════════")


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(args)


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────
#
# Install dependency:
#   pip install httpx
#
# 1. Validate CSV without any crawling:
#   python stac_ingest.py --csv collections.csv --dry-run
#
# 2. Full crawl, default 3 items sampled per collection:
#   python stac_ingest.py --csv collections.csv
#
# 3. Sample 5 items, cap at 100 collections per provider, verbose:
#   python stac_ingest.py --csv collections.csv \
#       --sample-items 5 --max-collections 100 --verbose
#
# 4. CSV where the URL column is named "api_endpoint":
#   python stac_ingest.py --csv collections.csv --url-column api_endpoint
#
# 5. Provider requires a Bearer token (stored in "token" column):
#   python stac_ingest.py --csv collections.csv --access-column token
#
# 6. Write outputs to a custom directory:
#   python stac_ingest.py --csv collections.csv --output-dir /data/stac
#
# 7. Slower networks — longer timeout, more retries:
#   python stac_ingest.py --csv collections.csv --timeout 120 --retries 5
#
# 8. Metadata-only crawl (no item sampling):
#   python stac_ingest.py --csv collections.csv --sample-items 0
#
# Output files produced in kb/outputs/ (or --output-dir):
#   stac_providers.jsonl      one record per crawled provider root
#   stac_collections.jsonl    one record per unique collection
#   stac_items_sample.jsonl   item id, datetime, bbox, asset keys/hrefs/roles/types
#   stac_rag_documents.jsonl  text + metadata for vector / BM25 indexing
#   stac_ingest_errors.jsonl  per-request errors: stage + url + exception
#   stac_ingest_summary.json  counts and timestamps for the full run
