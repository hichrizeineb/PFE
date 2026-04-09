"""
generate_queries.py
===================
Generates a large dataset of Copernicus satellite search queries
from official geographic databases (GeoNames + Natural Earth).

Zero manual coordinates, names, or biomes — everything is extracted
automatically from source files.

Usage:
    python3 generate_queries.py

Output:
    kb/Search_queries_world.json

Dependencies (beyond standard library):
    pip install requests
"""

import csv
import json
import os
import random
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")


# ============================================================================
# CONFIG
# ============================================================================

GEODATA_DIR = Path("kb/geodata")
OUTPUT_FILE = Path("kb/Search_queries_world.json")

SOURCES = {
    "geonames":  "https://download.geonames.org/export/dump/allCountries.zip",
    "oceans":    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_geography_marine_polys.geojson",
    "rivers":    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_rivers_lake_centerlines.geojson",
    "lakes":     "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_lakes.geojson",
    "countries": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
}

LOCAL_FILES = {
    "geonames":  GEODATA_DIR / "allCountries.txt",
    "oceans":    GEODATA_DIR / "ne_10m_geography_marine_polys.geojson",
    "rivers":    GEODATA_DIR / "ne_10m_rivers_lake_centerlines.geojson",
    "lakes":     GEODATA_DIR / "ne_10m_lakes.geojson",
    "countries": GEODATA_DIR / "ne_110m_admin_0_countries.geojson",
}

# Copernicus satellite config — dataset slug and optional productType
SATELLITES = {
    "S1":  {"dataset": "sentinel-1-global-mosaics",  "product_type": None},
    "S2":  {"dataset": "sentinel-2-l2a",              "product_type": "MSI2A"},
    "S3":  {"dataset": "sentinel-3-olci-1-efr-ntc",  "product_type": "OL_1_EFR___"},
    "S5P": {"dataset": "sentinel-5p-l2-no2-rpro",    "product_type": "L2__NO2___"},
}

URL_TEMPLATE = (
    "{{HOST_DATA}}/api/v1/search/programmes/copernicus"
    "/datasets/{dataset}/products/?page=1&count=100"
)

# GeoNames feature filters (col 6 = feature_class, col 7 = feature_code)
URBAN_MIN_POPULATION = 100_000

MOUNTAIN_CODES = {"MT", "MTS", "PK", "VLC"}                        # peaks & volcanoes only
TERRAIN_CODES  = {"ISL", "ISLS", "PEN", "DSRT"}    # islands, peninsulas, deserts (no plains/plateaux)
FOREST_CODES   = {"FRST", "FRSTS"}
WATER_CODES    = {"SEA", "OCN", "BAY", "GULF", "LK", "LKN", "STM", "STMI"}
PARK_CODES     = {"PRK", "RESV", "RES", "AREA"}

MIN_NAME_LEN  =    4   # skip unrecognizable short names globally
MAX_CITIES    = 5000
MAX_MOUNTAINS = 6000
MAX_TERRAIN   = 3000
MAX_FORESTS   = 1000
MAX_WATERS    = 3000
MAX_PARKS     = 1000
MAX_OCEANS    =  100
MAX_RIVERS    =  100
MAX_LAKES     =  150
MAX_LINESTRING_COORDS = 15     # simplify rivers to this many coordinate pairs

# Date generation — all dates start from 2025-01-01
DATE_BASE = datetime(2025, 1, 1)

# Each variant: max days to offset the start, duration range in days
DATE_VARIANTS = {
    "E": {"label": "weekly",    "start_max_offset": 334, "dur_min": 7,   "dur_max": 29},
    "A": {"label": "short",     "start_max_offset": 273, "dur_min": 30,  "dur_max": 60},
    "B": {"label": "seasonal",  "start_max_offset": 212, "dur_min": 61,  "dur_max": 120},
    "C": {"label": "annual",    "start_max_offset": 151, "dur_min": 121, "dur_max": 365},
    "D": {"label": "multiyear", "start_max_offset": 0,   "dur_min": 366, "dur_max": 730},
}


# ============================================================================
# STEP 1 — Download source files
# ============================================================================

def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download url → dest, printing a progress bar."""
    resp = requests.get(url, stream=True, timeout=180)
    resp.raise_for_status()
    total      = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=512 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                mb  = downloaded / 1_000_000
                print(f"\r    {pct:5.1f}%  ({mb:.1f} MB)", end="", flush=True)
    print()


def download_all() -> None:
    print("\n=== STEP 1 — Downloading source files ===")
    GEODATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, url in SOURCES.items():
        dest = LOCAL_FILES[key]

        if dest.exists():
            print(f"  ✓  Already exists: {dest.name}")
            continue

        print(f"  ↓  {key}: {url}")

        if key == "geonames":
            zip_dest = GEODATA_DIR / "allCountries.zip"
            _download_with_progress(url, zip_dest)
            print(f"     Extracting allCountries.txt …", flush=True)
            with zipfile.ZipFile(zip_dest) as zf:
                zf.extract("allCountries.txt", GEODATA_DIR)
            zip_dest.unlink()
            print(f"     → {dest.name}  ({dest.stat().st_size / 1_000_000:.0f} MB)")
        else:
            try:
                _download_with_progress(url, dest)
                print(f"     → {dest.name}  ({dest.stat().st_size / 1024:.0f} KB)")
            except Exception as exc:
                print(f"  ⚠  Failed to download {key}: {exc}")
                print(f"     URL: {url}")
                print(f"     Skipping — will proceed without this source.")


# ============================================================================
# STEP 4 — Country and continent mapping
# (done before Step 2 because GeoNames extraction needs it)
# ============================================================================

# Natural Earth uses abbreviated names for some countries.
# Normalize them to full names so chatbot filters work correctly.
_NAME_NORMALIZE = {
    "united states of america":          "united states",
    "dem. rep. congo":                   "democratic republic of the congo",
    "central african rep.":              "central african republic",
    "s. sudan":                          "south sudan",
    "solomon is.":                       "solomon islands",
    "bosnia and herz.":                  "bosnia and herzegovina",
    "dominican rep.":                    "dominican republic",
    "eq. guinea":                        "equatorial guinea",
    "w. sahara":                         "western sahara",
    "czechia":                           "czech republic",
    "côte d'ivoire":                     "ivory coast",
    "n. cyprus":                         "northern cyprus",
    "trinidad and tobago":               "trinidad and tobago",
    "papua new guinea":                  "papua new guinea",
}


def build_country_maps() -> tuple:
    """
    Parse Natural Earth countries GeoJSON → two dicts:
      code_to_name      : {"FR": "france", "DE": "germany", ...}
      code_to_continent : {"FR": "europe", "JP": "asia", ...}
    """
    print("\n=== STEP 4 — Building country/continent maps ===")
    with open(LOCAL_FILES["countries"], encoding="utf-8") as f:
        gj = json.load(f)

    code_to_name      = {}
    code_to_continent = {}

    for feat in gj["features"]:
        props     = feat.get("properties") or {}
        iso       = (props.get("ISO_A2") or "").strip().upper()
        # Natural Earth ne_110m has ISO_A2='-99' for France, Norway, Kosovo.
        # Fall back to ISO_A2_EH which is correct for those countries.
        if iso in ("-1", "-99", ""):
            iso = (props.get("ISO_A2_EH") or "").strip().upper()
        name      = (props.get("NAME")   or "").strip().lower()
        name      = _NAME_NORMALIZE.get(name, name)
        continent = (props.get("CONTINENT") or "").strip().lower().replace(" ", "_")
        if iso and iso not in ("-1", "-99", ""):
            code_to_name[iso]      = name
            code_to_continent[iso] = continent

    print(f"  Mapped {len(code_to_name)} country codes")
    return code_to_name, code_to_continent


# ============================================================================
# STEP 2 — Extract locations from GeoNames
# ============================================================================

def _biome_from_feature(feature_class: str, feature_code: str) -> str:
    """
    Map GeoNames feature class + code → biome hint string.
    Empty string means "let processor_standalone.py infer from the name."
    """
    if feature_class == "P":
        return "urban"
    if feature_class == "T":
        if feature_code in {"MT", "MTS", "PK", "VLC"}:
            return "mountain"
        if feature_code in {"ISL", "ISLS", "PEN"}:
            return "coastal"
        return "steppe"          # PLN, PLT, PLAT and other terrain
    if feature_class == "V":
        if feature_code == "FRST":
            return "forest"
        return "steppe"          # GRSLD, MOOR
    if feature_class == "H":
        if feature_code in {"SEA", "OCN", "BAY", "GULF"}:
            return "ocean"
        if feature_code in {"LK", "LKS", "RSV"}:
            return "freshwater"
        if feature_code in {"STM", "STMI"}:
            return "river"
    # L class (parks/reserves) — leave empty, processor infers from name
    return ""


def _cap_shortest(items: dict, n: int) -> list:
    """Return the n entries with shortest names (shortest = most globally recognised)."""
    return sorted(items.values(), key=lambda x: len(x["name"]))[:n]



def extract_geonames(code_to_name: dict, code_to_continent: dict) -> tuple:
    """
    Stream-parse allCountries.txt (can be >300 MB) and bucket locations
    by feature type.  Returns six lists:
        cities, mountains, terrain, forests, waters, parks
    """
    print("\n=== STEP 2 — Extracting locations from GeoNames ===")
    print("  (this may take a few minutes for the full ~11M line file)")

    cities    = {}   # name → dict  (dict used for deduplication by name)
    mountains = {}
    terrain   = {}
    forests   = {}
    waters    = {}
    parks     = {}

    line_count = 0

    with open(LOCAL_FILES["geonames"], encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for cols in reader:
            line_count += 1
            if line_count % 1_000_000 == 0:
                print(f"  … {line_count:,} lines read", flush=True)

            if len(cols) < 17:
                continue

            name   = cols[1].strip()
            lat_s  = cols[4].strip()
            lon_s  = cols[5].strip()
            fc     = cols[6].strip()    # feature class  (P / T / H / V / L)
            fcode  = cols[7].strip()    # feature code   (MT / PK / STM / …)
            cc     = cols[8].strip()    # ISO-2 country code
            pop_s  = cols[14].strip()
            elev_s = cols[15].strip() or cols[16].strip()  # elevation, fallback to DEM

            # Global minimum name length — skip "Y", "Å", "Gy" etc.
            if len(name) < MIN_NAME_LEN:
                continue

            # Skip null island
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
                continue
            if lat == 0.0 and lon == 0.0:
                continue

            country   = code_to_name.get(cc, "")
            continent = code_to_continent.get(cc, "")

            entry = {
                "name":          name,
                "lon":           round(lon, 4),
                "lat":           round(lat, 4),
                "geometry_type": "Point",
                "country":       country,
                "continent":     continent,
                "biome_hint":    _biome_from_feature(fc, fcode),
                "source":        f"geonames_{fc.lower()}",
            }

            # Urban: population cities (pop >= URBAN_MIN_POPULATION)
            # Deduplication key is name+country so "London UK" and "London Canada"
            # are kept separately; the final city_list then picks the largest-pop
            # entry per bare name so well-known capitals win over smaller namesakes.
            if fc == "P":
                try:
                    pop = int(pop_s)
                except ValueError:
                    pop = 0
                if pop >= URBAN_MIN_POPULATION:
                    key = f"{name}||{country}"   # unique per (name, country)
                    if key not in cities or pop > cities[key]["population"]:
                        cities[key] = {**entry, "population": pop}

            # Peaks and volcanoes — elevation >= 500m OR name length >= 6
            elif fc == "T" and fcode in MOUNTAIN_CODES:
                try:
                    elev = int(elev_s)
                except (ValueError, TypeError):
                    elev = 0
                if elev >= 500 or len(name) >= 6:
                    if name not in mountains or elev > mountains[name]["elevation"]:
                        mountains[name] = {**entry, "elevation": elev, "_fcode": fcode}

            # Plains, islands, peninsulas — name length >= 5
            elif fc == "T" and fcode in TERRAIN_CODES:
                if len(name) >= 5 and name not in terrain:
                    terrain[name] = {**entry, "_fcode": fcode}

            # Vegetation / forest zones — name length >= 5
            elif fc == "V" and fcode in FOREST_CODES:
                if len(name) >= 5 and name not in forests:
                    forests[name] = entry

            # Hydrography — skip tiny/unnamed streams (len < 5)
            elif fc == "H" and fcode in WATER_CODES:
                if fcode in {"STM", "STMI"} and len(name) < MIN_NAME_LEN:
                    pass   # skip unnamed/tiny streams
                elif name not in waters:
                    waters[name] = entry

            # Protected areas / parks — name length >= 6
            elif fc == "L" and fcode in PARK_CODES:
                if len(name) >= 6 and name not in parks:
                    parks[name] = entry

    # Per bare name, keep the entry with the highest population
    # (resolves ambiguous names like "London": UK >> Canada).
    best_by_name: dict = {}
    for entry in cities.values():
        bare = entry["name"]
        if bare not in best_by_name or entry["population"] > best_by_name[bare]["population"]:
            best_by_name[bare] = entry
    city_list = sorted(best_by_name.values(), key=lambda x: -x["population"])[:MAX_CITIES]

    # Mountains: all volcanoes ≥ 2500m always included, then fill by elevation
    vlc_high  = [m for m in mountains.values() if m["_fcode"] == "VLC" and m["elevation"] >= 2500]
    non_vlc   = [m for m in mountains.values() if not (m["_fcode"] == "VLC" and m["elevation"] >= 2500)]
    mtn_list  = vlc_high + sorted(non_vlc, key=lambda x: -x["elevation"])[:MAX_MOUNTAINS]

    # Terrain: all deserts always included, then fill by name length
    deserts      = [t for t in terrain.values() if t["_fcode"] == "DSRT"]
    non_desert   = [t for t in terrain.values() if t["_fcode"] != "DSRT"]
    terrain_list = deserts + sorted(non_desert, key=lambda x: len(x["name"]))[:MAX_TERRAIN]
    fst_list     = sorted(forests.values(),   key=lambda x:  len(x["name"]))[:MAX_FORESTS]
    wat_list     = sorted(waters.values(),    key=lambda x:  len(x["name"]))[:MAX_WATERS]
    prk_list     = sorted(parks.values(),     key=lambda x:  len(x["name"]))[:MAX_PARKS]

    print(f"\n  Cities    : {len(city_list):,}  (selected from {len(cities):,} raw)")
    print(f"  Mountains : {len(mtn_list):,}  (selected from {len(mountains):,} raw)")
    print(f"  Terrain   : {len(terrain_list):,}  (selected from {len(terrain):,} raw)")
    print(f"  Forests   : {len(fst_list):,}  (selected from {len(forests):,} raw)")
    print(f"  Waters    : {len(wat_list):,}  (selected from {len(waters):,} raw)")
    print(f"  Parks     : {len(prk_list):,}  (selected from {len(parks):,} raw)")

    return city_list, mtn_list, terrain_list, fst_list, wat_list, prk_list


# ============================================================================
# STEP 3 — Extract from Natural Earth GeoJSON files
# ============================================================================

def _polygon_centroid(coordinates: list) -> tuple:
    """
    Centroid of the exterior ring (index 0) of a GeoJSON Polygon.
    Returns (lon, lat) rounded to 4 decimal places.
    """
    ring = coordinates[0]
    n    = len(ring)
    if n == 0:
        raise ValueError("Empty ring")
    lon = round(sum(c[0] for c in ring) / n, 4)
    lat = round(sum(c[1] for c in ring) / n, 4)
    return lon, lat


def _multipolygon_centroid(coordinates: list) -> tuple:
    """Centroid of the first polygon in a MultiPolygon."""
    return _polygon_centroid(coordinates[0])


def _simplify_linestring(coordinates: list, max_pts: int = MAX_LINESTRING_COORDS) -> list:
    """
    Reduce a coordinate list to at most max_pts points by even sub-sampling.
    Always keeps the first and last point.
    """
    if len(coordinates) <= max_pts:
        return [[round(c[0], 4), round(c[1], 4)] for c in coordinates]

    step    = (len(coordinates) - 1) / (max_pts - 1)
    indices = sorted({round(i * step) for i in range(max_pts)} | {0, len(coordinates) - 1})
    return [[round(coordinates[i][0], 4), round(coordinates[i][1], 4)] for i in indices]


def extract_natural_earth() -> tuple:
    """
    Extract named features from the three Natural Earth GeoJSON files.
    Returns (ocean_locs, river_locs, lake_locs) — each a list of location dicts.
    """
    print("\n=== STEP 3 — Extracting from Natural Earth files ===")

    # Global seen-names set shared across all three files to avoid duplicates
    seen_names = set()

    # ── FILE B: Oceans ───────────────────────────────────────────────────────
    ocean_locs = []
    with open(LOCAL_FILES["oceans"], encoding="utf-8") as f:
        gj = json.load(f)

    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        name  = (props.get("name") or props.get("NAME") or "").strip()
        if not name or name in seen_names:
            continue
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        try:
            if gtype == "Polygon":
                lon, lat = _polygon_centroid(geom["coordinates"])
            elif gtype == "MultiPolygon":
                lon, lat = _multipolygon_centroid(geom["coordinates"])
            else:
                continue
        except (IndexError, ZeroDivisionError, ValueError):
            continue
        if lon == 0.0 and lat == 0.0:
            continue
        seen_names.add(name)
        ocean_locs.append({
            "name":          name,
            "lon":           lon,
            "lat":           lat,
            "geometry_type": "Point",
            "country":       "",
            "continent":     "",
            "biome_hint":    "ocean",
            "source":        "ne_ocean",
        })

    ocean_locs = sorted(ocean_locs, key=lambda x: len(x["name"]))   # keep all named NE marine features
    print(f"  Oceans : {len(ocean_locs)}")

    # ── FILE C: Rivers (LineString geometry preserved) ───────────────────────
    river_locs = []
    with open(LOCAL_FILES["rivers"], encoding="utf-8") as f:
        gj = json.load(f)

    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        name  = (props.get("name") or props.get("NAME") or "").strip()
        if not name or name in seen_names:
            continue
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        if gtype == "LineString":
            raw_coords = geom.get("coordinates", [])
        elif gtype == "MultiLineString":
            # Flatten all sub-linestrings into one continuous coordinate list
            raw_coords = [c for sub in geom.get("coordinates", []) for c in sub]
        else:
            continue
        if len(raw_coords) < 2:
            continue
        coords = _simplify_linestring(raw_coords)
        # Use midpoint coordinate for location reference (naming only, not used in entry)
        mid    = coords[len(coords) // 2]
        seen_names.add(name)
        river_locs.append({
            "name":               name,
            "lon":                mid[0],
            "lat":                mid[1],
            "geometry_type":      "LineString",
            "linestring_coords":  coords,
            "country":            "",
            "continent":          "",
            "biome_hint":         "river",
            "source":             "ne_river",
        })

    river_locs = sorted(river_locs, key=lambda x: len(x["name"]))   # keep all named NE rivers
    print(f"  Rivers : {len(river_locs)}")

    # ── FILE D: Lakes ────────────────────────────────────────────────────────
    lake_locs = []
    with open(LOCAL_FILES["lakes"], encoding="utf-8") as f:
        gj = json.load(f)

    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        name  = (props.get("name") or props.get("NAME") or "").strip()
        if not name or name in seen_names:
            continue
        geom  = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        try:
            if gtype == "Polygon":
                lon, lat = _polygon_centroid(geom["coordinates"])
            elif gtype == "MultiPolygon":
                lon, lat = _multipolygon_centroid(geom["coordinates"])
            else:
                continue
        except (IndexError, ZeroDivisionError, ValueError):
            continue
        if lon == 0.0 and lat == 0.0:
            continue
        seen_names.add(name)
        lake_locs.append({
            "name":          name,
            "lon":           lon,
            "lat":           lat,
            "geometry_type": "Point",
            "country":       "",
            "continent":     "",
            "biome_hint":    "freshwater",
            "source":        "ne_lake",
        })

    lake_locs = sorted(lake_locs, key=lambda x: len(x["name"]))[:MAX_LAKES]
    print(f"  Lakes  : {len(lake_locs)}")

    return ocean_locs, river_locs, lake_locs


# ============================================================================
# STEP 5 — Date generation
# ============================================================================

def _generate_date_pair(variant: str) -> tuple:
    """
    Return (sensingDateMin, sensingDateMax) ISO strings for the given variant.
    Variant D always starts at DATE_BASE (2025-01-01).
    """
    cfg   = DATE_VARIANTS[variant]
    offset = random.randint(0, cfg["start_max_offset"])
    start  = DATE_BASE + timedelta(days=offset)
    dur    = random.randint(cfg["dur_min"], cfg["dur_max"])
    end    = start + timedelta(days=dur)
    return (
        start.strftime("%Y-%m-%dT00:00:00.000Z"),
        end.strftime("%Y-%m-%dT00:00:00.000Z"),
    )


# ============================================================================
# STEP 6 — Build Copernicus entries in exact format
# ============================================================================

def _build_entry(loc: dict, sat: str, date_variant: str) -> dict:
    """
    Build one Copernicus search query entry matching the format
    in kb/Search queries.json exactly.
    """
    cfg     = SATELLITES[sat]
    dataset = cfg["dataset"]
    pt      = cfg["product_type"]

    date_start, date_end = _generate_date_pair(date_variant)

    gtype = loc["geometry_type"]
    if gtype == "LineString":
        aoi = {
            "type":        "LineString",
            "coordinates": loc["linestring_coords"],
        }
    else:
        aoi = {
            "type":        "Point",
            "coordinates": [loc["lon"], loc["lat"]],
        }

    body = {
        "aoi":            aoi,
        "sensingDateMin": date_start,
        "sensingDateMax": date_end,
    }
    if pt:
        body["productType"] = pt

    # json.dumps default separators (', ' and ': ') match the existing file format
    return {
        "name":       f"Search {sat} - {gtype} - {loc['name']}",
        "biome_hint": loc.get("biome_hint", ""),
        "continent":  loc.get("continent", ""),
        "country":    loc.get("country", ""),
        "request": {
            "method": "POST",
            "body": {
                "mode": "raw",
                "raw":  json.dumps(body),
            },
            "url": URL_TEMPLATE.format(dataset=dataset),
        },
    }


def generate_entries(all_locations: list) -> list:
    print("\n=== STEP 6 — Generating Copernicus entries ===")

    entries       = []
    total_expected = len(all_locations) * len(SATELLITES) * len(DATE_VARIANTS)
    progress      = 0

    for loc in all_locations:
        for sat in SATELLITES:
            for variant in DATE_VARIANTS:
                entries.append(_build_entry(loc, sat, variant))
                progress += 1
                if progress % 10_000 == 0:
                    print(f"  … {progress:,} / {total_expected:,} entries", flush=True)

    return entries


# ============================================================================
# STEP 7 — Write output and print stats
# ============================================================================

def write_output(entries: list) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== STEP 7 — Writing output ===")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    size_mb = OUTPUT_FILE.stat().st_size / 1_000_000
    print(f"  ✅ {OUTPUT_FILE}  —  {len(entries):,} entries  ({size_mb:.1f} MB)")


def print_stats(
    cities:    list,
    mountains: list,
    terrain:   list,
    forests:   list,
    waters:    list,
    parks:     list,
    oceans:    list,
    rivers:    list,
    lakes:     list,
    entries:   list,
) -> None:
    sources = {
        "GeoNames cities":       len(cities),
        "GeoNames mountains":    len(mountains),
        "GeoNames terrain":      len(terrain),
        "GeoNames forests":      len(forests),
        "GeoNames water":        len(waters),
        "GeoNames parks":        len(parks),
        "Natural Earth oceans":  len(oceans),
        "Natural Earth rivers":  len(rivers),
        "Natural Earth lakes":   len(lakes),
    }
    total_locs = sum(sources.values())

    sat_counts  = {s: 0 for s in SATELLITES}
    geom_counts = {}

    for e in entries:
        # Parse satellite from "Search S1 - Point - Name"
        parts = e["name"].split(" - ", 2)
        sat   = parts[0].replace("Search ", "").strip()
        gtype = parts[1].strip() if len(parts) > 1 else "?"
        sat_counts[sat]               = sat_counts.get(sat, 0) + 1
        geom_counts[gtype]            = geom_counts.get(gtype, 0) + 1

    per_variant = len(entries) // max(len(DATE_VARIANTS), 1)

    W = 28   # column width for alignment
    print("\n" + "=" * 58)
    print("  GENERATION COMPLETE")
    print("=" * 58)

    print("\n  Locations by source:")
    for src, n in sources.items():
        print(f"    {src:<{W}}: {n:>7,}")
    print(f"    {'─' * (W + 10)}")
    print(f"    {'Total locations':<{W}}: {total_locs:>7,}")

    print("\n  Entries per satellite:")
    for sat, n in sat_counts.items():
        print(f"    {sat:<{W}}: {n:>7,}")

    print("\n  Entries per geometry type:")
    for gtype, n in sorted(geom_counts.items()):
        print(f"    {gtype:<{W}}: {n:>7,}")

    print("\n  Entries per date variant:")
    for v, cfg in DATE_VARIANTS.items():
        label = f"{v} — {cfg['label']} ({cfg['dur_min']}–{cfg['dur_max']} days)"
        print(f"    {label:<{W+4}}: {per_variant:>7,}")

    print(f"\n    {'─' * (W + 10)}")
    print(f"    {'Total entries generated':<{W}}: {len(entries):>7,}")
    print(f"    {'Existing entries (396)':<{W}}: {'396':>7}")
    print(f"    {'Expected ChromaDB total':<{W}}: {len(entries) + 396:>7,}")
    print("=" * 58)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    random.seed(42)    # reproducible date generation

    download_all()

    # Build country/continent maps before GeoNames parsing (needed during parse)
    code_to_name, code_to_continent = build_country_maps()

    # Extract GeoNames locations (streams the large file line by line)
    cities, mountains, terrain, forests, waters, parks = extract_geonames(
        code_to_name, code_to_continent
    )

    # Extract Natural Earth geometry features
    oceans, rivers, lakes = extract_natural_earth()

    # Combine all location sources
    all_locations = (
        cities + mountains + terrain + forests + waters + parks
        + oceans + rivers + lakes
    )
    print(f"\n  Total unique locations across all sources: {len(all_locations):,}")

    # Generate 4 satellites × 4 date variants per location
    entries = generate_entries(all_locations)

    write_output(entries)

    print_stats(
        cities, mountains, terrain, forests, waters, parks,
        oceans, rivers, lakes,
        entries,
    )


if __name__ == "__main__":
    main()
