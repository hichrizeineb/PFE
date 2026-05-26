"""
build_geo_index.py
==================
Builds a local geo resolver index from query_lookup.jsonl.
Output: kb/outputs/geo_index.json

No hardcoded country→continent mapping. All geographic relationships
are derived from the data itself (which came from GeoNames).

The index enables rag_chatbot_v2.py to:
  - resolve location → country, continent, dominant context
  - resolve country → continent, coastal/inland character
  - check if a location is inland, coastal, or ocean-based

Usage:
    python3 build_geo_index.py
    python3 build_geo_index.py --input kb/outputs/query_lookup.jsonl
    python3 build_geo_index.py --output kb/outputs/geo_index.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

DEFAULT_INPUT  = Path("kb/outputs/query_lookup.jsonl")
DEFAULT_OUTPUT = Path("kb/outputs/geo_index.json")

# Biome → dominant context mapping (data-driven, not country-specific)
OCEAN_BIOMES      = {"ocean"}
COASTAL_BIOMES    = {"coastal"}
FRESHWATER_BIOMES = {"river", "freshwater", "wetland", "lake"}
INLAND_BIOMES     = {"urban", "forest", "desert", "steppe", "mountain",
                     "agriculture", "tundra", "savanna"}


def infer_dominant_context(biome_counts: dict[str, int]) -> str:
    """Infer geographic context from aggregated biome counts."""
    if not biome_counts:
        return "unknown"

    # Remove empty biome key
    valid = {k: v for k, v in biome_counts.items() if k}
    if not valid:
        return "unknown"

    dominant = max(valid, key=valid.get)

    if dominant in OCEAN_BIOMES:
        return "ocean"
    if dominant in COASTAL_BIOMES:
        return "coastal"
    if dominant in FRESHWATER_BIOMES:
        return "freshwater"
    if dominant in INLAND_BIOMES:
        return "inland"
    return "unknown"


def build_geo_index(input_file: Path, output_file: Path) -> None:
    print(f"Reading {input_file} …")
    if not input_file.exists():
        sys.exit(f"Input not found: {input_file}\nRun build_query_lookup.py first.")

    # ── Accumulate data ───────────────────────────────────────────────────────
    # loc_norm → {(co_norm, cont_norm): count, biome_counts, lat_sum, lon_sum, latlon_count}
    loc_acc: dict[str, dict] = {}
    # co_norm → {cont_norm: most_common, biome_counts, location_set}
    co_acc:  dict[str, dict] = {}
    # cont_norm → {co_set, loc_set, count}
    cont_acc: dict[str, dict] = {}

    total_lines = 0

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_lines += 1
            loc_norm  = q.get("loc_norm", "")
            co_norm   = q.get("co_norm",  "")
            cont_norm = q.get("cont_norm", "")
            bio_norm  = q.get("bio_norm", "")
            loc_raw   = q.get("loc", "")
            lat       = q.get("lat")
            lon       = q.get("lon")

            # ── Locations ────────────────────────────────────────────────────
            if loc_norm:
                if loc_norm not in loc_acc:
                    loc_acc[loc_norm] = {
                        "name_counter":    Counter(),
                        "country_counter": Counter(),   # (co_norm, cont_norm) → count
                        "biome_counts":    Counter(),
                        "lat_sum":  0.0,
                        "lon_sum":  0.0,
                        "latlon_n": 0,
                    }
                a = loc_acc[loc_norm]
                a["name_counter"][loc_raw] += 1
                a["country_counter"][(co_norm, cont_norm)] += 1
                a["biome_counts"][bio_norm] += 1
                if lat is not None and lon is not None:
                    a["lat_sum"]  += lat
                    a["lon_sum"]  += lon
                    a["latlon_n"] += 1

            # ── Countries ────────────────────────────────────────────────────
            if co_norm:
                if co_norm not in co_acc:
                    co_acc[co_norm] = {
                        "cont_counter": Counter(),
                        "biome_counts": Counter(),
                        "location_set": set(),
                    }
                b = co_acc[co_norm]
                b["cont_counter"][cont_norm] += 1
                b["biome_counts"][bio_norm]  += 1
                if loc_norm:
                    b["location_set"].add(loc_norm)

            # ── Continents ───────────────────────────────────────────────────
            if cont_norm:
                if cont_norm not in cont_acc:
                    cont_acc[cont_norm] = {
                        "count":    0,
                        "co_set":   set(),
                        "loc_set":  set(),
                    }
                c = cont_acc[cont_norm]
                c["count"]  += 1
                if co_norm:
                    c["co_set"].add(co_norm)
                if loc_norm:
                    c["loc_set"].add(loc_norm)

            if total_lines % 100_000 == 0:
                print(f"  {total_lines:,} lines processed …", end="\r", flush=True)

    print(f"\n  {total_lines:,} total records")

    # ── Build locations dict ──────────────────────────────────────────────────
    print("Building locations index …")
    locations: dict[str, dict] = {}

    for loc_norm, a in loc_acc.items():
        name  = a["name_counter"].most_common(1)[0][0]
        total = sum(a["country_counter"].values())

        # Primary country (most frequent)
        (primary_co, primary_cont), primary_count = a["country_counter"].most_common(1)[0]

        # Alternatives (other countries for same location name)
        alternatives = [
            {"country": co, "continent": ct, "count": cnt}
            for (co, ct), cnt in a["country_counter"].most_common()
            if co != primary_co
        ]

        biome_counts = dict(a["biome_counts"])

        lat_avg = (a["lat_sum"] / a["latlon_n"]) if a["latlon_n"] > 0 else None
        lon_avg = (a["lon_sum"] / a["latlon_n"]) if a["latlon_n"] > 0 else None
        if lat_avg is not None:
            lat_avg = round(lat_avg, 4)
            lon_avg = round(lon_avg, 4)

        has_ocean   = int(biome_counts.get("ocean",   0)) > 0
        has_coastal = int(biome_counts.get("coastal", 0)) > 0
        dominant_context = infer_dominant_context(biome_counts)
        dominant_biome   = max(
            (k for k in biome_counts if k), key=lambda k: biome_counts[k], default=""
        )

        locations[loc_norm] = {
            "name":             name,
            "loc_norm":         loc_norm,
            "country":          primary_co,
            "continent":        primary_cont,
            "lat":              lat_avg,
            "lon":              lon_avg,
            "count":            total,
            "biome_counts":     biome_counts,
            "dominant_biome":   dominant_biome,
            "dominant_context": dominant_context,
            "has_ocean_records":   has_ocean,
            "has_coastal_records": has_coastal,
            "alternatives":     alternatives,
        }

    # ── Build countries dict ──────────────────────────────────────────────────
    print("Building countries index …")
    countries: dict[str, dict] = {}

    for co_norm, b in co_acc.items():
        total = sum(b["cont_counter"].values())
        # Most frequent continent for this country
        primary_cont = b["cont_counter"].most_common(1)[0][0] if b["cont_counter"] else ""
        biome_counts = dict(b["biome_counts"])

        countries[co_norm] = {
            "country":         co_norm,
            "continent":       primary_cont,
            "count":           total,
            "location_count":  len(b["location_set"]),
            "has_ocean_records":   int(biome_counts.get("ocean",   0)) > 0,
            "has_coastal_records": int(biome_counts.get("coastal", 0)) > 0,
            "biome_counts":    biome_counts,
        }

    # ── Build continents dict ─────────────────────────────────────────────────
    print("Building continents index …")
    continents: dict[str, dict] = {}

    for cont_norm, c in cont_acc.items():
        continents[cont_norm] = {
            "continent":      cont_norm,
            "count":          c["count"],
            "country_count":  len(c["co_set"]),
            "location_count": len(c["loc_set"]),
        }

    # ── Write output ──────────────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "locations":  locations,
        "countries":  countries,
        "continents": continents,
    }
    output_file.write_text(json.dumps(index, ensure_ascii=False, indent=2))

    size_mb = output_file.stat().st_size / 1_048_576
    print(f"\nGeo index written: {output_file}  ({size_mb:.1f} MB)")
    print(f"  Locations : {len(locations):,}")
    print(f"  Countries : {len(countries):,}")
    print(f"  Continents: {len(continents):,}")

    # ── Quick sanity checks ───────────────────────────────────────────────────
    checks = ["france", "toulouse", "nice", "bangladesh", "morocco"]
    print("\nSanity checks:")
    for name in checks:
        if name in locations:
            loc = locations[name]
            print(f"  loc[{name}]: country={loc['country']}, "
                  f"continent={loc['continent']}, "
                  f"context={loc['dominant_context']}")
        elif name in countries:
            co = countries[name]
            print(f"  co[{name}]: continent={co['continent']}, "
                  f"coastal={co['has_coastal_records']}")
        else:
            print(f"  [{name}]: not found")

    print(f"\nNext step: python3 rag_chatbot_v2.py --verbose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build geo_index.json from query_lookup.jsonl"
    )
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_geo_index(Path(args.input), Path(args.output))
