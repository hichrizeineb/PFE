"""
RAG Pipeline - Version Standalone (sans dépendances externes)
==============================================================
Pipeline complet pour transformer les données Copernicus en documents RAG.

Input:  kb/Search queries.json  (396 queries, 4 satellites)
Output: kb/outputs/documents_embedding.jsonl
        kb/outputs/documents_vectorstore.jsonl
        kb/outputs/documents_full.json
"""

import json
import re
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Tuple
from pathlib import Path


# ============================================================================
# ENUMS
# ============================================================================

class Satellite(str, Enum):
    S1  = "S1"    # Sentinel-1: Radar SAR
    S2  = "S2"    # Sentinel-2: Optique multispectrale
    S3  = "S3"    # Sentinel-3: Océan & Terre
    S5P = "S5P"   # Sentinel-5P: Atmosphère


class GeometryType(str, Enum):
    POINT         = "Point"
    POLYGON       = "Polygon"
    LINE_STRING   = "LineString"
    MULTI_POLYGON = "MultiPolygon"


class ProductType(str, Enum):
    MSI2A    = "MSI2A"
    OL_1_EFR = "OL_1_EFR___"
    L2_NO2   = "L2__NO2___"


class Dataset(str, Enum):
    SENTINEL_1_MOSAICS = "sentinel-1-global-mosaics"
    SENTINEL_2_L2A     = "sentinel-2-l2a"
    SENTINEL_3_OLCI    = "sentinel-3-olci-1-efr-ntc"
    SENTINEL_5P_NO2    = "sentinel-5p-l2-no2-rpro"


class Continent(str, Enum):
    EUROPE        = "europe"
    ASIA          = "asia"
    AFRICA        = "africa"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA       = "oceania"
    ARCTIC        = "arctic"


class Biome(str, Enum):
    FOREST       = "forest"
    DESERT       = "desert"
    OCEAN        = "ocean"
    COASTAL      = "coastal"
    MOUNTAIN     = "mountain"
    AGRICULTURAL = "agricultural"
    URBAN        = "urban"
    ICE          = "ice"
    STEPPE       = "steppe"
    RAINFOREST   = "rainforest"
    FRESHWATER   = "freshwater"
    RIVER        = "river"


class MissionType(str, Enum):
    RADAR_IMAGING    = "radar_imaging"
    OPTICAL_IMAGING  = "optical_imaging"
    OCEAN_MONITORING = "ocean_monitoring"
    ATMOSPHERIC      = "atmospheric"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_lon + self.max_lon) / 2,
            (self.min_lat + self.max_lat) / 2,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalRange:
    start: datetime
    end:   datetime

    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days

    @property
    def year(self) -> int:
        return self.start.year

    @property
    def season(self) -> str:
        m = self.start.month
        if m in [12, 1, 2]:  return "winter"
        if m in [3, 4, 5]:   return "spring"
        if m in [6, 7, 8]:   return "summer"
        return "autumn"

    def to_dict(self) -> dict:
        return {
            "start":         self.start.isoformat(),
            "end":           self.end.isoformat(),
            "duration_days": self.duration_days,
            "year":          self.year,
            "season":        self.season,
        }


@dataclass
class Geometry:
    type:        GeometryType
    coordinates: list

    def compute_bbox(self) -> BoundingBox:
        all_coords = self._flatten(self.coordinates)
        if not all_coords:
            if self.type == GeometryType.POINT:
                lon, lat = self.coordinates[0], self.coordinates[1]
                return BoundingBox(lon, lat, lon, lat)
            return BoundingBox(0, 0, 0, 0)
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        return BoundingBox(min(lons), min(lats), max(lons), max(lats))

    def _flatten(self, coords: list) -> List[List[float]]:
        if not coords:
            return []
        if isinstance(coords[0], (int, float)):
            return [coords]
        if isinstance(coords[0], list) and isinstance(coords[0][0], (int, float)):
            return coords
        result = []
        for item in coords:
            result.extend(self._flatten(item))
        return result

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.compute_bbox().center

    def to_dict(self) -> dict:
        bbox = self.compute_bbox()
        return {
            "type":        self.type.value,
            "coordinates": self.coordinates,
            "bbox":        bbox.to_dict(),
            "centroid":    list(self.centroid),
        }


@dataclass
class CopernicusDocument:
    """Normalized RAG document for one Copernicus satellite search query."""
    id:             str
    original_name:  str
    satellite:      Satellite
    mission_type:   MissionType
    dataset:        Dataset
    product_type:   Optional[ProductType]
    geometry:       Geometry
    region_name:    str
    continent:      Optional[Continent]
    biome:          Optional[Biome]
    temporal_range: TemporalRange
    api_endpoint:   str
    country:        str = ""
    semantic_text:  str = ""
    tags:           List[str] = field(default_factory=list)
    created_at:     str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.semantic_text:
            self.semantic_text = self._generate_semantic_text()

    # ------------------------------------------------------------------
    # Semantic text — rich enough for natural-language vector search
    # ------------------------------------------------------------------
    def _generate_semantic_text(self) -> str:
        mission_plain = {
            MissionType.RADAR_IMAGING:    "radar SAR synthetic aperture radar imaging",
            MissionType.OPTICAL_IMAGING:  "optical multispectral imaging visible infrared",
            MissionType.OCEAN_MONITORING: "ocean land surface monitoring OLCI colour",
            MissionType.ATMOSPHERIC:      "atmospheric air quality pollution NO2 ozone monitoring",
        }
        biome_synonyms = {
            Biome.FOREST:       "forest woodland trees vegetation forestry arboreal tree cover",
            Biome.DESERT:       "desert arid dry barren xeric sand dunes",
            Biome.OCEAN:        "ocean sea marine water pelagic lake",
            Biome.COASTAL:      "coast coastal shore beach harbour estuary port inlet",
            Biome.MOUNTAIN:     "mountain alpine highland elevation terrain fjord summit valley",
            Biome.AGRICULTURAL: "agriculture agricultural farming crops fields arable land use",
            Biome.URBAN:        "urban city metropolitan built-up industrial area capital",
            Biome.ICE:          "ice glacier frozen arctic snow polar permafrost",
            Biome.STEPPE:       "steppe grassland prairie semi-arid",
            Biome.RAINFOREST:   "rainforest tropical jungle dense canopy",
        }
        continent_hints = {
            Continent.EUROPE:        "Europe European EU continent France Germany Spain Italy UK",
            Continent.ASIA:          "Asia Asian China Japan India Korea",
            Continent.AFRICA:        "Africa African Morocco Egypt Nigeria Kenya",
            Continent.NORTH_AMERICA: "North America USA Canada Mexico",
            Continent.SOUTH_AMERICA: "South America Brazil Argentina Chile",
            Continent.OCEANIA:       "Oceania Australia New Zealand Pacific",
            Continent.ARCTIC:        "Arctic polar Greenland high-latitude",
        }

        country = self.country or self._infer_country()

        parts = [
            f"Satellite {self.satellite.value} "
            f"{mission_plain.get(self.mission_type, '')} search query",
            f"for {self.region_name}",
        ]

        if country:
            parts.append(f"in {country}")

        if self.continent:
            hint = continent_hints.get(self.continent, self.continent.value)
            parts.append(f"continent {hint}")

        parts.append(f"using {self.geometry.type.value} geometry")

        if self.product_type:
            parts.append(f"product type {self.product_type.value}")

        # ISO dates + human-readable (both improve keyword & semantic matching)
        parts.extend([
            f"from {self.temporal_range.start.strftime('%B %Y')} "
            f"({self.temporal_range.start.strftime('%Y-%m-%d')})",
            f"to {self.temporal_range.end.strftime('%B %Y')} "
            f"({self.temporal_range.end.strftime('%Y-%m-%d')})",
            f"({self.temporal_range.duration_days} days)",
            f"in {self.temporal_range.season}",
        ])

        if self.biome:
            synonyms = biome_synonyms.get(self.biome, self.biome.value)
            parts.append(f"environment {synonyms}")

        center = self.geometry.centroid
        parts.append(f"centered at ({center[0]:.2f}, {center[1]:.2f})")

        return ". ".join(parts) + "."

    def _infer_country(self) -> Optional[str]:
        region_lower = self.region_name.lower()
        for keyword, country in REGION_COUNTRY_MAP.items():
            if keyword in region_lower:
                return country
        return None

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------
    def to_embedding_input(self) -> dict:
        """Format for embedding model + ChromaDB.
        No None values — ChromaDB requires strings or numbers only.
        """
        return {
            "id":   self.id,
            "text": self.semantic_text,
            "metadata": {
                "original_name": self.original_name,
                "satellite":     self.satellite.value,
                "mission_type":  self.mission_type.value,
                "geometry_type": self.geometry.type.value,
                "region":        self.region_name,
                "dataset":       self.dataset.value,
                "year":          self.temporal_range.year,
                "season":        self.temporal_range.season,
                # Human-readable date strings (display only)
                "date_start":     self.temporal_range.start.strftime("%Y-%m-%d"),
                "date_end":       self.temporal_range.end.strftime("%Y-%m-%d"),
                "duration_days":  self.temporal_range.duration_days,
                # Integer dates for ChromaDB $gte/$lte filtering (YYYYMMDD)
                "date_start_int": int(self.temporal_range.start.strftime("%Y%m%d")),
                "date_end_int":   int(self.temporal_range.end.strftime("%Y%m%d")),
                # Empty string instead of None (ChromaDB requirement)
                "continent":     self.continent.value if self.continent else "",
                "country":       self.country,
                "biome":         self.biome.value if self.biome else "",
                "center_lon":    self.geometry.centroid[0],
                "center_lat":    self.geometry.centroid[1],
            },
        }

    def to_vector_store_doc(self) -> dict:
        """Format for vector store (ChromaDB, Pinecone, etc.)."""
        return {
            "id":      self.id,
            "content": self.semantic_text,
            "metadata": {
                "original_name": self.original_name,
                "satellite":     self.satellite.value,
                "mission_type":  self.mission_type.value,
                "dataset":       self.dataset.value,
                "product_type":  self.product_type.value if self.product_type else "",
                "geometry_type": self.geometry.type.value,
                "region_name":   self.region_name,
                "continent":     self.continent.value if self.continent else "",
                "biome":         self.biome.value if self.biome else "",
                "year":          self.temporal_range.year,
                "season":        self.temporal_range.season,
                "duration_days": self.temporal_range.duration_days,
                "date_start":    self.temporal_range.start.strftime("%Y-%m-%d"),
                "date_end":      self.temporal_range.end.strftime("%Y-%m-%d"),
                "center_lon":    self.geometry.centroid[0],
                "center_lat":    self.geometry.centroid[1],
                "bbox":          self.geometry.compute_bbox().to_dict(),
                "tags":          self.tags,
                "created_at":    self.created_at,
            },
        }

    def to_full_dict(self) -> dict:
        return {
            "id":             self.id,
            "original_name":  self.original_name,
            "satellite":      self.satellite.value,
            "mission_type":   self.mission_type.value,
            "dataset":        self.dataset.value,
            "product_type":   self.product_type.value if self.product_type else None,
            "geometry":       self.geometry.to_dict(),
            "region_name":    self.region_name,
            "continent":      self.continent.value if self.continent else None,
            "biome":          self.biome.value if self.biome else None,
            "temporal_range": self.temporal_range.to_dict(),
            "api_endpoint":   self.api_endpoint,
            "semantic_text":  self.semantic_text,
            "tags":           self.tags,
            "created_at":     self.created_at,
        }


# ============================================================================
# MAPPINGS
# ============================================================================

SATELLITE_MISSION_MAP = {
    Satellite.S1:  MissionType.RADAR_IMAGING,
    Satellite.S2:  MissionType.OPTICAL_IMAGING,
    Satellite.S3:  MissionType.OCEAN_MONITORING,
    Satellite.S5P: MissionType.ATMOSPHERIC,
}

DATASET_PATTERNS = {
    "sentinel-1-global-mosaics": Dataset.SENTINEL_1_MOSAICS,
    "sentinel-2-l2a":            Dataset.SENTINEL_2_L2A,
    "sentinel-3-olci-1-efr-ntc": Dataset.SENTINEL_3_OLCI,
    "sentinel-5p-l2-no2-rpro":   Dataset.SENTINEL_5P_NO2,
}

# Expanded biome detection — matched against region_name.lower()
BIOME_KEYWORDS: Dict[Biome, List[str]] = {
    Biome.FOREST:       ["forest", "woodland", "taiga", "arboreal"],
    Biome.DESERT:       ["desert", "sahara", "gobi", "arid"],
    Biome.OCEAN:        ["sea", "ocean", "gulf", "bay", "channel"],
    Biome.COASTAL:      ["coast", "coastline", "shore", "beach", "harbour", "port",
                         "sound", "lough", "estuary", "mouth", "archipelago",
                         "kvarner", "oresund", "kattegat"],
    Biome.MOUNTAIN:     ["mountain", "alps", "highland", "andes", "peak", "fjord",
                         "valley", "alpine", "foreland", "upland", "plateau", "hills"],
    Biome.AGRICULTURAL: ["agriculture", "plain", "farmland", "pampas", "polders",
                         "lowland", "moravia", "pannonian", "silesia"],
    Biome.URBAN:        ["industrial", "suburbs", "metropolitan", "capital",
                         "urban", "belt",
                         # River-corridor city patterns (cover ~208 untagged docs)
                         "corridor", "confluence", "basin", "central", "randstad",
                         "mittelland", "lesser", "greater", "ilfov", "banat",
                         "moldavia", "mazovia", "transylvania", "bohemia",
                         "franconia", "dalmatia", "kosovo", "thrace",
                         # River names paired with cities
                         "rhine", "danube", "weser", "elbe", "severn", "clyde",
                         "mersey", "tyne", "avon", "sava", "drava", "vardar",
                         "daugava", "leman", "isar", "neckar", "main", "garonne",
                         "vineyards",
                         "firth", "moray", "flanders", "broadland", "solent",
                         "provence", "brandenburg"],
    Biome.ICE:          ["ice", "glacier", "arctic", "greenland", "antarctica"],
    Biome.STEPPE:       ["steppe", "prairie", "grassland"],
    Biome.RAINFOREST:   ["rainforest", "jungle", "amazonia", "congo", "borneo"],
    Biome.FRESHWATER:   ["lake", "loch", "reservoir", "pond", "wetland", "lagoon",
                         "lough", "volta", "titicaca", "baikal", "tanganyika",
                         "malawi", "victoria", "huron", "superior", "michigan",
                         "ontario", "erie"],
    Biome.RIVER:        ["river", "stream", "creek", "tributary", "delta",
                         "centerline", "nile", "amazon", "mississippi", "yangtze",
                         "mekong", "niger", "zambezi", "congo", "murray", "ob",
                         "yenisei", "lena", "mackenzie", "irrawaddy"],
}

# Region keyword → Continent
REGION_CONTINENT_MAP: Dict[str, Continent] = {
    # ── Rivers / Seas / Geographic features ──────────────────────────
    "iceland": Continent.EUROPE, "reykjavik": Continent.EUROPE,
    "akureyri": Continent.EUROPE,
    "alps": Continent.EUROPE, "mediterranean": Continent.EUROPE,
    "baltic": Continent.EUROPE, "adriatic": Continent.EUROPE,
    "aegean": Continent.EUROPE, "celtic": Continent.EUROPE,
    "north sea": Continent.EUROPE, "marmara": Continent.EUROPE,
    "ionian": Continent.EUROPE, "tyrrhenian": Continent.EUROPE,
    "ligurian": Continent.EUROPE, "black sea": Continent.EUROPE,
    "rhine": Continent.EUROPE, "danube": Continent.EUROPE,
    "garonne": Continent.EUROPE, "rhone": Continent.EUROPE,
    "seine": Continent.EUROPE, "elbe": Continent.EUROPE,
    "weser": Continent.EUROPE, "thames": Continent.EUROPE,
    "severn": Continent.EUROPE, "clyde": Continent.EUROPE,
    "mersey": Continent.EUROPE, "aire": Continent.EUROPE,
    "tyne": Continent.EUROPE, "avon": Continent.EUROPE,
    "daugava": Continent.EUROPE, "drava": Continent.EUROPE,
    "sava": Continent.EUROPE, "vardar": Continent.EUROPE,
    "leman": Continent.EUROPE, "moray": Continent.EUROPE,
    "firth": Continent.EUROPE, "solent": Continent.EUROPE,
    "broadland": Continent.EUROPE, "flanders": Continent.EUROPE,
    "franconia": Continent.EUROPE, "bohemia": Continent.EUROPE,
    "moravia": Continent.EUROPE, "transylvania": Continent.EUROPE,
    "pannonian": Continent.EUROPE, "dalmatia": Continent.EUROPE,
    "silesia": Continent.EUROPE, "mazovia": Continent.EUROPE,
    "lapland": Continent.EUROPE, "riviera": Continent.EUROPE,
    "provence": Continent.EUROPE, "randstad": Continent.EUROPE,
    "ruhr": Continent.EUROPE, "kattegat": Continent.EUROPE,
    "oresund": Continent.EUROPE, "kvarner": Continent.EUROPE,
    "thrace": Continent.EUROPE, "kosovo": Continent.EUROPE,
    "moldova": Continent.EUROPE,
    # ── France ───────────────────────────────────────────────────────
    "france": Continent.EUROPE, "french": Continent.EUROPE,
    "paris": Continent.EUROPE, "lyon": Continent.EUROPE,
    "marseille": Continent.EUROPE, "bordeaux": Continent.EUROPE,
    "toulouse": Continent.EUROPE, "nantes": Continent.EUROPE,
    "strasbourg": Continent.EUROPE, "lille": Continent.EUROPE,
    "le havre": Continent.EUROPE, "nice": Continent.EUROPE,
    # ── United Kingdom ───────────────────────────────────────────────
    "london": Continent.EUROPE, "manchester": Continent.EUROPE,
    "leeds": Continent.EUROPE, "liverpool": Continent.EUROPE,
    "edinburgh": Continent.EUROPE, "glasgow": Continent.EUROPE,
    "cardiff": Continent.EUROPE, "belfast": Continent.EUROPE,
    "bristol": Continent.EUROPE, "newcastle": Continent.EUROPE,
    "southampton": Continent.EUROPE, "norwich": Continent.EUROPE,
    "plymouth": Continent.EUROPE, "aberdeen": Continent.EUROPE,
    "inverness": Continent.EUROPE,
    # ── Ireland ──────────────────────────────────────────────────────
    "dublin": Continent.EUROPE, "cork": Continent.EUROPE,
    "galway": Continent.EUROPE,
    # ── Germany ──────────────────────────────────────────────────────
    "berlin": Continent.EUROPE, "munich": Continent.EUROPE,
    "hamburg": Continent.EUROPE, "frankfurt": Continent.EUROPE,
    "cologne": Continent.EUROPE, "stuttgart": Continent.EUROPE,
    "nuremberg": Continent.EUROPE, "hanover": Continent.EUROPE,
    "leipzig": Continent.EUROPE, "dresden": Continent.EUROPE,
    "bremen": Continent.EUROPE,
    # ── Netherlands ──────────────────────────────────────────────────
    "amsterdam": Continent.EUROPE, "rotterdam": Continent.EUROPE,
    "groningen": Continent.EUROPE,
    # ── Belgium ──────────────────────────────────────────────────────
    "brussels": Continent.EUROPE, "antwerp": Continent.EUROPE,
    # ── Austria ──────────────────────────────────────────────────────
    "vienna": Continent.EUROPE, "salzburg": Continent.EUROPE,
    "linz": Continent.EUROPE, "innsbruck": Continent.EUROPE,
    # ── Switzerland ──────────────────────────────────────────────────
    "zurich": Continent.EUROPE, "geneva": Continent.EUROPE,
    "bern": Continent.EUROPE, "lausanne": Continent.EUROPE,
    "basel": Continent.EUROPE,
    # ── Scandinavia ──────────────────────────────────────────────────
    "oslo": Continent.EUROPE, "bergen": Continent.EUROPE,
    "stockholm": Continent.EUROPE, "goteborg": Continent.EUROPE,
    "malmo": Continent.EUROPE, "helsinki": Continent.EUROPE,
    "tampere": Continent.EUROPE, "turku": Continent.EUROPE,
    # ── Baltic States ────────────────────────────────────────────────
    "tallinn": Continent.EUROPE, "riga": Continent.EUROPE,
    "vilnius": Continent.EUROPE, "kaunas": Continent.EUROPE,
    # ── Poland ───────────────────────────────────────────────────────
    "warsaw": Continent.EUROPE, "krakow": Continent.EUROPE,
    "wroclaw": Continent.EUROPE, "gdansk": Continent.EUROPE,
    "poznan": Continent.EUROPE,
    # ── Czech & Slovakia ─────────────────────────────────────────────
    "prague": Continent.EUROPE, "brno": Continent.EUROPE,
    "bratislava": Continent.EUROPE,
    # ── Hungary & Romania ────────────────────────────────────────────
    "budapest": Continent.EUROPE, "debrecen": Continent.EUROPE,
    "bucharest": Continent.EUROPE, "iasi": Continent.EUROPE,
    "cluj": Continent.EUROPE, "timisoara": Continent.EUROPE,
    # ── Balkans ──────────────────────────────────────────────────────
    "sofia": Continent.EUROPE, "plovdiv": Continent.EUROPE,
    "belgrade": Continent.EUROPE, "novi sad": Continent.EUROPE,
    "zagreb": Continent.EUROPE, "rijeka": Continent.EUROPE,
    "dubrovnik": Continent.EUROPE, "split": Continent.EUROPE,
    "ljubljana": Continent.EUROPE, "maribor": Continent.EUROPE,
    "sarajevo": Continent.EUROPE, "skopje": Continent.EUROPE,
    "tirana": Continent.EUROPE, "podgorica": Continent.EUROPE,
    "chisinau": Continent.EUROPE, "pristina": Continent.EUROPE,
    "luxembourg": Continent.EUROPE,
    # ── North America ────────────────────────────────────────────────
    "alaska": Continent.NORTH_AMERICA, "quebec": Continent.NORTH_AMERICA,
    "mexico": Continent.NORTH_AMERICA, "newfoundland": Continent.NORTH_AMERICA,
    "yukon": Continent.NORTH_AMERICA,
    "gulf of mexico": Continent.NORTH_AMERICA,
    # ── South America ────────────────────────────────────────────────
    "amazonia": Continent.SOUTH_AMERICA, "patagonia": Continent.SOUTH_AMERICA,
    "buenos aires": Continent.SOUTH_AMERICA, "santiago": Continent.SOUTH_AMERICA,
    "lima": Continent.SOUTH_AMERICA, "argentina": Continent.SOUTH_AMERICA,
    "andes": Continent.SOUTH_AMERICA,
    # ── Asia ─────────────────────────────────────────────────────────
    "tokyo": Continent.ASIA, "seoul": Continent.ASIA,
    "mongolia": Continent.ASIA, "siberia": Continent.ASIA,
    "kazakhstan": Continent.ASIA, "caspian": Continent.ASIA,
    "tehran": Continent.ASIA, "bangkok": Continent.ASIA,
    "manila": Continent.ASIA, "karachi": Continent.ASIA,
    "borneo": Continent.ASIA, "kamchatka": Continent.ASIA,
    "oman": Continent.ASIA, "persian gulf": Continent.ASIA,
    "arabian": Continent.ASIA, "bengal": Continent.ASIA,
    # ── Africa ───────────────────────────────────────────────────────
    "morocco": Continent.AFRICA, "namibia": Continent.AFRICA,
    "tunisia": Continent.AFRICA, "johannesburg": Continent.AFRICA,
    "cairo": Continent.AFRICA, "congo": Continent.AFRICA,
    "madagascar": Continent.AFRICA, "ethiopia": Continent.AFRICA,
    "lagos": Continent.AFRICA, "nairobi": Continent.AFRICA,
    "guinea": Continent.AFRICA, "sudan": Continent.AFRICA,
    "red sea": Continent.AFRICA, "mozambique": Continent.AFRICA,
    # ── Oceania ──────────────────────────────────────────────────────
    "new zealand": Continent.OCEANIA, "tasmania": Continent.OCEANIA,
    "melbourne": Continent.OCEANIA, "papua": Continent.OCEANIA,
    "new guinea": Continent.OCEANIA,
    # ── Arctic ───────────────────────────────────────────────────────
    "greenland": Continent.ARCTIC,
}

# Region keyword → country name (used inside semantic text)
REGION_COUNTRY_MAP: Dict[str, str] = {
    # France
    "paris": "France", "lyon": "France", "marseille": "France",
    "bordeaux": "France", "toulouse": "France", "nantes": "France",
    "strasbourg": "France", "lille": "France", "le havre": "France",
    "nice": "France", "garonne": "France", "rhone": "France",
    "seine": "France", "provence": "France", "flanders": "France",
    # UK
    "london": "United Kingdom", "manchester": "United Kingdom",
    "leeds": "United Kingdom", "liverpool": "United Kingdom",
    "edinburgh": "United Kingdom", "glasgow": "United Kingdom",
    "cardiff": "United Kingdom", "belfast": "United Kingdom",
    "bristol": "United Kingdom", "newcastle": "United Kingdom",
    "southampton": "United Kingdom", "norwich": "United Kingdom",
    "plymouth": "United Kingdom", "aberdeen": "United Kingdom",
    "inverness": "United Kingdom", "thames": "United Kingdom",
    "severn": "United Kingdom", "clyde": "United Kingdom",
    "mersey": "United Kingdom", "tyne": "United Kingdom",
    "moray": "United Kingdom", "firth": "United Kingdom",
    "solent": "United Kingdom", "broadland": "United Kingdom",
    # Ireland
    "dublin": "Ireland", "cork": "Ireland", "galway": "Ireland",
    # Germany
    "berlin": "Germany", "munich": "Germany", "hamburg": "Germany",
    "frankfurt": "Germany", "cologne": "Germany", "stuttgart": "Germany",
    "nuremberg": "Germany", "hanover": "Germany", "leipzig": "Germany",
    "dresden": "Germany", "bremen": "Germany", "ruhr": "Germany",
    "elbe": "Germany", "weser": "Germany",
    # Netherlands
    "amsterdam": "Netherlands", "rotterdam": "Netherlands",
    "groningen": "Netherlands",
    # Belgium
    "brussels": "Belgium", "antwerp": "Belgium",
    # Austria
    "vienna": "Austria", "salzburg": "Austria", "linz": "Austria",
    "innsbruck": "Austria",
    # Switzerland
    "zurich": "Switzerland", "geneva": "Switzerland", "bern": "Switzerland",
    "lausanne": "Switzerland", "basel": "Switzerland", "leman": "Switzerland",
    # Norway
    "oslo": "Norway", "bergen": "Norway",
    # Sweden
    "stockholm": "Sweden", "goteborg": "Sweden", "malmo": "Sweden",
    # Finland
    "helsinki": "Finland", "tampere": "Finland", "turku": "Finland",
    # Baltic
    "tallinn": "Estonia", "riga": "Latvia", "vilnius": "Lithuania",
    "kaunas": "Lithuania",
    # Poland
    "warsaw": "Poland", "krakow": "Poland", "wroclaw": "Poland",
    "gdansk": "Poland", "poznan": "Poland",
    # Czech & Slovakia
    "prague": "Czech Republic", "brno": "Czech Republic",
    "bratislava": "Slovakia",
    # Hungary & Romania
    "budapest": "Hungary", "debrecen": "Hungary",
    "bucharest": "Romania", "iasi": "Romania", "cluj": "Romania",
    "timisoara": "Romania",
    # Balkans
    "sofia": "Bulgaria", "plovdiv": "Bulgaria",
    "belgrade": "Serbia", "novi sad": "Serbia",
    "zagreb": "Croatia", "rijeka": "Croatia",
    "dubrovnik": "Croatia", "split": "Croatia",
    "ljubljana": "Slovenia", "maribor": "Slovenia",
    "sarajevo": "Bosnia and Herzegovina",
    "skopje": "North Macedonia",
    "tirana": "Albania", "podgorica": "Montenegro",
    "chisinau": "Moldova", "pristina": "Kosovo",
    "luxembourg": "Luxembourg",
    # Iceland
    "reykjavik": "Iceland", "akureyri": "Iceland",
    # Others
    "alaska": "USA", "mexico": "Mexico",
}


# ============================================================================
# PROCESSOR
# ============================================================================

class CopernicusProcessor:
    """Pipeline: Parse → Validate → Normalize → Enrich → Output"""

    def __init__(self):
        self.stats = {
            "total": 0, "success": 0, "failed": 0,
            "by_satellite": {}, "by_geometry": {},
            "by_continent": {}, "by_biome": {},
        }

    def process_entry(self, raw_entry: dict) -> Optional[CopernicusDocument]:
        try:
            name    = raw_entry.get("name", "")
            request = raw_entry.get("request", {})

            # Satellite
            sat_match = re.search(r"Search (S\d+P?)", name)
            sat_str   = sat_match.group(1) if sat_match else "S1"
            satellite = Satellite(sat_str)

            # Region name — everything after the second " - "
            region_match = re.search(r"- \w+ - (.+)$", name)
            region_name  = region_match.group(1) if region_match else name

            # Parse body
            body_raw = request.get("body", {}).get("raw", "{}")
            body     = json.loads(body_raw)

            # AOI geometry
            aoi           = body.get("aoi", {})
            geometry_type = GeometryType(aoi.get("type", "Point"))
            coordinates   = aoi.get("coordinates", [])
            geometry      = Geometry(type=geometry_type, coordinates=coordinates)

            # Dates
            date_start = datetime.fromisoformat(
                body.get("sensingDateMin", "2024-01-01T00:00:00Z").replace("Z", "+00:00")
            )
            date_end = datetime.fromisoformat(
                body.get("sensingDateMax", "2024-12-31T23:59:59Z").replace("Z", "+00:00")
            )
            temporal_range = TemporalRange(start=date_start, end=date_end)

            # Product type (optional)
            product_type = None
            pt_str = body.get("productType")
            if pt_str:
                try:
                    product_type = ProductType(pt_str)
                except ValueError:
                    pass

            # Dataset from URL
            url = request.get("url", "")
            ds_match  = re.search(r"/datasets/([^/]+)/", url)
            ds_str    = ds_match.group(1) if ds_match else "sentinel-1-global-mosaics"
            dataset   = DATASET_PATTERNS.get(ds_str, Dataset.SENTINEL_1_MOSAICS)

            # Stable ID — full 32-char MD5, no truncation, collision-proof at any scale
            hash_input = f"{name}_{satellite.value}_{date_start.isoformat()}_{date_end.isoformat()}"
            doc_id     = hashlib.md5(hash_input.encode()).hexdigest()

            # Enrichment — use passthrough values from generate_queries.py first,
            # fall back to keyword inference when the field is absent or empty.
            continent = self._resolve_continent(
                raw_entry.get("continent", ""), region_name
            )
            biome = self._resolve_biome(
                raw_entry.get("biome_hint", ""), region_name
            )
            country   = raw_entry.get("country", "")
            tags      = self._generate_tags(
                satellite, geometry_type, temporal_range, continent, biome, product_type
            )

            doc = CopernicusDocument(
                id=doc_id,
                original_name=name,
                satellite=satellite,
                mission_type=SATELLITE_MISSION_MAP[satellite],
                dataset=dataset,
                product_type=product_type,
                geometry=geometry,
                region_name=region_name,
                continent=continent,
                biome=biome,
                temporal_range=temporal_range,
                api_endpoint=url,
                country=country,
                tags=tags,
            )
            self._update_stats(doc)
            return doc

        except Exception as e:
            self.stats["failed"] += 1
            print(f"  ⚠️  Error processing '{raw_entry.get('name', 'unknown')}': {e}")
            return None

    def _infer_continent(self, region_name: str) -> Optional[Continent]:
        region_lower = region_name.lower()
        for keyword, continent in REGION_CONTINENT_MAP.items():
            if keyword in region_lower:
                return continent
        return None

    def _infer_biome(self, region_name: str) -> Optional[Biome]:
        region_lower = region_name.lower()
        for biome, keywords in BIOME_KEYWORDS.items():
            for kw in keywords:
                if kw in region_lower:
                    return biome
        return None

    def _resolve_continent(self, hint: str, region_name: str) -> Optional[Continent]:
        """Use passthrough continent from generate_queries.py; fall back to keyword inference."""
        if hint:
            try:
                return Continent(hint.lower().replace(" ", "_"))
            except ValueError:
                pass
        return self._infer_continent(region_name)

    def _resolve_biome(self, hint: str, region_name: str) -> Optional[Biome]:
        """Use passthrough biome_hint from generate_queries.py; fall back to keyword inference."""
        if hint:
            try:
                return Biome(hint.lower())
            except ValueError:
                pass
        return self._infer_biome(region_name)

    def _generate_tags(self, satellite, geometry_type, temporal_range,
                       continent, biome, product_type) -> List[str]:
        tags = [
            satellite.value,
            geometry_type.value.lower(),
            f"year_{temporal_range.year}",
            temporal_range.season,
        ]
        if product_type:
            tags.append(product_type.value)
        if continent:
            tags.append(continent.value)
        if biome:
            tags.append(biome.value)
        days = temporal_range.duration_days
        if days <= 31:    tags.append("short_term")
        elif days <= 180: tags.append("medium_term")
        else:             tags.append("long_term")
        return tags

    def _update_stats(self, doc: CopernicusDocument):
        self.stats["total"]   += 1
        self.stats["success"] += 1
        sat = doc.satellite.value
        self.stats["by_satellite"][sat] = self.stats["by_satellite"].get(sat, 0) + 1
        geom = doc.geometry.type.value
        self.stats["by_geometry"][geom] = self.stats["by_geometry"].get(geom, 0) + 1
        if doc.continent:
            c = doc.continent.value
            self.stats["by_continent"][c] = self.stats["by_continent"].get(c, 0) + 1
        if doc.biome:
            b = doc.biome.value
            self.stats["by_biome"][b] = self.stats["by_biome"].get(b, 0) + 1

    def process_file(self, file_path: str) -> List[CopernicusDocument]:
        """Process JSON file — handles both list and {satellite: [...]} dict formats."""
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = []
            for v in data.values():
                if isinstance(v, list):
                    entries.extend(v)
                else:
                    entries.append(v)
        else:
            entries = [data]

        return [doc for entry in entries for doc in [self.process_entry(entry)] if doc]

    def export_jsonl(self, documents: List[CopernicusDocument],
                     output_path: str, format_type: str = "vector_store"):
        with open(output_path, "w") as f:
            for doc in documents:
                if format_type == "vector_store":
                    f.write(json.dumps(doc.to_vector_store_doc()) + "\n")
                elif format_type == "embedding":
                    f.write(json.dumps(doc.to_embedding_input()) + "\n")
                else:
                    f.write(json.dumps(doc.to_full_dict()) + "\n")

    def export_json(self, documents: List[CopernicusDocument], output_path: str):
        with open(output_path, "w") as f:
            json.dump([doc.to_full_dict() for doc in documents], f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🛰️  COPERNICUS RAG PIPELINE - Data Processor")
    print("=" * 70)

    processor  = CopernicusProcessor()
    input_file = "/home/zhich/dataionics_rag/kb/Search_queries_world.json"
    print(f"\n📂 Processing: {input_file}")

    t0        = time.time()
    documents = processor.process_file(input_file)
    elapsed   = (time.time() - t0) * 1000

    print(f"\n📊 RÉSULTATS")
    print(f"   ✅ Traités avec succès: {processor.stats['success']}")
    print(f"   ❌ Échecs:              {processor.stats['failed']}")
    print(f"   ⏱️  Temps:              {elapsed:.1f}ms")

    print(f"\n📈 STATISTIQUES PAR CATÉGORIE")
    print(f"   Satellites: {processor.stats['by_satellite']}")
    print(f"   Géométries: {processor.stats['by_geometry']}")
    print(f"   Continents: {processor.stats['by_continent']}")
    print(f"   Biomes:     {processor.stats['by_biome']}")

    output_dir = Path("/home/zhich/dataionics_rag/kb/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    processor.export_jsonl(documents, str(output_dir / "documents_vectorstore.jsonl"), "vector_store")
    processor.export_jsonl(documents, str(output_dir / "documents_embedding.jsonl"),   "embedding")
    processor.export_json (documents, str(output_dir / "documents_full.json"))

    print(f"\n💾 EXPORTS")
    print(f"   → {output_dir}/documents_vectorstore.jsonl")
    print(f"   → {output_dir}/documents_embedding.jsonl")
    print(f"   → {output_dir}/documents_full.json")

    # Sample check
    print(f"\n📄 SAMPLE DOCUMENTS")
    for doc in documents[:2]:
        print(f"\n   ID:        {doc.id}")
        print(f"   Name:      {doc.original_name}")
        print(f"   Satellite: {doc.satellite.value} ({doc.mission_type.value})")
        print(f"   Region:    {doc.region_name} → {doc.continent.value if doc.continent else 'N/A'}")
        print(f"   Biome:     {doc.biome.value if doc.biome else 'N/A'}")
        print(f"   Dates:     {doc.temporal_range.start.strftime('%Y-%m-%d')} → "
              f"{doc.temporal_range.end.strftime('%Y-%m-%d')}")
        print(f"   Semantic:  {doc.semantic_text[:120]}...")
