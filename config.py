# =============================================================================
# config.py — The settings file for the entire project.
# One place to change anything. All other files read from here.
# =============================================================================

from pathlib import Path  # tool to build file paths that work on any OS

# "Where is this file right now?" → that folder becomes our starting point
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Where are the files stored?
# ---------------------------------------------------------------------------

KB_SOURCE        = BASE_DIR / "kb" / "Search queries.json"  # the 90 example queries (our recipe book)
FAISS_INDEX_PATH = BASE_DIR / "kb" / "index.faiss"          # the search index (built once by ingest.py)
RECORDS_PATH     = BASE_DIR / "kb" / "records.json"         # the cleaned version of the 90 queries

# ---------------------------------------------------------------------------
# The AI model that converts text into numbers
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small 90MB model, reads a sentence → outputs 384 numbers
EMBEDDING_DIM   = 384                  # every sentence becomes exactly 384 numbers

# ---------------------------------------------------------------------------
# The local AI chatbot (runs on your machine, no internet needed)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama runs on your own computer at this address
OLLAMA_MODEL    = "mistral:latest"          # the AI model we're using (like a private ChatGPT)

# ---------------------------------------------------------------------------
# The  satellite API address
# ---------------------------------------------------------------------------

# HOST_DATA = "https://your-satellite-api-host.example.com"  # TODO: replace with the real address

# ---------------------------------------------------------------------------
# How many results to fetch and show
# ---------------------------------------------------------------------------

TOP_K_FAISS   = 15    # grab the 15 most similar examples from the knowledge base first
TOP_K_RESULTS = 3     # then keep only the best 3 to show the user
MAX_GEO_KM    = 5000.0  # if a location is more than 5000km away, its geography score = 0

# ---------------------------------------------------------------------------
# How important is each scoring factor? (must add up to 1.0)
# ---------------------------------------------------------------------------

W_SEMANTIC = 0.6  # 60% — does the meaning match? (most important)
W_DATE     = 0.2  # 20% — do the dates overlap?
W_GEO      = 0.2  # 20% — is the location nearby?

# ---------------------------------------------------------------------------
# Translation table: what the user says → what the satellite API needs
# ---------------------------------------------------------------------------
# user types "optical" → API needs ("sentinel-2-l2a", "MSI2A")

MISSION_MAP: dict[str, tuple[str, str]] = {

    # --- Sentinel-2: optical photos of land (vegetation, cities, fields) ---
    "optical":       ("sentinel-2-l2a", "MSI2A"),
    "optique":       ("sentinel-2-l2a", "MSI2A"),   # French
    "infrared":      ("sentinel-2-l2a", "MSI2A"),
    "infrarouge":    ("sentinel-2-l2a", "MSI2A"),   # French
    "ndvi":          ("sentinel-2-l2a", "MSI2A"),
    "moisture":      ("sentinel-2-l2a", "MSI2A"),
    "multispectral": ("sentinel-2-l2a", "MSI2A"),
    "msi2a":         ("sentinel-2-l2a", "MSI2A"),
    "sentinel-2":    ("sentinel-2-l2a", "MSI2A"),
    "s2":            ("sentinel-2-l2a", "MSI2A"),

    # --- Sentinel-1: radar (works at night and through clouds) ---
    "sar":           ("sentinel-1-global-mosaics", "IW"),
    "radar":         ("sentinel-1-global-mosaics", "IW"),
    "sentinel-1":    ("sentinel-1-global-mosaics", "IW"),
    "s1":            ("sentinel-1-global-mosaics", "IW"),

    # --- Sentinel-5P: air quality (NO2, ozone, pollution gases) ---
    "no2":           ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "o3":            ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "so2":           ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "co":            ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "atmospheric":   ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "atmospherique": ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),  # French without accent
    "atmosphérique": ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),  # French with accent
    "pollution":     ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "sentinel-5p":   ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),
    "s5p":           ("sentinel-5p-l2-no2-rpro", "L2__NO2___"),

    # --- Sentinel-3: ocean and coastal water monitoring ---
    "olci":          ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "ocean":         ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "océan":         ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),  # French
    "ocean color":   ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "efr":           ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "sentinel-3":    ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "s3":            ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
    "mer":           ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),  # French for "sea"
    "sea":           ("sentinel-3-olci-1-efr-ntc", "OL_1_EFR___"),
}

# ---------------------------------------------------------------------------
# Reverse lookup: API slug → short mission name
# ---------------------------------------------------------------------------
# "sentinel-2-l2a" → "sentinel-2"

SLUG_TO_MISSION: dict[str, str] = {
    "sentinel-1-global-mosaics":  "sentinel-1",
    "sentinel-2-l2a":             "sentinel-2",
    "sentinel-3-olci-1-efr-ntc":  "sentinel-3",
    "sentinel-5p-l2-no2-rpro":    "sentinel-5p",
}

# ---------------------------------------------------------------------------
# Friendly display names shown to the user
# ---------------------------------------------------------------------------

MISSION_DESCRIPTION: dict[str, str] = {
    "sentinel-1":  "Sentinel-1 SAR Radar",
    "sentinel-2":  "Sentinel-2 Optical/Multispectral",
    "sentinel-3":  "Sentinel-3 Ocean Color (OLCI)",
    "sentinel-5p": "Sentinel-5P Atmospheric (NO₂)",
}

# ---------------------------------------------------------------------------
# Safety lists — only these values are accepted, nothing invented by the AI
# ---------------------------------------------------------------------------

VALID_PRODUCT_TYPES: set[str] = {"MSI2A", "IW", "L2__NO2___", "OL_1_EFR___"}
VALID_DATASET_SLUGS: set[str] = set(SLUG_TO_MISSION.keys())