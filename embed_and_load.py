"""
Step 2 - Embed & Load into ChromaDB
=====================================
Reads documents_embedding.jsonl, generates HuggingFace vectors,
and stores everything into a local ChromaDB collection.

Usage:
    python3 embed_and_load.py

Requirements:
    pip install sentence-transformers chromadb
"""

import json
import time
from pathlib import Path

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("❌ Missing: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    raise SystemExit("❌ Missing: pip install chromadb")


# ── CONFIG ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"          # fast, free, 384-dim
INPUT_JSONL       = "kb/outputs/documents_embedding.jsonl"  # output of processor_standalone.py
CHROMA_DIR        = "./chroma_db"                # local folder for ChromaDB
COLLECTION_NAME   = "copernicus_rag"
BATCH_SIZE        = 32                           # how many docs to embed at once


# ── Load documents ────────────────────────────────────────────────────────────
print("=" * 60)
print("🛰️  STEP 2 — Embed & Load into ChromaDB")
print("=" * 60)

input_path = Path(INPUT_JSONL)
if not input_path.exists():
    raise SystemExit(f"❌ File not found: {INPUT_JSONL}\n   Run processor_standalone.py first.")

documents = []
with open(input_path) as f:
    for line in f:
        line = line.strip()
        if line:
            documents.append(json.loads(line))

print(f"\n📂 Loaded {len(documents)} documents from {INPUT_JSONL}")


# ── Load embedding model ──────────────────────────────────────────────────────
print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
print("   (First run downloads ~90MB — subsequent runs are instant)")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"   ✅ Model loaded — embedding dimension: {model.get_sentence_embedding_dimension()}")


# ── Generate embeddings ───────────────────────────────────────────────────────
print(f"\n⚙️  Generating embeddings (batch size={BATCH_SIZE})...")
texts = [doc["text"] for doc in documents]

start = time.time()
embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
elapsed = time.time() - start

print(f"   ✅ Done — {len(embeddings)} vectors in {elapsed:.1f}s")
print(f"   Vector shape: {embeddings.shape}")


# ── Init ChromaDB ─────────────────────────────────────────────────────────────
print(f"\n💾 Connecting to ChromaDB at: {CHROMA_DIR}")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete collection if it already exists (clean reload)
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    print(f"   ⚠️  Collection '{COLLECTION_NAME}' exists — recreating it.")
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}   # cosine similarity
)
print(f"   ✅ Collection '{COLLECTION_NAME}' created.")


# ── Insert into ChromaDB ──────────────────────────────────────────────────────
print(f"\n📥 Inserting {len(documents)} documents into ChromaDB...")

# ChromaDB expects flat string metadata values
def flatten_metadata(meta: dict) -> dict:
    """Convert None and non-string values to strings for ChromaDB compatibility."""
    flat = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, (int, float)):
            flat[k] = v          # ChromaDB accepts numbers
        else:
            flat[k] = str(v)
    return flat

ids        = [doc["id"] for doc in documents]
texts_list = [doc["text"] for doc in documents]
metadatas  = [flatten_metadata(doc["metadata"]) for doc in documents]
vectors    = embeddings.tolist()

# Insert in batches
for i in range(0, len(documents), BATCH_SIZE):
    collection.add(
        ids        = ids[i:i+BATCH_SIZE],
        documents  = texts_list[i:i+BATCH_SIZE],
        embeddings = vectors[i:i+BATCH_SIZE],
        metadatas  = metadatas[i:i+BATCH_SIZE],
    )

print(f"   ✅ Inserted {collection.count()} documents.")


# ── Quick sanity check query ──────────────────────────────────────────────────
print(f"\n🔍 Sanity check — querying: 'S2 optical data over European forests'")
test_vec = model.encode(["S2 optical data over European forests"]).tolist()
results = collection.query(query_embeddings=test_vec, n_results=3)

for i, (doc_text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n   Result {i+1}:")
    print(f"   Text : {doc_text[:90]}...")
    print(f"   Sat  : {meta.get('satellite')} | Region: {meta.get('region')} | Biome: {meta.get('biome')}")

print(f"\n✅ ChromaDB ready at: {CHROMA_DIR}/")
print(f"   Collection : {COLLECTION_NAME}")
print(f"   Documents  : {collection.count()}")
print(f"\n▶️  Next step: python3 rag_chatbot.py")