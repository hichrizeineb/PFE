#!/usr/bin/env python3
"""
embed_stac.py
=============
Reads stac_rag_documents.jsonl produced by stac_ingest.py, generates
sentence embeddings with all-MiniLM-L6-v2, and loads everything into a
new ChromaDB collection called 'stac_collections'.

Keeps the existing 'copernicus_grouped' collection untouched.

Usage:
    python3 embed_stac.py
    python3 embed_stac.py --input kb/outputs/stac_rag_documents.jsonl
    python3 embed_stac.py --batch-size 64 --force

Requirements:
    pip install sentence-transformers chromadb
"""

import argparse
import json
import time
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Missing: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    raise SystemExit("Missing: pip install chromadb")


# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INPUT   = "kb/outputs/stac_rag_documents.jsonl"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "stac_collections"
BATCH_SIZE      = 32


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed stac_rag_documents.jsonl into ChromaDB stac_collections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Path to stac_rag_documents.jsonl")
    p.add_argument("--chroma-dir", default=CHROMA_DIR,
                   help="ChromaDB persistence directory")
    p.add_argument("--collection", default=COLLECTION_NAME,
                   help="ChromaDB collection name")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Embedding batch size")
    p.add_argument("--force", action="store_true",
                   help="Recreate collection if it already exists")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def flatten_metadata(meta: dict) -> dict:
    """
    ChromaDB requires flat values (str, int, float — no None, no lists, no dicts).
    Lists are joined as comma-separated strings.
    """
    flat: dict = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, bool):
            flat[k] = str(v)
        elif isinstance(v, (int, float)):
            flat[k] = v
        elif isinstance(v, list):
            # Nested lists/dicts (e.g. extent_spatial, extent_temporal) must be
            # stored as JSON strings so they stay parseable at query time.
            if any(isinstance(x, (list, dict)) for x in v):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            # Skip nested dicts — not representable as a flat value
            continue
        else:
            flat[k] = str(v)
    return flat


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    print("=" * 60)
    print("embed_stac.py — STAC Collections → ChromaDB")
    print("=" * 60)

    # ── Load documents ─────────────────────────────────────────────────────
    if not input_path.exists():
        raise SystemExit(
            f"Input not found: {input_path}\n"
            "Run stac_ingest.py first."
        )

    documents: list[dict] = []
    skipped = 0
    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping corrupt line {lineno} (truncated by interrupted crawl)")
                skipped += 1

    print(f"\nLoaded {len(documents):,} documents from {input_path}")
    if skipped:
        print(f"  Skipped {skipped} corrupt lines")

    if not documents:
        raise SystemExit("No documents found — nothing to embed.")

    # Deduplicate by doc_id (in case stac_ingest was run multiple times)
    seen: set[str] = set()
    unique_docs: list[dict] = []
    for doc in documents:
        doc_id = doc.get("doc_id") or doc.get("id", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    if len(unique_docs) < len(documents):
        print(f"  Deduplicated: {len(documents):,} → {len(unique_docs):,} unique docs")
    documents = unique_docs

    # ── Load embedding model ───────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("  (First run downloads ~90 MB — subsequent runs are instant)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model ready — output dimension: {dim}")

    # ── Generate embeddings ────────────────────────────────────────────────
    texts = [doc["text"] for doc in documents]
    print(f"\nEmbedding {len(texts):,} documents (batch_size={args.batch_size}) ...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"  Done — {len(embeddings):,} vectors in {elapsed:.1f}s  "
          f"(shape {embeddings.shape})")

    # ── ChromaDB ───────────────────────────────────────────────────────────
    print(f"\nConnecting to ChromaDB at: {args.chroma_dir}")
    client = chromadb.PersistentClient(path=args.chroma_dir)

    existing = [c.name for c in client.list_collections()]
    if args.collection in existing:
        if args.force:
            print(f"  --force: deleting existing '{args.collection}'")
            client.delete_collection(args.collection)
        else:
            existing_col = client.get_collection(args.collection)
            existing_count = existing_col.count()
            print(f"  Collection '{args.collection}' already exists "
                  f"({existing_count:,} docs).")
            print("  Use --force to recreate it, or choose a different --collection name.")
            raise SystemExit("Aborted — collection already exists.")

    collection = client.create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{args.collection}' created (cosine similarity).")

    # ── Insert in batches ──────────────────────────────────────────────────
    print(f"\nInserting {len(documents):,} documents into ChromaDB ...")

    ids       = [doc.get("doc_id") or doc.get("id", str(i))
                 for i, doc in enumerate(documents)]
    metadatas = [flatten_metadata(doc.get("metadata", {})) for doc in documents]
    vectors   = embeddings.tolist()

    for i in range(0, len(documents), args.batch_size):
        batch_end = min(i + args.batch_size, len(documents))
        collection.add(
            ids        = ids[i:batch_end],
            documents  = texts[i:batch_end],
            embeddings = vectors[i:batch_end],
            metadatas  = metadatas[i:batch_end],
        )

    final_count = collection.count()
    print(f"  Inserted {final_count:,} documents.")

    # ── Sanity check ───────────────────────────────────────────────────────
    sanity_queries = [
        "Sentinel-2 optical vegetation monitoring",
        "ocean color chlorophyll sea surface temperature",
        "SAR flood detection radar",
        "atmospheric NO2 air quality",
        "Landsat surface reflectance land cover",
        "digital elevation model terrain",
    ]

    print("\nSanity check — top result per test query:")
    for query in sanity_queries:
        vec = model.encode([query]).tolist()
        results = collection.query(query_embeddings=vec, n_results=1)
        if not results["metadatas"][0]:
            continue
        meta  = results["metadatas"][0][0]
        dist  = results["distances"][0][0]
        score = round(1 - dist, 3)
        cid   = meta.get("collection_id", "?")
        title = meta.get("title", "")[:55]
        prov  = meta.get("provider_root", "").replace("https://", "")[:40]
        print(f"  [{score:.3f}]  {cid[:35]:<35}  {title}  ({prov})")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Collection : {args.collection}")
    print(f"  Documents  : {final_count:,}")
    print(f"  Dimensions : {dim}")
    print(f"  ChromaDB   : {args.chroma_dir}/")
    print(f"{'=' * 60}")
    print("\nNext step: python3 rag_chatbot_v2.py")


if __name__ == "__main__":
    main()
