# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
index_hybrid_expert.py
Builds three indexes from processed legal data:
  1. BM25 (sparse) via rank_bm25
  2. FAISS (dense) via sentence-transformers
  3. Graph (adjacency list) from LegalEdge references
"""
import os
import json
import pickle
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR     = os.environ.get("DATA_DIR", "data_processed")
INDEX_DIR    = os.environ.get("RETRIEVAL_INDEX_DIR", "indexes")
EMBED_MODEL  = "BAAI/bge-m3"
INDEX_PREFIX = "legal"

os.makedirs(INDEX_DIR, exist_ok=True)


COUNTRY_FILTER = os.environ.get("COUNTRY_FILTER", "").lower()
THEME_FILTER   = os.environ.get("THEME_FILTER", "").lower()

def load_all_nodes(data_dir: str) -> List[Dict]:
    """Load every LegalNode from every processed JSON file."""
    nodes = []
    for fname in os.listdir(data_dir):
        if fname.endswith("_processed.json"):
            if COUNTRY_FILTER and COUNTRY_FILTER not in fname.lower():
                continue
            if THEME_FILTER and THEME_FILTER not in fname.lower():
                continue
            with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                node["_source_file"] = fname   # keep provenance
                nodes.append(node)
    return nodes


def load_all_edges(data_dir: str) -> List[Dict]:
    """Load every LegalEdge from every processed JSON file."""
    edges = []
    for fname in os.listdir(data_dir):
        if fname.endswith("_processed.json"):
            if COUNTRY_FILTER and COUNTRY_FILTER not in fname.lower():
                continue
            if THEME_FILTER and THEME_FILTER not in fname.lower():
                continue
            with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            edges.extend(data.get("edges", []))
    return edges


# â”€â”€ BM25 index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_bm25_index(nodes: List[Dict]) -> BM25Okapi:
    print(f"[BM25] Tokenising {len(nodes)} nodes â€¦")
    corpus = []
    for n in nodes:
        text = (n.get("text", "") + " " + n.get("summary", "")).lower()
        tokens = text.split()
        corpus.append(tokens)
    bm25 = BM25Okapi(corpus)
    path = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_bm25.pkl")
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[BM25] Saved -> {path}")
    return bm25


# â”€â”€ Dense (FAISS) index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_dense_index(nodes: List[Dict]) -> faiss.IndexFlatIP:
    print(f"[FAISS] Encoding {len(nodes)} nodes with {EMBED_MODEL} â€¦")
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 1024
    texts = [n.get("text", "") + " " + n.get("summary", "") for n in nodes]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product = cosine on normalised vecs
    index.add(embeddings)
    path = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_faiss.bin")
    faiss.write_index(index, path)
    print(f"[FAISS] Saved {index.ntotal} vectors -> {path}")
    return index


# â”€â”€ Graph (adjacency list) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_graph_index(edges: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Returns an adjacency list keyed by source node_id.
    Each value is a list of {"target": ..., "relation": ..., "weight": ...}
    """
    print(f"[GRAPH] Building adjacency list from {len(edges)} edges â€¦")
    graph: Dict[str, List[Dict]] = {}
    for e in edges:
        src = e.get("source_id", "")
        tgt = e.get("target_id", "")
        rel = e.get("relation_type", "references")
        w   = e.get("weight", 1.0)
        if src:
            graph.setdefault(src, []).append({
                "target": tgt, "relation": rel, "weight": w
            })
    path = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_graph.pkl")
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[GRAPH] Saved {len(graph)} source nodes -> {path}")
    return graph


# â”€â”€ Node lookup helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_node_lookup(nodes: List[Dict]):
    """Save ordered node list and idâ†’index map for fast lookup."""
    id2idx = {n["node_id"]: i for i, n in enumerate(nodes)}
    path_nodes = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_nodes.pkl")
    path_id2idx = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_id2idx.pkl")
    with open(path_nodes, "wb") as f:
        pickle.dump(nodes, f)
    with open(path_id2idx, "wb") as f:
        pickle.dump(id2idx, f)
    print(f"[META] Node lookup saved ({len(nodes)} nodes)")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import sys

if __name__ == "__main__":
    import sys
    # Si un argument est passÃ©, on l'utilise comme DATA_DIR
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    elif os.getenv("DATA_DIR"):
        DATA_DIR = os.getenv("DATA_DIR")
    else:
        DATA_DIR = "data_processed"
        
    print(f"Building indexes from {DATA_DIR}...")
    nodes = load_all_nodes(DATA_DIR)
    edges = load_all_edges(DATA_DIR)
    print(f"\nLoaded {len(nodes)} nodes and {len(edges)} edges.\n")

    save_node_lookup(nodes)
    build_bm25_index(nodes)
    # build_dense_index(nodes)
    build_graph_index(edges)

    print("\n[OK] All indexes built successfully.")
