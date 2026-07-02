# -*- coding: utf-8 -*-
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import pickle
import faiss
import chromadb
from typing import List, Dict

# Configuration
INDEX_DIR = "indexes_global"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "legal_rag"

def main():
    print("=== HIGH-SPEED FAISS-TO-CHROMADB MIGRATION ===")
    
    nodes_path = os.path.join(INDEX_DIR, "legal_nodes.pkl")
    faiss_path = os.path.join(INDEX_DIR, "legal_faiss.bin")
    
    if not os.path.exists(nodes_path) or not os.path.exists(faiss_path):
        print(f"[ERROR] Required index files not found in '{INDEX_DIR}'!")
        return

    # 1. Load pre-computed nodes
    print(f"[*] Loading pre-computed nodes from '{nodes_path}'...")
    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"[+] Loaded {len(nodes)} legal nodes.")

    # 2. Load pre-computed FAISS embeddings
    print(f"[*] Loading pre-computed FAISS index from '{faiss_path}'...")
    faiss_index = faiss.read_index(faiss_path)
    num_vectors = faiss_index.ntotal
    print(f"[+] FAISS index loaded with {num_vectors} vectors.")

    if len(nodes) != num_vectors:
        print("[ERROR] Mismatch between nodes count and FAISS vectors count!")
        return

    # 3. Reconstruct embeddings from FAISS index directly (Instantaneous!)
    print("[*] Reconstructing pre-computed embeddings matrix from FAISS (Instant!)...")
    embeddings_matrix = faiss_index.reconstruct_n(0, num_vectors)
    print(f"[+] Reconstructed {embeddings_matrix.shape[0]} vectors of dimension {embeddings_matrix.shape[1]}.")

    # 4. Initialize ChromaDB persistent client
    print(f"[*] Connecting to Persistent ChromaDB at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Reset/Create collection
    print(f"[*] Re-creating collection '{COLLECTION_NAME}'...")
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 5. Prepare documents, metadata, and unique IDs
    print("[*] Preparing documents, cleaned metadata, and ensuring unique IDs...")
    documents = []
    metadatas = []
    ids = []
    seen_ids = set()
    
    for idx, node in enumerate(nodes):
        node_id = node.get("node_id", f"node_{idx}")
        # Resolve duplicate IDs
        base_id = node_id
        counter = 1
        while node_id in seen_ids:
            node_id = f"{base_id}_dup{counter}"
            counter += 1
        seen_ids.add(node_id)
        
        text = node.get("text", "").strip()
        summary = node.get("summary", "").strip()
        
        # Combine text and summary
        combined_text = f"{text}\n\n[Summary] {summary}" if summary else text
        documents.append(combined_text)
        
        # Clean metadata to contain only primitives (str, int, float, bool)
        metadata = {
            "node_id": node_id,
            "country": str(node.get("country", "Unknown")),
            "article": str(node.get("article", "")),
            "theme": str(node.get("theme", "")),
            "_source_file": str(node.get("_source_file", ""))
        }
        metadatas.append(metadata)
        ids.append(node_id)

    # Convert embeddings to lists
    print("[*] Converting embeddings matrix to standard lists...")
    embeddings_list = [emb.tolist() for emb in embeddings_matrix]

    # 6. Upload to ChromaDB in batches (using upsert for robustness)
    batch_size = 500
    print(f"[*] Upserting into ChromaDB in batches of {batch_size}...")
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end_idx],
            embeddings=embeddings_list[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        if (i // batch_size) % 10 == 0 or end_idx == len(ids):
            print(f"   -> Progress: Indexed {end_idx} / {len(ids)} documents...")

    print("\n==================================================")
    print("  ✅ CHROMADB INDEXATION COMPLETED IN SECONDS!")
    print(f"  Total Nodes Indexed: {collection.count()}")
    print(f"  Storage Location   : {os.path.abspath(CHROMA_DB_PATH)}")
    print("==================================================")

if __name__ == "__main__":
    main()
