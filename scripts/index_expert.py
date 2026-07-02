import json
import os
import glob
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pickle

class ExpertIndexer:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.nodes = []
        self.bm25 = None
        self.faiss_index = None

    def load_nodes(self, data_dir: str):
        print(f"Loading nodes from {data_dir}...")
        files = glob.glob(os.path.join(data_dir, "*.json"))
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.nodes.extend(data.get("nodes", []))
        print(f"Total nodes loaded: {len(self.nodes)}")

    def build_bm25(self):
        print("Building BM25 index...")
        tokenized_corpus = [node['text'].lower().split() for node in self.nodes]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def build_faiss(self):
        print("Building FAISS index...")
        texts = [node['text'] for node in self.nodes]
        self.model.max_seq_length = 1024
        embeddings = self.model.encode(texts, batch_size=16, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

    def save_indices(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # Save BM25
        with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        # Save FAISS
        faiss.write_index(self.faiss_index, os.path.join(output_dir, "faiss_index.bin"))
        # Save Nodes (to map index to node_id)
        with open(os.path.join(output_dir, "nodes.json"), "w", encoding='utf-8') as f:
            json.dump(self.nodes, f, ensure_ascii=False, indent=2)
        print(f"Indices saved to {output_dir}")

if __name__ == "__main__":
    indexer = ExpertIndexer()
    indexer.load_nodes("data_processed")
    indexer.build_bm25()
    indexer.build_faiss()
    indexer.save_indices("expert_index")
