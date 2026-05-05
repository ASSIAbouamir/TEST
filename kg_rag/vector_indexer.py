"""
Phase 3 – Indexation vectorielle : ChromaDB, embeddings multilingual-e5-large,
stockage des collections enrichies et externes.
"""
import logging
from typing import Dict, List, Optional, Any

from . import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Embedding avec sentence-transformers
# ══════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """Moteur d'embedding utilisant sentence-transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Chargement du modèle d'embedding : {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Modèle d'embedding chargé")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode une liste de textes en embeddings."""
        self._load_model()
        # Préfixe "query: " ou "" selon le modèle e5
        prefixed = [f"query: {t}" if "e5" in self.model_name else t for t in texts]
        # Tronquer les textes trop longs pour éviter OOM
        max_length = 512
        truncated = [t[:max_length] if len(t) > max_length else t for t in prefixed]
        embeddings = self._model.encode(truncated, show_progress_bar=True, batch_size=4)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encode une requête unique."""
        self._load_model()
        prefix = "query: " if "e5" in self.model_name else ""
        embedding = self._model.encode([prefix + query])
        return embedding.tolist()[0]


# ══════════════════════════════════════════════════════════════════════
# ChromaDB Vector Store
# ══════════════════════════════════════════════════════════════════════

class ChromaStore:
    """Gestionnaire de collections ChromaDB."""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or str(config.CHROMA_PERSIST_DIR)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def get_or_create_collection(self, name: str):
        """Récupère ou crée une collection ChromaDB."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_collection(
        self, data: List[Dict[str, Any]], collection_name: str
    ) -> None:
        """Construit une collection ChromaDB à partir des données."""
        collection = self.get_or_create_collection(collection_name)
        if not data:
            logger.warning(f"Aucune donnée pour la collection {collection_name}")
            return

        texts = [d["text"] for d in data]
        metadatas = [d["metadata"] for d in data]
        ids = [d["id"] for d in data]

        # Utiliser l'engine d'embedding
        engine = EmbeddingEngine()
        engine._load_model()  # Forcer le chargement

        # Embeddings par batch très petits pour économiser mémoire
        embeddings = []
        batch_size = 2  # Réduit drastiquement pour éviter OOM
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Tronquer les textes trop longs
            max_length = 512
            truncated_batch = [t[:max_length] if len(t) > max_length else t for t in batch]
            batch_emb = engine._model.encode(
                truncated_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1,  # Traitement un par un pour économiser mémoire
            )
            embeddings.extend(batch_emb)
            # Forcer garbage collection
            del batch_emb, truncated_batch
            import gc
            gc.collect()

        # Ajout par petits lots pour éviter la mémoire
        add_batch_size = 50  # Réduit pour plus de sécurité
        for i in range(0, len(ids), add_batch_size):
            batch_ids = ids[i:i+add_batch_size]
            batch_texts = texts[i:i+add_batch_size]
            batch_metas = metadatas[i:i+add_batch_size]
            batch_embeddings = embeddings[i:i+add_batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
                embeddings=batch_embeddings
            )
            # Nettoyage mémoire
            del batch_ids, batch_texts, batch_metas, batch_embeddings
            import gc
            gc.collect()

        logger.info(f"Collection {collection_name} : {len(data)} documents indexés")

    def index_enriched_articles(self, indexing_data: List[Dict], batch_size: int = 100):
        """
        Indexe les articles enrichis dans la collection principale.
        """
        self.build_collection(indexing_data, config.CHROMA_COLLECTION_ENRICHED)

    def index_external_laws(self, all_docs: Dict, batch_size: int = 100):
        """
        Indexe toutes les clauses de tous les documents comme base externe
        pour la résolution de références externes.
        """
        collection = self.get_or_create_collection(config.CHROMA_COLLECTION_EXTERNAL)
        engine = EmbeddingEngine()

        external_data = []
        for cat_name, docs in all_docs.items():
            for doc in docs:
                metadata = doc.get("metadata", {})
                for c in doc.get("clauses", []):
                    if c.get("full_text"):
                        ext_id = f"EXT::{metadata.get('country', '')}::{c.get('clause_id', '')}::{c.get('source_file', '')}"
                        external_data.append({
                            "id": ext_id,
                            "text": c["full_text"],
                            "metadata": {
                                "clause_id": c.get("clause_id", ""),
                                "country": metadata.get("country", ""),
                                "title": metadata.get("title", ""),
                                "category": cat_name,
                                "source_file": c.get("source_file", ""),
                            },
                        })

        total = len(external_data)
        for i in range(0, total, batch_size):
            batch = external_data[i:i + batch_size]
            ids = [d["id"] for d in batch]
            texts = [d["text"] for d in batch]
            metadatas = [d["metadata"] for d in batch]
            embeddings = engine.embed_texts(texts)

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Index externe : {min(i + batch_size, total)}/{total}")

        logger.info(f"Collection '{config.CHROMA_COLLECTION_EXTERNAL}' : {collection.count()} documents")

    def index_definitions(self, definitions: List[Dict], batch_size: int = 100):
        """Indexe les définitions dans une collection séparée."""
        collection = self.get_or_create_collection(config.CHROMA_COLLECTION_DEFINITIONS)
        engine = EmbeddingEngine()

        def_data = []
        for d in definitions:
            text = f"{d['term']} : {d['definition']}"
            def_id = f"DEF::{d['term'].lower()}::{d.get('source_file', '')}"
            def_data.append({
                "id": def_id,
                "text": text,
                "metadata": {
                    "term": d["term"],
                    "source_clause": d.get("source_clause", ""),
                    "category": d.get("category", ""),
                },
            })

        for i in range(0, len(def_data), batch_size):
            batch = def_data[i:i + batch_size]
            ids = [d["id"] for d in batch]
            texts = [d["text"] for d in batch]
            metadatas = [d["metadata"] for d in batch]
            embeddings = engine.embed_texts(texts)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        logger.info(f"Collection '{config.CHROMA_COLLECTION_DEFINITIONS}' : {collection.count()} documents")

    def query(
        self,
        query_text: str,
        collection_name: str = None,
        n_results: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Recherche vectorielle dans une collection.
        Retourne une liste de {id, text, metadata, distance}.
        """
        collection_name = collection_name or config.CHROMA_COLLECTION_ENRICHED
        n_results = n_results or config.TOP_K_INITIAL

        collection = self.get_or_create_collection(collection_name)
        engine = EmbeddingEngine()
        query_embedding = engine.embed_query(query_text)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        docs = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

        return docs


# ══════════════════════════════════════════════════════════════════════
# BM25 Retriever
# ══════════════════════════════════════════════════════════════════════

class BM25Retriever:
    """Retriever BM25 pour la recherche par mots-clés."""

    def __init__(self, documents: List[Dict]):
        """
        documents : liste de {id, text, metadata}
        """
        from rank_bm25 import BM25Okapi
        self.documents = documents
        self.doc_ids = [d["id"] for d in documents]
        self.doc_texts = [d["text"] for d in documents]

        # Tokenisation simple (split + lower)
        tokenized_corpus = [self._tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenisation simple pour BM25."""
        import re
        # Garder les mots et les nombres
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def query(self, query_text: str, top_k: int = None) -> List[Dict]:
        """Recherche BM25."""
        top_k = top_k or config.TOP_K_BM25
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Trier par score décroissant
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append({
                    "id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "metadata": self.documents[idx].get("metadata", {}),
                    "bm25_score": float(score),
                })
        return results


# ══════════════════════════════════════════════════════════════════════
# Hybrid Retriever (Vector + BM25)
# ══════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """Combine les résultats vectoriels et BM25 avec fusion de scores."""

    def __init__(self, chroma_store: ChromaStore, bm25_retriever: BM25Retriever):
        self.chroma = chroma_store
        self.bm25 = bm25_retriever

    def query(
        self,
        query_text: str,
        top_k: int = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[Dict]:
        """
        Recherche hybride : combine vectoriel et BM25.
        Normalise les scores et fusionne.
        """
        top_k = top_k or config.TOP_K_INITIAL

        # Recherche vectorielle
        vector_results = self.chroma.query(query_text, n_results=top_k * 2)

        # Recherche BM25
        bm25_results = self.bm25.query(query_text, top_k=top_k * 2)

        # Fusion des scores
        # Normaliser les distances vectorielles (cosine distance → similarité)
        max_dist = max((d["distance"] for d in vector_results), default=1) or 1
        min_dist = min((d["distance"] for d in vector_results), default=0)

        scored = {}
        for d in vector_results:
            doc_id = d["id"]
            # Convertir distance en score (0-1)
            norm_score = 1 - (d["distance"] - min_dist) / (max_dist - min_dist + 1e-6)
            scored[doc_id] = {
                "id": doc_id,
                "text": d["text"],
                "metadata": d["metadata"],
                "vector_score": norm_score,
                "bm25_score": 0.0,
            }

        # Normaliser les scores BM25
        max_bm25 = max((d["bm25_score"] for d in bm25_results), default=1) or 1
        for d in bm25_results:
            doc_id = d["id"]
            norm_bm25 = d["bm25_score"] / max_bm25
            if doc_id in scored:
                scored[doc_id]["bm25_score"] = norm_bm25
            else:
                scored[doc_id] = {
                    "id": doc_id,
                    "text": d["text"],
                    "metadata": d["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": norm_bm25,
                }

        # Score combiné
        for doc_id, d in scored.items():
            d["combined_score"] = (
                vector_weight * d["vector_score"]
                + bm25_weight * d["bm25_score"]
            )

        # Trier par score combiné
        ranked = sorted(scored.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked[:top_k]
