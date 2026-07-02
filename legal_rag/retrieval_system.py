import logging
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .models import DocumentNode

logger = logging.getLogger(__name__)

# Optional heavy dependencies
LLAMA_INDEX_AVAILABLE = True
try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.embeddings.openai import OpenAIEmbedding

    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError:
        BM25Okapi = None
except ModuleNotFoundError:
    LLAMA_INDEX_AVAILABLE = False
    VectorStoreIndex = None
    TextNode = None
    NodeWithScore = None
    OpenAIEmbedding = None
    HuggingFaceEmbedding = None
    BM25Okapi = None


class LegalRetrievalSystem:
    """
    Hybrid retrieval system with Fusion scoring:
        final_score = 0.35 * BM25 + 0.50 * Dense + 0.15 * Graph
    
    Includes Traversal Boosting (cites boosted to 1.0) and Cross-Encoder Reranking.
    """

    def __init__(self):
        self.nodes_map: Dict[str, DocumentNode] = {}
        self.reference_graph: Dict[str, List[str]] = {}

        # Offline storage
        self._offline_index = None
        self._offline_cleaned_text_by_id: Dict[str, str] = {}

        # Online storage
        self.embed_model = None
        self.vector_index = None
        self.documents: List[Any] = []
        self._use_offline = True

        # Initialize CrossEncoder Reranker with fallback
        self._reranker = None
        try:
            from sentence_transformers import CrossEncoder
            # Safe CPU initialization to prevent hangs in sandbox
            self._reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", device="cpu")
            logger.info("Reranker loaded successfully: cross-encoder/ms-marco-TinyBERT-L-2-v2")
        except Exception as e:
            logger.warning(f"Reranker loading failed ({e}). Running in fusion-only mode.")

        # Embedding model loading
        if self._use_offline:
            logger.info("LegalRetrievalSystem: running in OFFLINE mode.")
            return

        try:
            if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("sk-" + "or"):
                self.embed_model = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
                logger.info("Using OpenAI embeddings")
            else:
                self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
                logger.info("Using HuggingFace embeddings for local processing")
        except Exception as e:
            logger.warning(f"Falling back to OFFLINE mode (embedding init failed): {e}")
            self._use_offline = True

    def index_documents(self, document_nodes: List[DocumentNode]):
        logger.info(f"Indexing {len(document_nodes)} document nodes...")
        self.nodes_map = {node.node_id: node for node in document_nodes}
        
        # Build reference graph for Graph score calculation
        self.reference_graph = {node.node_id: node.links_to for node in document_nodes}

        # Build offline BM25 index
        from .offline_rag import BM25Index, build_clean_corpus
        corpus = build_clean_corpus(document_nodes)
        self._offline_cleaned_text_by_id = {node.node_id: cleaned for node, cleaned in corpus}
        self._offline_index = BM25Index(corpus, k1=1.5, b=0.75)

        if self._use_offline:
            logger.info("Indexes ready (OFFLINE)")
            return

        if TextNode is None or VectorStoreIndex is None:
            self._use_offline = True
            logger.info("Indexes ready (OFFLINE fallback)")
            return

        # Prepare vector index
        self.documents = []
        for node in document_nodes:
            searchable_text = node.content
            if node.metadata and "keywords" in node.metadata:
                try:
                    keywords_str = " ".join(node.metadata["keywords"])
                    searchable_text = f"{node.content}\nKeywords: {keywords_str}"
                except Exception:
                    pass

            llama_node = TextNode(
                text=searchable_text,
                id_=node.node_id,
                metadata={
                    "node_type": node.node_type,
                    "page_number": node.page_number,
                    "parent_id": node.parent_id,
                    "links_to": node.links_to,
                    "footnotes": node.footnotes,
                    **(node.metadata or {}),
                },
            )
            self.documents.append(llama_node)

        try:
            self.vector_index = VectorStoreIndex(nodes=self.documents, embed_model=self.embed_model)
            logger.info("Indexes ready (ONLINE)")
        except Exception as e:
            logger.warning(f"Online indexing failed, switching to OFFLINE mode: {e}")
            self._use_offline = True

    def _normalise(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalise an array to [0, 1]."""
        if len(scores) == 0:
            return scores
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.ones_like(scores)
        return (scores - mn) / (mx - mn)

    def _graph_expand(self, seed_ids: List[str], hop: int = 1) -> Dict[str, float]:
        """
        Calculates a citation graph expansion score for nodes.
        Bonus decays by 0.5 per hop.
        """
        bonus: Dict[str, float] = {}
        current = set(seed_ids)
        decay = 1.0
        for _ in range(hop):
            decay *= 0.5
            next_nodes = set()
            for nid in current:
                links = self.reference_graph.get(nid, [])
                for target in links:
                    bonus[target] = bonus.get(target, 0.0) + decay
                    next_nodes.add(target)
            current = next_nodes
        return bonus

    def retrieve_with_fusion(self, query: str, top_k: int = None) -> Tuple[List[DocumentNode], List[Dict[str, Any]]]:
        """
        Performs hybrid score fusion, traversal boosting, and cross-encoder reranking.
        """
        top_k_final = top_k or settings.TOP_K_FINAL or 20
        if self._offline_index is None:
            return [], []

        # List all nodes to map index coordinates
        node_list = list(self.nodes_map.values())
        node_ids = [n.node_id for n in node_list]
        id2idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # 1. BM25 score
        bm25_raw = np.zeros(len(node_list))
        retrieved_bm25 = self._offline_index.retrieve(query, top_k=50)
        for r in retrieved_bm25:
            if r.node.node_id in id2idx:
                bm25_raw[id2idx[r.node.node_id]] = r.score

        # 2. Dense vector score
        dense_raw = np.zeros(len(node_list))
        if not self._use_offline and self.vector_index is not None:
            try:
                vector_retriever = self.vector_index.as_retriever(similarity_top_k=50)
                vector_results = vector_retriever.retrieve(query)
                for res in vector_results:
                    node_id = getattr(res.node, "id_", None) or getattr(res.node, "node_id", None)
                    if node_id and node_id in id2idx:
                        dense_raw[id2idx[node_id]] = float(getattr(res, "score", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"Vector search failed during fusion: {e}")

        # 3. Graph expansion score from BM25 + Dense top seeds
        bm25_top5_idxs = np.argsort(bm25_raw)[::-1][:5]
        dense_top5_idxs = np.argsort(dense_raw)[::-1][:5]
        
        seeds = [node_ids[idx] for idx in bm25_top5_idxs if bm25_raw[idx] > 0]
        seeds += [node_ids[idx] for idx in dense_top5_idxs if dense_raw[idx] > 0]
        
        graph_bonus = self._graph_expand(seeds, hop=1)
        graph_raw = np.zeros(len(node_list))
        for nid, bonus in graph_bonus.items():
            if nid in id2idx:
                graph_raw[id2idx[nid]] = bonus

        # 4. Normalize signals
        bm25_norm = self._normalise(bm25_raw)
        dense_norm = self._normalise(dense_raw)
        graph_norm = self._normalise(graph_raw)

        # 5. Weighted Score Fusion: (0.35 BM25 + 0.50 dense + 0.15 graphe)
        fused_scores = 0.35 * bm25_norm + 0.50 * dense_norm + 0.15 * graph_norm

        # Filter candidates (keep up to 50 positive candidates)
        candidate_idxs = np.argsort(fused_scores)[::-1][:50]
        candidates = []
        for idx in candidate_idxs:
            if fused_scores[idx] > 0:
                candidates.append((node_list[idx], float(fused_scores[idx])))

        # 6. Traversal Boosting (If A is retrieved and cites B, boost B to 1.0)
        cited_nodes = set()
        for node, score in candidates:
            if hasattr(node, "links_to") and node.links_to:
                for link in node.links_to:
                    # Match citations inside our nodes
                    for key in self.nodes_map.keys():
                        if key == link or key.endswith(f"Article_{link}") or f"Article_{link}" in key:
                            cited_nodes.add(key)
                            break

        boosted_candidates = []
        for node, score in candidates:
            if node.node_id in cited_nodes:
                logger.info(f"Traversal Boosting: Boosting cited node {node.node_id} to 1.0")
                boosted_candidates.append((node, 1.0))
            else:
                boosted_candidates.append((node, score))

        # Re-sort after boosting
        boosted_candidates = sorted(boosted_candidates, key=lambda x: x[1], reverse=True)

        # 7. Cross-Encoder Reranking (if available)
        if self._reranker is not None and boosted_candidates:
            try:
                pairs = [[query, item[0].content] for item in boosted_candidates]
                rerank_scores = self._reranker.predict(pairs)
                
                # Zip and sort by rerank score
                reranked_results = sorted(zip(boosted_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
                
                final_candidates = []
                fused_info = []
                for (node, fused_score), r_score in reranked_results[:top_k_final]:
                    final_candidates.append(node)
                    fused_info.append({
                        "node_id": node.node_id,
                        "score": float(fused_score),
                        "content": node.content,
                        "metadata": node.metadata,
                        "retriever": "reranked_hybrid_fusion",
                        "_rerank_score": float(r_score)
                    })
                return final_candidates, fused_info
            except Exception as e:
                logger.warning(f"Reranking execution failed: {e}. Falling back to fusion scores.")

        # Fallback (Reranker disabled or failed)
        final_candidates = []
        fused_info = []
        for node, score in boosted_candidates[:top_k_final]:
            final_candidates.append(node)
            fused_info.append({
                "node_id": node.node_id,
                "score": float(score),
                "content": node.content,
                "metadata": node.metadata,
                "retriever": "hybrid_fusion",
                "_rerank_score": 0.0
            })
            
        return final_candidates, fused_info

    def retrieve_by_node_ids(self, node_ids: List[str]) -> List[DocumentNode]:
        out = []
        for node_id in node_ids:
            node = self.nodes_map.get(node_id)
            if node:
                out.append(node)
        return out

    def retrieve_by_keywords_bm25(self, query: str, top_k: int = 10) -> List[DocumentNode]:
        if self._offline_index is None:
            return []
        retrieved = self._offline_index.retrieve(query, top_k=max(1, int(top_k)))
        return [r.node for r in retrieved]
