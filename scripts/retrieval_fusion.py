"""
retrieval_fusion.py
Hybrid retrieval with Fusion scoring (conforme LATELL 2026):
    final_score = 0.50 * BM25 + 0.35 * Dense + 0.15 * Graph

Le modele d'embedding est detecte automatiquement selon la dimension
de l'index FAISS existant :
    dim=384  -> paraphrase-multilingual-MiniLM-L12-v2
    dim=1024 -> BAAI/bge-m3  (cible de l'article)
Produces a full RetrievalTrace for explainability.
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pickle
import os
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from scripts.models import RetrievalTrace

# -- Configuration -------------------------------------------------------------
INDEX_DIR     = os.environ.get("RETRIEVAL_INDEX_DIR", "indexes_global")
INDEX_PREFIX  = "legal"
RERANK_MODEL  = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
TOP_K_RETRIEVAL = 50   # retrieve before reranking
TOP_K_FINAL     = 20   # return after reranking

# Fusion weights - Conformes a la publication LATELL 2026 (§3.2)
W_BM25  = 0.50
W_DENSE = 0.35
W_GRAPH = 0.15

# Mapping dimension -> modele d'embedding (detection automatique)
_DIM_TO_MODEL = {
    384:  "paraphrase-multilingual-MiniLM-L12-v2",  # index existant
    768:  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    1024: "BAAI/bge-m3",                             # cible de l'article
}

# -- Lazy index loading --------------------------------------------------------
_bm25: Optional[BM25Okapi]    = None
_faiss_index                  = None
_graph: Optional[Dict]        = None
_nodes: Optional[List[Dict]]  = None
_id2idx: Optional[Dict]       = None
_embed_model                  = None
_reranker                     = None
_current_loaded_dir           = None


def _load_indexes():
    global _bm25, _faiss_index, _graph, _nodes, _id2idx, _embed_model, _reranker, _current_loaded_dir
    if _bm25 is not None and _current_loaded_dir == INDEX_DIR:
        return   # already loaded and same dir

    print("[Retrieval] Loading indexes...")
    with open(os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_bm25.pkl"), "rb") as f:
        _bm25 = pickle.load(f)

    embed_model_name = _DIM_TO_MODEL.get(1024, "BAAI/bge-m3")  # fallback
    try:
        faiss_path = os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_faiss.bin")
        if os.path.exists(faiss_path):
            _faiss_index = faiss.read_index(faiss_path)
            # Auto-detection du modele selon la dimension reelle de l'index
            actual_dim = _faiss_index.d
            embed_model_name = _DIM_TO_MODEL.get(actual_dim, "BAAI/bge-m3")
            print(f"[Retrieval] FAISS dim={actual_dim} -> modele: {embed_model_name}")
        else:
            print(f"[Retrieval] Warning: FAISS index not found in {INDEX_DIR}. Mode BM25 seul.")
            _faiss_index = None
    except Exception as e:
        print(f"[Retrieval] Warning: Chargement FAISS echoue: {e}")
        _faiss_index = None

    with open(os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_graph.pkl"), "rb") as f:
        _graph = pickle.load(f)
    with open(os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_nodes.pkl"), "rb") as f:
        _nodes = pickle.load(f)
    with open(os.path.join(INDEX_DIR, f"{INDEX_PREFIX}_id2idx.pkl"), "rb") as f:
        _id2idx = pickle.load(f)

    print(f"[Retrieval] Chargement embedding: {embed_model_name}")
    _embed_model = SentenceTransformer(embed_model_name)

    # Reranker : tenter offline d'abord, puis online, sinon desactiver
    print(f"[Retrieval] Chargement reranker: {RERANK_MODEL}")
    try:
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    except Exception as e:
        print(f"[Retrieval] Warning: Reranker non disponible ({type(e).__name__}). Fusion seule.")
        _reranker = None

    _current_loaded_dir = INDEX_DIR
    print(f"[Retrieval] [OK] Indexes prets ({INDEX_DIR})")


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.ones_like(scores)
    return (scores - mn) / (mx - mn)


# -- Individual retrievers -----------------------------------------------------
def _bm25_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    tokens = query.lower().split()
    scores = np.array(_bm25.get_scores(tokens))
    idxs   = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in idxs]


def _dense_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    if _embed_model is None:
        _load_indexes()
    vec = _embed_model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)
    dists, idxs = _faiss_index.search(vec, top_k)
    return [(int(i), float(d)) for i, d in zip(idxs[0], dists[0]) if i >= 0]


def _graph_expand(node_ids: List[str], hop: int = 1) -> Dict[str, float]:
    """
    Return a bonus score for nodes reachable from `node_ids` via the graph.
    Bonus decays by 0.5 per hop.
    """
    bonus: Dict[str, float] = {}
    current = set(node_ids)
    decay   = 1.0
    for _ in range(hop):
        decay *= 0.5
        next_nodes = set()
        for nid in current:
            for edge in _graph.get(nid, []):
                tgt = edge["target"]
                bonus[tgt] = bonus.get(tgt, 0.0) + decay * edge.get("weight", 1.0)
                next_nodes.add(tgt)
        current = next_nodes
    return bonus


# -- Semantic Query Expansion --------------------------------------------------
DOMAIN_SYNONYMS = {
    "baleine": [
        "cetace", "mammifere marin", "mammifere aquatique", "grand cetace",
        "mysticete", "odontocete", "balenoptere", "rorqual", "megaptere",
        "lamentin", "manatee", "dugong", "dauphin", "marsouin",
        "trichechus senegalensis", "sousa teuszii", "delphinidae",
        "classe A", "espece integralement protegee", "espece protegee",
        "especes menacees", "especes menacees d'extinction", "especes protegees",
        "faune sauvage", "faune marine", "organisme aquatique",
    ],
    "cetace": ["mammifere aquatique", "baleine", "mammifere marin",
               "especes menacees", "espece protegee"],
    "interdiction": ["interdit", "sont interdites", "est interdit", "interdire",
                     "il est defendu", "prohibe", "prohibition"],
    "sanction": ["amende", "emprisonnement", "peine", "punition", "infraction",
                 "poursuite", "saisie", "penalite", "puni", "est puni"],
    "prison": ["emprisonnement", "peine d'emprisonnement", "detention", "puni",
               "peine d'emprisonnement", "mois", "ans"],
    "infraction": ["emprisonnement", "amende", "punition", "poursuite",
                   "penalite", "puni", "est puni"],
    "peine": ["emprisonnement", "amende", "prison", "puni", "mois", "ans", "GNF", "tonneaux", "jauge brute", "dirhams", "dirham", "7 ans", "3 ans", "sept ans", "trois ans"],
    "controle": ["surveillance", "inspection", "police", "arraisonnement",
                 "observateur", "agent", "suivi", "veiller", "coordination",
                 "operations de suivi", "constatation", "arraisonner", "flagrant delit",
                 "constat", "verifier", "superviser", "inspection materielle",
                 "visites periodiques", "visite periodique"],
    "autorite": ["agent", "ministere", "administration", "institution",
                 "controle", "surveillance", "police", "coordination",
                 "administration maritime", "autorite maritime", "Direction de la Marine Marchande",
                 "Marine Marchande", "Direction de l'Environnement", "autorite competente"],
    "institution": ["ministere", "administration", "autorite", "agent",
                    "ministere en charge", "administration maritime", "Direction de la Marine Marchande",
                    "autorite competente"],
    "veiller": ["controle", "surveillance", "suivi", "application", "agent",
                "ministere", "coordination"],
    "procedure": ["disposition", "regle", "mesure", "controle", "surveillance",
                  "coordination", "constatation", "arraisonnement", "arraisonner",
                  "mise en demeure", "notification", "information immediate", "informer immediatement",
                  "compte rendu", "rapport"],
    "zone": ["eaux maritimes", "mer", "lagune", "littoral", "aire", "region",
             "eaux nationales", "eaux sous juridiction", "eaux maritimes marocaines",
             "plateau continental", "zone economique exclusive"],
    "navire": ["bateau", "batiment", "embarcation", "petrolier", "engin",
               "unite", "navires", "pavillon etranger", "navire etranger"],
    "loi": ["decret", "ordonnance", "arrete", "code", "texte", "dahir", "Dahir"],
    "superviser": ["surveillance", "suivi", "controle", "agent", "ministere",
                   "coordination", "responsable"],
    "evaluer": ["suivi", "controle", "surveillance", "inspection"],
    "constater": ["infraction", "controle", "agent", "inspection",
                  "constatation", "recherche"],
    "comite": ["administration", "commission", "autorite", "service",
               "ministere"],
    "service": ["administration", "ministere", "agent", "autorite"],
    "chasse": ["capture", "prelevement", "peche", "detention", "abattage",
               "commercialisation", "transport", "destruction"],
    "accidentelle": ["accidentellement", "remise a l'eau", "immediatement",
                     "vivante"],
    "exception": ["derogation", "autorisation", "sauf", "hormis", "a l'exception",
                  "sous reserve"],
    "hydrocarbure": ["polluant", "dechet", "contaminant", "produit chimique",
                     "matiere dangereuse", "substance toxique", "effluent",
                     "deversement", "pollution", "dechet polluant", "matiere polluante",
                     "contaminants"],
    "rejet": ["deversement", "emission", "depot", "degagement", "enfouissement",
              "deposer", "emettre", "rejeter", "jet", "jeter"],
}

def _expand_query(query: str) -> str:
    """Expands the query with domain-specific synonyms to improve recall."""
    lowered = query.lower()
    expanded_terms = set()
    for word, synonyms in DOMAIN_SYNONYMS.items():
        if word in lowered:
            expanded_terms.update(synonyms)
    
    if expanded_terms:
        expansion = " " + " ".join(expanded_terms)
        return query + expansion
    return query

# -- Fusion --------------------------------------------------------------------
def retrieve(query: str, 
             top_k_final: int = 10, 
             jurisdiction_filter: Optional[str] = None,
             theme_filter: Optional[str] = None) -> Tuple[List[Dict], RetrievalTrace]:
    """
    Main entry point for hybrid retrieval.
    """
    _load_indexes()
    trace = RetrievalTrace(query=query)

    # 1. Base Retrieval
    bm25_res = _bm25_retrieve(query, TOP_K_RETRIEVAL)
    dense_res = _dense_retrieve(query, TOP_K_RETRIEVAL)

    # Combine into a map
    combined: Dict[int, Dict] = {}
    
    # Process BM25
    for idx, score in bm25_res:
        combined[idx] = {"bm25": score, "dense": 0.0, "graph": 0.0}
    
    # Process Dense
    for idx, dist in dense_res:
        if idx not in combined:
            combined[idx] = {"bm25": 0.0, "dense": 0.0, "graph": 0.0}
        combined[idx]["dense"] = dist

    # 2. Filter by Jurisdiction and Theme
    filtered_indices = []
    for idx, scores in combined.items():
        node = _nodes[idx]
        
        # Filtre de juridiction
        if jurisdiction_filter:
            if node.get("country", "").lower() != jurisdiction_filter.lower():
                continue
        
        # Filtre de theme
        if theme_filter:
            node_theme = node.get("metadata", {}).get("theme", "").lower()
            if node_theme != theme_filter.lower():
                continue
                
        filtered_indices.append(idx)
    
    # Apply Semantic Expansion
    expanded_query = _expand_query(query)
    trace.reasoning = f"Expanded query: {expanded_query} | "

    # 1. BM25 (using expanded query for better keyword matching)
    bm25_hits = _bm25_retrieve(expanded_query, TOP_K_RETRIEVAL)
    bm25_raw  = np.zeros(len(_nodes))
    for idx, score in bm25_hits:
        bm25_raw[idx] = score
        trace.bm25_hits.append({"node_id": _nodes[idx]["node_id"], "score": score})

    # 2. Dense (using original query to preserve embedding semantics)
    dense_hits = _dense_retrieve(query, TOP_K_RETRIEVAL)
    dense_raw  = np.zeros(len(_nodes))
    for idx, score in dense_hits:
        dense_raw[idx] = score
        trace.dense_hits.append({"node_id": _nodes[idx]["node_id"], "score": score})

    # 3. Graph expansion from BM25 + Dense seeds
    seed_ids = [_nodes[i]["node_id"] for i, _ in bm25_hits[:5]]
    seed_ids += [_nodes[i]["node_id"] for i, _ in dense_hits[:5]]
    graph_bonus = _graph_expand(seed_ids, hop=1)
    graph_raw   = np.zeros(len(_nodes))
    for nid, bonus in graph_bonus.items():
        if nid in _id2idx:
            graph_raw[_id2idx[nid]] = bonus
            trace.graph_hops.append({"node_id": nid, "bonus": bonus})

    # 4. Normalise each signal
    bm25_norm  = _normalise(bm25_raw)
    dense_norm = _normalise(dense_raw)
    graph_norm = _normalise(graph_raw)

    # 5. Fusion de scores lineaire (conforme a la formule de l'article)
    fused = W_BM25 * bm25_norm + W_DENSE * dense_norm + W_GRAPH * graph_norm

    # 6. Jurisdiction filter
    if jurisdiction_filter:
        norm_filter = jurisdiction_filter.lower().strip()
        # Handle known jurisdiction aliases
        possible_matches = [norm_filter]
        if norm_filter in ["cameroun", "cameroon"]:
            possible_matches = ["cameroun", "cameroon"]
        elif norm_filter in ["comores", "comoros"]:
            possible_matches = ["comores", "comoros", "union des comores"]
        elif norm_filter in ["rdc", "congo", "rd congo", "republique democratique du congo"]:
            possible_matches = ["rdc", "congo", "rd congo", "republique democratique du congo"]
        elif norm_filter == "madagascar":
            possible_matches = ["madagascar"]
        elif norm_filter == "guinee":
            possible_matches = ["guinee", "guinee"]
        elif norm_filter == "benin":
            possible_matches = ["benin", "benin"]
        elif norm_filter == "mauritanie":
            possible_matches = ["mauritanie"]
        elif norm_filter == "senegal":
            possible_matches = ["senegal", "senegal"]
        elif norm_filter == "togo":
            possible_matches = ["togo"]
        elif norm_filter == "tunisie":
            possible_matches = ["tunisie"]
            
        for i, node in enumerate(_nodes):
            node_country = node.get("country", "").lower().strip()
            if not any(pm in node_country for pm in possible_matches):
                fused[i] = 0.0

    # Rank by fused score
    candidate_idxs = np.argsort(fused)[::-1][:TOP_K_RETRIEVAL]
    
    results = []
    for idx in candidate_idxs:
        if fused[idx] > 0:
            node = _nodes[idx].copy()
            node["_fused_score"] = float(fused[idx])
            results.append(node)
            
    # 4. Meta Boost : Donner un bonus aux articles dont le nom de loi contient le theme
    boost_keywords = ["baleine", "cetace", "protection", "arrete", "decret"]
    for idx, node in enumerate(results):
        law_name = node.get("law_name", "").lower()
        if any(kw in law_name for kw in boost_keywords):
            results[idx]["_fused_score"] *= 1.2
            
    # Re-sort after boost
    results = sorted(results, key=lambda x: x["_fused_score"], reverse=True)
    candidates = results[:top_k_final]

    # 8. Reranking (if enabled)
    if _reranker is not None and candidates:
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = _reranker.predict(pairs)
        
        # Sort by rerank score
        reranked_results = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for node, r_score in reranked_results:
            node_id = node["node_id"]
            orig_fused = node["_fused_score"]
            trace.final_scores[node_id] = {
                "fused": round(float(orig_fused), 4),
                "rerank": round(float(r_score), 4)
            }
            results.append({**node, "_fused_score": float(orig_fused), "_rerank_score": float(r_score)})
    else:
        # Fallback to fused scores only
        results = []
        for node in candidates[:top_k_final]:
            node_id = node["node_id"]
            orig_fused = fused[_id2idx[node_id]]
            trace.final_scores[node_id] = {
                "fused": round(float(orig_fused), 4),
                "rerank": 0.0
            }
            results.append({**node, "_fused_score": float(orig_fused), "_rerank_score": 0.0})

    trace.reasoning += (
        f"BM25 (×{W_BM25}) + Dense (×{W_DENSE}) + Graph (×{W_GRAPH}). "
        f"Reranking: {'Enabled' if _reranker else 'Disabled'}. "
        f"Jurisdiction filter: {jurisdiction_filter or 'None'}."
    )

    return results, trace


# -- Quick CLI test ------------------------------------------------------------
if __name__ == "__main__":
    query = "sanctions pour chasse illegale aux baleines"
    results, trace = retrieve(query, top_k_final=5)
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for r in results:
        print(f"  [{r['country']}] {r['node_id']}")
        print(f"    Summary : {r.get('summary', '')[:80]}")
        print(f"    Scores  : fused={r['_fused_score']:.4f}, rerank={r['_rerank_score']:.4f}\n")
    print("Retrieval Trace (JSON):")
    print(trace.to_json())
