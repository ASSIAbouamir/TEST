"""
export_to_neo4j.py
==================
Exporte les 3 graphes du projet vers Neo4j Aura en ligne :

  1. Lexical Graph      → nœuds :LexicalNode  + relations [:CONTAINS / :REFERENCES / :HAS_FOOTNOTE]
  2. Definitions Graph  → nœuds :DefinitionNode + relations [:MEANS / :DEFINED_ON_PAGE]
  3. Retrieval Graph    → nœuds :RetrievalNode  + relations [:LINKED_TO]

Connexion : neo4j+s://b8632a91.databases.neo4j.io
"""

import json
import pickle
import os
import sys
import logging
from pathlib import Path

import neo4j
from neo4j import GraphDatabase

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(
        open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
    )]
)
log = logging.getLogger(__name__)

# ── Connexion Neo4j ───────────────────────────────────────────────────────────
URI      = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "LegalGraph2024!")

# ── Chemins des fichiers ──────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
LEXICAL_JSON     = BASE_DIR / "legal_rag" / "lexical_graph.json"
DEFINITIONS_JSON = BASE_DIR / "legal_rag" / "definitions_graph.json"
RETRIEVAL_PKL    = BASE_DIR / "indexes_all" / "legal_graph.pkl"
NODES_PKL        = BASE_DIR / "indexes_all" / "legal_nodes.pkl"

# ── Taille des batchs pour les requêtes Cypher ───────────────────────────────
BATCH_SIZE = 500


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════════════════
def _safe_str(val, max_len: int = 2000) -> str:
    """Tronque une valeur en chaîne propre pour Neo4j."""
    s = str(val) if val is not None else ""
    return s[:max_len]


def _batch(lst: list, size: int):
    """Découpe une liste en sous-listes de taille `size`."""
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def _run_batch(session, query: str, rows: list):
    """Exécute une requête Cypher en mode batch UNWIND."""
    for chunk in _batch(rows, BATCH_SIZE):
        session.run(query, rows=chunk)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Lexical Graph
# ══════════════════════════════════════════════════════════════════════════════
def export_lexical_graph(session):
    log.info("━━━ [1/3] Export du Lexical Graph ━━━")

    if not LEXICAL_JSON.exists():
        log.warning(f"Fichier introuvable : {LEXICAL_JSON}")
        return

    with open(LEXICAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    log.info(f"  Nœuds lexicaux : {len(nodes):,}  |  Arêtes : {len(edges):,}")

    # ── Contrainte d'unicité ──────────────────────────────────────────────────
    session.run(
        "CREATE CONSTRAINT lexical_node_id IF NOT EXISTS "
        "FOR (n:LexicalNode) REQUIRE n.node_id IS UNIQUE"
    )

    # ── Nœuds ─────────────────────────────────────────────────────────────────
    node_rows = []
    for n in nodes:
        nid = n.get("node_id") or n.get("id", "")
        if not nid:
            continue
        meta = n.get("metadata", {}) or {}
        node_rows.append({
            "node_id":       _safe_str(nid),
            "content":       _safe_str(n.get("content", ""), 2000),
            "node_type":     _safe_str(n.get("node_type", "paragraph")),
            "page_number":   int(n.get("page_number", 0)),
            "country":       _safe_str(meta.get("country", "")),
            "law_name":      _safe_str(meta.get("law_name", "")),
            "theme":         _safe_str(meta.get("theme", "")),
            "clause_id":     _safe_str(meta.get("clause_id", "")),
            "summary":       _safe_str(meta.get("summary", "")),
            "authority_level": float(meta.get("authority_level", 0.5)),
            "valid_from":    _safe_str(meta.get("valid_from", "")),
            "source_file":   _safe_str(meta.get("source_file", "")),
            "is_chunk":      bool(meta.get("is_chunk", False)),
            "chunk_index":   int(meta.get("chunk_index", 0)),
        })

    _run_batch(session,
        """
        UNWIND $rows AS r
        MERGE (n:LexicalNode {node_id: r.node_id})
        SET n.content        = r.content,
            n.node_type      = r.node_type,
            n.page_number    = r.page_number,
            n.country        = r.country,
            n.law_name       = r.law_name,
            n.theme          = r.theme,
            n.clause_id      = r.clause_id,
            n.summary        = r.summary,
            n.authority_level= r.authority_level,
            n.valid_from     = r.valid_from,
            n.source_file    = r.source_file,
            n.is_chunk       = r.is_chunk,
            n.chunk_index    = r.chunk_index
        """,
        node_rows
    )
    log.info(f"  ✓ {len(node_rows):,} nœuds LexicalNode créés/mis à jour")

    # ── Relations ──────────────────────────────────────────────────────────────
    # Mapping predicate → label de relation Neo4j
    PREDICATE_MAP = {
        "contains":     "CONTAINS",
        "references":   "REFERENCES",
        "has_footnote": "HAS_FOOTNOTE",
    }
    # Regrouper par type de relation pour des batchs efficaces
    rel_groups: dict[str, list] = {}
    for e in edges:
        pred   = e.get("predicate", "references").lower()
        rel_lbl = PREDICATE_MAP.get(pred, "REFERENCES")
        row = {
            "source":     _safe_str(e.get("source", "")),
            "target":     _safe_str(e.get("target", "")),
            "confidence": float(e.get("confidence", 1.0)),
        }
        rel_groups.setdefault(rel_lbl, []).append(row)

    total_rels = 0
    for rel_lbl, rows in rel_groups.items():
        _run_batch(session,
            f"""
            UNWIND $rows AS r
            MATCH (a:LexicalNode {{node_id: r.source}})
            MATCH (b:LexicalNode {{node_id: r.target}})
            MERGE (a)-[rel:{rel_lbl}]->(b)
            SET rel.confidence = r.confidence
            """,
            rows
        )
        total_rels += len(rows)
        log.info(f"    → {len(rows):,} relations :{rel_lbl}")

    log.info(f"  ✓ {total_rels:,} relations lexicales créées")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Definitions Graph
# ══════════════════════════════════════════════════════════════════════════════
def export_definitions_graph(session):
    log.info("━━━ [2/3] Export du Definitions Graph ━━━")

    if not DEFINITIONS_JSON.exists():
        log.warning(f"Fichier introuvable : {DEFINITIONS_JSON}")
        return

    with open(DEFINITIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    log.info(f"  Nœuds définitions : {len(nodes):,}  |  Arêtes : {len(edges):,}")

    # ── Contrainte ────────────────────────────────────────────────────────────
    session.run(
        "CREATE CONSTRAINT def_node_id IF NOT EXISTS "
        "FOR (n:DefinitionNode) REQUIRE n.node_id IS UNIQUE"
    )

    # ── Nœuds complets (avec term/definition) ─────────────────────────────────
    full_nodes = [n for n in nodes if n.get("term")]
    raw_nodes  = [n for n in nodes if not n.get("term")]  # nœuds cibles simples

    full_rows = [{
        "node_id":     _safe_str(n.get("node_id") or n["id"]),
        "term":        _safe_str(n.get("term", "")),
        "definition":  _safe_str(n.get("definition", ""), 2000),
        "source_page": int(n.get("source_page", 0)),
        "context":     _safe_str(n.get("context", ""), 1000),
    } for n in full_nodes]

    raw_rows = [{
        "node_id": _safe_str(n["id"]),
    } for n in raw_nodes if n.get("id")]

    _run_batch(session,
        """
        UNWIND $rows AS r
        MERGE (n:DefinitionNode {node_id: r.node_id})
        SET n.term        = r.term,
            n.definition  = r.definition,
            n.source_page = r.source_page,
            n.context     = r.context
        """,
        full_rows
    )

    _run_batch(session,
        """
        UNWIND $rows AS r
        MERGE (n:DefinitionNode {node_id: r.node_id})
        """,
        raw_rows
    )
    log.info(f"  ✓ {len(full_rows)+len(raw_rows):,} nœuds DefinitionNode créés")

    # ── Relations ──────────────────────────────────────────────────────────────
    PRED_MAP = {
        "means":           "MEANS",
        "defined_on_page": "DEFINED_ON_PAGE",
    }
    rel_groups: dict[str, list] = {}
    for e in edges:
        pred    = e.get("predicate", "means").lower()
        rel_lbl = PRED_MAP.get(pred, "MEANS")
        row = {
            "source":     _safe_str(e.get("source", "")),
            "target":     _safe_str(e.get("target", "")),
            "confidence": float(e.get("confidence", 1.0)),
        }
        rel_groups.setdefault(rel_lbl, []).append(row)

    total_rels = 0
    for rel_lbl, rows in rel_groups.items():
        _run_batch(session,
            f"""
            UNWIND $rows AS r
            MATCH (a:DefinitionNode {{node_id: r.source}})
            MATCH (b:DefinitionNode {{node_id: r.target}})
            MERGE (a)-[rel:{rel_lbl}]->(b)
            SET rel.confidence = r.confidence
            """,
            rows
        )
        total_rels += len(rows)
        log.info(f"    → {len(rows):,} relations :{rel_lbl}")

    log.info(f"  ✓ {total_rels:,} relations de définition créées")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Retrieval Graph (depuis legal_graph.pkl + legal_nodes.pkl)
# ══════════════════════════════════════════════════════════════════════════════
def export_retrieval_graph(session):
    log.info("━━━ [3/3] Export du Retrieval Graph ━━━")

    if not RETRIEVAL_PKL.exists():
        log.warning(f"Fichier introuvable : {RETRIEVAL_PKL}")
        return
    if not NODES_PKL.exists():
        log.warning(f"Fichier introuvable : {NODES_PKL}")
        return

    with open(NODES_PKL, "rb") as f:
        all_nodes: list[dict] = pickle.load(f)
    with open(RETRIEVAL_PKL, "rb") as f:
        graph: dict = pickle.load(f)   # { node_id → [{"target": ..., "weight": ...}] }

    log.info(f"  Nœuds retrieval : {len(all_nodes):,}  |  Nœuds avec voisins : {len(graph):,}")

    # ── Contrainte ────────────────────────────────────────────────────────────
    session.run(
        "CREATE CONSTRAINT retrieval_node_id IF NOT EXISTS "
        "FOR (n:RetrievalNode) REQUIRE n.node_id IS UNIQUE"
    )

    # ── Nœuds ─────────────────────────────────────────────────────────────────
    node_rows = []
    for n in all_nodes:
        nid = n.get("node_id", "")
        if not nid:
            continue
        node_rows.append({
            "node_id":        _safe_str(nid),
            "text":           _safe_str(n.get("text", ""), 2000),
            "summary":        _safe_str(n.get("summary", ""), 500),
            "country":        _safe_str(n.get("country", "")),
            "law_name":       _safe_str(n.get("law_name", "")),
            "authority_level": float(n.get("authority_level", 0.5)),
            "valid_from":     _safe_str(n.get("valid_from", "")),
        })

    _run_batch(session,
        """
        UNWIND $rows AS r
        MERGE (n:RetrievalNode {node_id: r.node_id})
        SET n.text            = r.text,
            n.summary         = r.summary,
            n.country         = r.country,
            n.law_name        = r.law_name,
            n.authority_level = r.authority_level,
            n.valid_from      = r.valid_from
        """,
        node_rows
    )
    log.info(f"  ✓ {len(node_rows):,} nœuds RetrievalNode créés")

    # ── Relations ──────────────────────────────────────────────────────────────
    edge_rows = []
    for src_id, neighbors in graph.items():
        if not isinstance(neighbors, list):
            continue
        for edge in neighbors:
            tgt = edge.get("target", "")
            if tgt:
                edge_rows.append({
                    "source": _safe_str(src_id),
                    "target": _safe_str(tgt),
                    "weight": float(edge.get("weight", 1.0)),
                })

    _run_batch(session,
        """
        UNWIND $rows AS r
        MATCH (a:RetrievalNode {node_id: r.source})
        MATCH (b:RetrievalNode {node_id: r.target})
        MERGE (a)-[rel:LINKED_TO]->(b)
        SET rel.weight = r.weight
        """,
        edge_rows
    )
    log.info(f"  ✓ {len(edge_rows):,} relations :LINKED_TO créées")


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée principal
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=== Connexion a Neo4j Aura ===")
    try:
        driver = GraphDatabase.driver(
            URI,                       # neo4j+ssc:// = TLS sans verif certificat
            auth=(USERNAME, PASSWORD),
        )
        driver.verify_connectivity()
        log.info("[OK] Connexion reussie !")
    except Exception as e:
        log.error(f"[ERREUR] Echec de connexion : {e}")
        sys.exit(1)

    with driver.session() as session:
        # ── 1. Lexical Graph ──────────────────────────────────────────────────
        try:
            export_lexical_graph(session)
        except Exception as e:
            log.error(f"Erreur export Lexical Graph : {e}")

        # ── 2. Definitions Graph ──────────────────────────────────────────────
        try:
            export_definitions_graph(session)
        except Exception as e:
            log.error(f"Erreur export Definitions Graph : {e}")

        # ── 3. Retrieval Graph ────────────────────────────────────────────────
        try:
            export_retrieval_graph(session)
        except Exception as e:
            log.error(f"Erreur export Retrieval Graph : {e}")

    driver.close()
    log.info("")
    log.info("=== [DONE] Export termine ! ===")
    log.info("Verifiez votre base sur : https://console.neo4j.io/projects/47c0c4fc-3ea1-4590-b9d4-c8e77e75ef47/developer-hub")


if __name__ == "__main__":
    main()