"""
Phase 2 – Expansion sémantique : construction du graphe de références G_ref,
expansion récursive, injection des définitions, formatage structuré.
"""
import logging
import os
from typing import Dict, List, Optional, Set

import networkx as nx

from . import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Construction du graphe de références G_ref
# ══════════════════════════════════════════════════════════════════════

def build_reference_graph(
    G_lex: nx.DiGraph,
    resolution_results: Dict[str, Dict],
) -> nx.DiGraph:
    """
    Construit le graphe de références G_ref à partir des résultats
    du résolveur hybride.
    - Arêtes internes : source → target_node_id (type=internal)
    - Arêtes externes : source → EXTERNAL::id (type=external)
    """
    G_ref = nx.DiGraph()

    # Copier les noeuds de G_lex (non-placeholder)
    for node_id, data in G_lex.nodes(data=True):
        if data.get("node_type") != "placeholder":
            G_ref.add_node(node_id, **data)

    # Ajouter les arêtes de référence
    for source_id, result in resolution_results.items():
        if source_id not in G_ref.nodes:
            continue

        # Références internes
        for ref in result.get("internal_refs", []):
            target_id = ref.get("target_node_id")
            if target_id and target_id in G_ref.nodes:
                G_ref.add_edge(
                    source_id, target_id,
                    edge_type="internal",
                    confidence=ref.get("confidence", 0.5),
                    ref_type=ref.get("type", "unknown"),
                )

            # Plages d'articles
            if ref.get("type") == "internal_range":
                for tid in ref.get("target_node_ids", []):
                    if tid in G_ref.nodes:
                        G_ref.add_edge(
                            source_id, tid,
                            edge_type="internal",
                            confidence=ref.get("confidence", 0.5),
                            ref_type="article_range",
                        )

        # Références externes
        for ref in result.get("external_refs", []):
            ext_id = ref.get("ext_id", f"EXT::{ref.get('raw', 'unknown')}")
            if ext_id not in G_ref.nodes:
                G_ref.add_node(ext_id, node_type="external", raw=ref.get("raw", ""))
            G_ref.add_edge(
                source_id, ext_id,
                edge_type="external",
                confidence=ref.get("confidence", 0.5),
                ext_id=ext_id,
            )

    logger.info(
        f"G_ref construit : {G_ref.number_of_nodes()} noeuds, "
        f"{G_ref.number_of_edges()} arêtes de référence"
    )
    return G_ref


# ══════════════════════════════════════════════════════════════════════
# Expansion récursive
# ══════════════════════════════════════════════════════════════════════

MAX_EXPANDED_TEXT_CHARS = int(os.getenv("MAX_EXPANDED_TEXT_CHARS", "3000")) if os.getenv("MAX_EXPANDED_TEXT_CHARS") else 3000


def _truncate_text(text: str, max_chars: int = None) -> str:
    """Tronque un texte à max_chars en gardant le premier paragraphe complet."""
    if max_chars is None:
        max_chars = MAX_EXPANDED_TEXT_CHARS
    if len(text) <= max_chars:
        return text
    # Couper au premier paragraphe complet sous la limite
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars // 2:
        truncated = truncated[:last_newline]
    return truncated + "\n[... texte tronqué pour limiter le bruit]"


def expand_clause(
    node_id: str,
    G_ref: nx.DiGraph,
    G_lex: nx.DiGraph,
    definitions: List[Dict],
    depth: int = 0,
    max_depth: int = None,
    visited: Optional[Set[str]] = None,
    include_definitions: bool = True,
) -> str:
    """
    Expansion récursive d'une clause :
    1. Texte de la clause
    2. Pour chaque référence interne : texte de la clause référencée (récursif, tronqué si trop long)
    3. Pour chaque référence externe : bandeau si non disponible
    4. Injection des définitions si les termes apparaissent

    Limitation connue : si deux articles citent la même référence, le texte
    sera dupliqué dans les deux expansions (acceptable pour la vectorisation).

    Retourne le texte enrichi.
    """
    if max_depth is None:
        max_depth = config.MAX_EXPANSION_DEPTH

    if visited is None:
        visited = set()

    if depth > max_depth or node_id in visited:
        return ""

    visited.add(node_id)

    data = G_ref.nodes.get(node_id, {})
    text = data.get("full_text", "")
    if not text:
        text = G_lex.nodes.get(node_id, {}).get("full_text", "")

    if not text:
        return ""

    enriched = f"### {data.get('clause_id', node_id)}\n{text}\n"

    # Parcourir les références sortantes
    for target in G_ref.successors(node_id):
        edge = G_ref.edges[node_id, target]
        edge_type = edge.get("edge_type", "")

        if edge_type == "internal":
            target_data = G_ref.nodes.get(target, {})
            header = f"#### ↳ Réf. : {target_data.get('clause_id', target)}"
            target_text = target_data.get("full_text", "")
            if target_text:
                # Tronquer si le texte référencé est trop long
                target_text = _truncate_text(target_text)
                enriched += f"\n{header}\n{target_text}\n"
                # Récursion
                sub_text = expand_clause(
                    target, G_ref, G_lex, definitions,
                    depth + 1, max_depth, visited, include_definitions,
                )
                if sub_text:
                    enriched += sub_text

        elif edge_type == "external":
            ext_data = G_ref.nodes.get(target, {})
            ext_raw = ext_data.get("raw", "")
            if ext_raw:
                enriched += f"\n[Référence externe non disponible : {ext_raw}]\n"
            else:
                enriched += f"\n[Référence externe non disponible : {target}]\n"

    # Injection des définitions
    if include_definitions:
        text_lower = enriched.lower()
        for d in definitions:
            term_lower = d["term"].lower()
            if term_lower in text_lower and len(term_lower) > 3:
                enriched += f"\n*Définition ({d['term']}) : {d['definition']}*\n"

    return enriched


def expand_all_clauses(
    G_ref: nx.DiGraph,
    G_lex: nx.DiGraph,
    definitions: List[Dict],
    max_depth: int = None,
) -> Dict[str, str]:
    """
    Expansion de toutes les clauses du graphe.
    Retourne {node_id: enriched_text}.
    """
    if max_depth is None:
        max_depth = config.MAX_EXPANSION_DEPTH

    expanded = {}
    total = G_ref.number_of_nodes()
    for i, node_id in enumerate(G_ref.nodes):
        data = G_ref.nodes[node_id]
        if data.get("node_type") in ("placeholder", "external"):
            continue
        text = expand_clause(
            node_id, G_ref, G_lex, definitions,
            depth=0, max_depth=max_depth,
        )
        if text:
            expanded[node_id] = text
        if (i + 1) % 100 == 0:
            logger.info(f"Expansion : {i+1}/{total} clauses")

    logger.info(f"Expansion terminée : {len(expanded)} clauses enrichies")
    return expanded


# ══════════════════════════════════════════════════════════════════════
# Formatage pour indexation vectorielle
# ══════════════════════════════════════════════════════════════════════

def chunk_enriched_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Découpe un texte enrichi en chunks avec chevauchement.
    Découpe par paragraphes (###) puis par taille si nécessaire.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    # Découpe par sections (###)
    sections = text.split("### ")
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    current_chunk = ""

    for section in sections:
        section_text = "### " + section
        # Estimer la taille en tokens (~4 chars/token)
        est_tokens = len(current_chunk + section_text) // 4

        if est_tokens > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Chevauchement : garder les derniers caractères
            overlap_text = current_chunk[-chunk_overlap * 4:] if len(current_chunk) > chunk_overlap * 4 else current_chunk
            current_chunk = overlap_text + "\n" + section_text
        else:
            current_chunk += "\n" + section_text

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def prepare_indexing_data(
    expanded: Dict[str, str],
    G_ref: nx.DiGraph,
) -> List[Dict]:
    """
    Prépare les données pour l'indexation ChromaDB.
    Découpe en chunks et ajoute les métadonnées.
    """
    indexing_data = []

    for node_id, enriched_text in expanded.items():
        data = G_ref.nodes.get(node_id, {})
        chunks = chunk_enriched_text(enriched_text)

        # Références sortantes pour les métadonnées
        refs = [
            t for t in G_ref.successors(node_id)
            if G_ref.edges[node_id, t].get("edge_type") == "internal"
        ]
        ext_refs = [
            t for t in G_ref.successors(node_id)
            if G_ref.edges[node_id, t].get("edge_type") == "external"
        ]

        for i, chunk in enumerate(chunks):
            chunk_id = f"{node_id}__chunk_{i}"
            indexing_data.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "node_id": node_id,
                    "clause_id": data.get("clause_id", ""),
                    "title": data.get("title_or_summary", ""),
                    "country": data.get("country", ""),
                    "doc_title": data.get("doc_title", ""),
                    "category": data.get("category", ""),
                    "source_file": data.get("source_file", ""),
                    "node_type": data.get("node_type", ""),
                    "page_range": str(data.get("page_range", [])),
                    "refs": str(refs),
                    "external_refs": str(ext_refs),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            })

    logger.info(f"Données d'indexation : {len(indexing_data)} chunks préparés")
    return indexing_data
