"""
Phase 0 – Prétraitement : chargement JSON, normalisation des clauses,
construction du graphe hiérarchique G_lex et de l'index des articles.
"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from . import config

logger = logging.getLogger(__name__)


# ── Normalisation des clause_id ────────────────────────────────────────

def normalize_article_num(clause_id: str) -> Optional[int]:
    """
    Extrait le numéro d'article d'un clause_id.
    'Art. 16.' → 16, 'Article 16' → 16, 'Art. 9 alinéa 1' → 9
    Retourne None si ce n'est pas un article numéroté.
    """
    m = re.match(r"(?:Art\.?|Article)\s+(\d+)", clause_id, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def normalize_clause_id(clause_id: str) -> str:
    """Normalise un clause_id pour l'indexage : minuscule, sans espaces superflus."""
    return clause_id.strip().lower().replace("\u2011", "-").replace("\u00a0", " ")


def infer_node_type(clause_id: str, level: int) -> str:
    """Infère le type de noeud à partir du clause_id et du level."""
    cid_lower = clause_id.lower()
    if "titre" in cid_lower:
        return "titre"
    elif "chapitre" in cid_lower:
        return "chapitre"
    elif "section" in cid_lower:
        return "section"
    elif "annexe" in cid_lower:
        return "annexe"
    elif "alinéa" in cid_lower:
        return "alinéa"
    elif re.search(r"(?:art\.?|article)\s+\d+", cid_lower):
        return "article"
    elif level == 1:
        return "titre"
    elif level == 2:
        return "article"
    elif level >= 3:
        return "alinéa"
    return "clause"


# ── Chargement d'un document JSON ──────────────────────────────────────

def load_json_document(filepath: Path) -> Dict:
    """
    Charge un fichier JSON de document juridique.
    Retourne un dict avec metadata, clauses normalisées.
    Gère les deux formats observés :
      - Format complet : {document_metadata, clauses}
      - Format simplifié : {clauses} sans metadata
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("document_metadata", {})
    raw_clauses = data.get("clauses", [])
    # Certains documents ont un champ definitions structuré séparé
    raw_definitions = data.get("definitions", None)

    # Inférer le pays et le titre depuis le nom de fichier si metadata absente
    if not metadata.get("country"):
        stem = filepath.stem
        # Enlever les préfixes llama-extract-* ou new-
        clean = re.sub(r"^(llama-extract-[a-f0-9\-]+-|new-|enriched-)", "", stem)
        # Enlever les suffixes _compressed, _repaired
        clean = re.sub(r"(_compressed.*|_repaired)", "", clean)
        metadata["filename"] = filepath.name
        metadata["source_file"] = str(filepath)

    clauses = []
    for i, c in enumerate(raw_clauses):
        clause = {
            "clause_id": c.get("clause_id", f"clause_{i}"),
            "parent_id": c.get("parent_id"),
            "level": c.get("level", 1),
            "title_or_summary": c.get("title_or_summary", ""),
            "full_text": c.get("full_text", ""),
            "page_range": c.get("page_range", []),
            "cross_references": c.get("cross_references", []),
            "external_reference": c.get("external_reference"),
            "is_footnote": c.get("is_footnote", False),
            "footnote_text": c.get("footnote_text"),
            "node_type": infer_node_type(
                c.get("clause_id", f"clause_{i}"), c.get("level", 1)
            ),
            "seq_position": i,
            "source_file": str(filepath),
            "country": metadata.get("country", ""),
            "doc_title": metadata.get("title", ""),
        }
        clauses.append(clause)

    return {"metadata": metadata, "clauses": clauses, "structured_definitions": raw_definitions}


# ── Chargement de tous les documents ───────────────────────────────────

def load_all_documents() -> Dict[str, List[Dict]]:
    """
    Charge tous les JSON de toutes les catégories.
    Retourne {category: [doc_dict, ...]}.
    """
    all_docs = {}
    for cat_name, cat_dir in config.CATEGORIES.items():
        if not cat_dir.exists():
            logger.warning(f"Répertoire introuvable : {cat_dir}")
            continue
        docs = []
        for fp in sorted(cat_dir.glob("*.json")):
            try:
                doc = load_json_document(fp)
                doc["metadata"]["category"] = cat_name
                for c in doc["clauses"]:
                    c["category"] = cat_name
                docs.append(doc)
                logger.info(f"Chargé {fp.name} ({len(doc['clauses'])} clauses)")
            except Exception as e:
                logger.error(f"Erreur chargement {fp}: {e}")
        all_docs[cat_name] = docs
    return all_docs


# ── Construction du graphe lexical hiérarchique ────────────────────────

def build_lexical_graph(all_docs: Dict[str, List[Dict]]) -> nx.DiGraph:
    """
    Construit le graphe lexical hiérarchique G_lex.
    - Noeuds : chaque clause
    - Arêtes parent_id → enfant (hiérarchie)
    - Attributs : node_type, full_text, etc.
    """
    G = nx.DiGraph()

    for cat_name, docs in all_docs.items():
        for doc in docs:
            for c in doc["clauses"]:
                node_id = f"{cat_name}::{c['clause_id']}::{c['source_file']}"
                G.add_node(
                    node_id,
                    clause_id=c["clause_id"],
                    parent_id=c["parent_id"],
                    level=c["level"],
                    title_or_summary=c["title_or_summary"],
                    full_text=c["full_text"],
                    page_range=c["page_range"],
                    cross_references=c["cross_references"],
                    external_reference=c["external_reference"],
                    is_footnote=c["is_footnote"],
                    footnote_text=c["footnote_text"],
                    node_type=c["node_type"],
                    seq_position=c["seq_position"],
                    source_file=c["source_file"],
                    country=c["country"],
                    doc_title=c["doc_title"],
                    category=cat_name,
                )

                # Arête hiérarchique parent → enfant
                if c["parent_id"]:
                    parent_node_id = f"{cat_name}::{c['parent_id']}::{c['source_file']}"
                    if parent_node_id in G.nodes:
                        G.add_edge(
                            parent_node_id,
                            node_id,
                            edge_type="hierarchy",
                        )
                    else:
                        # Parent pas encore ajouté, on crée un noeud fantôme
                        G.add_node(parent_node_id, clause_id=c["parent_id"], node_type="placeholder")
                        G.add_edge(parent_node_id, node_id, edge_type="hierarchy")

    logger.info(f"G_lex construit : {G.number_of_nodes()} noeuds, {G.number_of_edges()} arêtes")
    return G


# ── Index des articles ─────────────────────────────────────────────────

def build_article_index(G: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Construit un index {numéro_article_normalisé → [node_ids]}.
    Permet de retrouver rapidement un article par numéro.
    """
    index: Dict[str, List[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "article":
            num = normalize_article_num(data.get("clause_id", ""))
            if num is not None:
                key = str(num)
                index.setdefault(key, []).append(node_id)
    logger.info(f"Index des articles : {len(index)} numéros uniques")
    return index


# ── Index séquentiel ───────────────────────────────────────────────────

def build_sequential_index(G: nx.DiGraph) -> Dict[str, int]:
    """
    Construit {node_id → position_séquentielle} pour un même document.
    Permet de résoudre les références relatives (précédent, suivant).
    """
    seq: Dict[str, int] = {}
    # Grouper par document source
    docs_nodes: Dict[str, List[Tuple[int, str]]] = {}
    for node_id, data in G.nodes(data=True):
        src = data.get("source_file", "unknown")
        pos = data.get("seq_position", 0)
        docs_nodes.setdefault(src, []).append((pos, node_id))

    for src, nodes in docs_nodes.items():
        nodes.sort(key=lambda x: x[0])
        for i, (_, node_id) in enumerate(nodes):
            seq[node_id] = i

    return seq


# ── Extraction des définitions ─────────────────────────────────────────

def extract_structured_definitions(raw_definitions: list, source_file: str, category: str) -> List[Dict]:
    """
    Extrait les définitions depuis un champ definitions structuré
    (format [{term, definition}, ...] ou {term: definition, ...}).
    """
    definitions = []
    if not raw_definitions:
        return definitions
    if isinstance(raw_definitions, list):
        for d in raw_definitions:
            if isinstance(d, dict) and d.get("term") and d.get("definition"):
                definitions.append({
                    "term": d["term"],
                    "definition": d["definition"],
                    "source_clause": d.get("source_clause", ""),
                    "source_file": source_file,
                    "category": category,
                })
            elif isinstance(d, dict):
                for term, definition in d.items():
                    if len(str(term)) > 2 and len(str(definition)) > 10:
                        definitions.append({
                            "term": str(term),
                            "definition": str(definition),
                            "source_clause": "",
                            "source_file": source_file,
                            "category": category,
                        })
    elif isinstance(raw_definitions, dict):
        for term, definition in raw_definitions.items():
            if len(str(term)) > 2 and len(str(definition)) > 10:
                definitions.append({
                    "term": str(term),
                    "definition": str(definition),
                    "source_clause": "",
                    "source_file": source_file,
                    "category": category,
                })
    return definitions


def extract_definitions_from_clauses(clauses: List[Dict]) -> List[Dict]:
    """
    Extrait les définitions légales des clauses de type définitions.
    Détecte les patterns du type "terme : définition" ou "terme — définition".
    """
    definitions = []
    for c in clauses:
        text = c.get("full_text", "")
        if not text:
            continue
        # Pattern typique : "— la chasse : la recherche..."
        # ou "- **Chasse**, action visant..."
        # On split par les tirets d'énumération
        items = re.split(r"(?:\n[—\-]|\n\*\*)", text)
        for item in items:
            # Chercher "terme : définition" ou "terme, définition"
            m = re.match(r"\*?\*?([^:,\n]{3,50})\*?\*?\s*[:，]\s*(.+)", item.strip(), re.DOTALL)
            if m:
                term = m.group(1).strip().strip("*")
                definition = m.group(2).strip()
                if len(term) > 2 and len(definition) > 10:
                    definitions.append({
                        "term": term,
                        "definition": definition,
                        "source_clause": c["clause_id"],
                        "source_file": c.get("source_file", ""),
                        "category": c.get("category", ""),
                    })
    return definitions


def build_definition_graph(definitions: List[Dict]) -> nx.DiGraph:
    """
    Construit le graphe de définitions G_def.
    Noeuds = termes, avec attribut definition.
    """
    G = nx.DiGraph()
    for d in definitions:
        node_id = f"def::{d['term'].lower()}::{d.get('source_file', '')}"
        G.add_node(
            node_id,
            term=d["term"],
            definition=d["definition"],
            source_clause=d["source_clause"],
            source_file=d.get("source_file", ""),
            category=d.get("category", ""),
        )
    logger.info(f"G_def construit : {G.number_of_nodes()} définitions")
    return G


# ── Pipeline Phase 0 complète ──────────────────────────────────────────

class PreprocessedData:
    """Conteneur pour toutes les structures de données prétraitées."""

    def __init__(self):
        self.all_docs: Dict[str, List[Dict]] = {}
        self.G_lex: nx.DiGraph = nx.DiGraph()
        self.G_def: nx.DiGraph = nx.DiGraph()
        self.article_index: Dict[str, List[str]] = {}
        self.seq_index: Dict[str, int] = {}
        self.definitions: List[Dict] = []
        self.clauses_dict: Dict[str, Dict] = {}  # node_id → clause data

    def build(self):
        """Exécute tout le pipeline Phase 0."""
        logger.info("═══ Phase 0 : Prétraitement ═══")

        # 1. Chargement
        self.all_docs = load_all_documents()

        # 2. Graphe lexical
        self.G_lex = build_lexical_graph(self.all_docs)

        # 3. Index
        self.article_index = build_article_index(self.G_lex)
        self.seq_index = build_sequential_index(self.G_lex)

        # 4. Définitions (priorité au champ structuré, fallback au parsing texte)
        self.definitions = []
        for cat_name, docs in self.all_docs.items():
            for doc in docs:
                # D'abord essayer le champ definitions structuré
                structured = doc.get("structured_definitions")
                if structured:
                    src = doc["clauses"][0].get("source_file", "") if doc["clauses"] else ""
                    defs = extract_structured_definitions(structured, src, cat_name)
                    if defs:
                        self.definitions.extend(defs)
                        continue
                # Fallback : parsing texte des clauses
                defs = extract_definitions_from_clauses(doc["clauses"])
                self.definitions.extend(defs)

        # 5. Graphe de définitions
        self.G_def = build_definition_graph(self.definitions)

        # 6. Dictionnaire des clauses (accès rapide)
        for node_id, data in self.G_lex.nodes(data=True):
            if data.get("node_type") != "placeholder":
                self.clauses_dict[node_id] = dict(data)

        # 7. Validation de la hiérarchie
        self._validate_hierarchy()

        logger.info(
            f"Phase 0 terminée : "
            f"{self.G_lex.number_of_nodes()} noeuds lexicaux, "
            f"{len(self.definitions)} définitions, "
            f"{len(self.article_index)} articles indexés"
        )
        return self

    def _validate_hierarchy(self):
        """Vérifie que chaque parent_id référencé existe dans le graphe."""
        orphan_count = 0
        for node_id, data in self.G_lex.nodes(data=True):
            if data.get("node_type") == "placeholder":
                # Noeud fantôme créé car un enfant référençait un parent pas encore ajouté
                children = list(self.G_lex.successors(node_id))
                logger.warning(
                    f"Parent fantôme '{data.get('clause_id', node_id)}' "
                    f"référencé par {len(children)} enfant(s) mais jamais défini"
                )
                orphan_count += 1
        if orphan_count > 0:
            logger.warning(f"Validation hiérarchie : {orphan_count} parent(s) fantôme(s) détecté(s)")
        else:
            logger.info("Validation hiérarchie : OK, aucun parent fantôme")
