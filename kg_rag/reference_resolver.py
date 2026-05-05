"""
Phase 1 – Résolution hybride de références.
4 étages en cascade : Regex → Heuristiques graphe → Externe → Fallback NER/LLM.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx

from . import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Étage 1 – Regex d'extraction
# ══════════════════════════════════════════════════════════════════════

# Patterns regex pour le français juridique
RE_ARTICLE_RANGE = re.compile(
    r"articles?\s+(\d+)\s*(?:à|et)\s+(\d+)", re.IGNORECASE
)
RE_ARTICLE_SINGLE = re.compile(
    r"(?:articles?|art\.)\s+(\d+)(?:\s*(?:ci[-\s]dessus|ci[-\s]après|suivant|précédent))?",
    re.IGNORECASE,
)
RE_ALINEA = re.compile(r"alinéa\s+(\d+)", re.IGNORECASE)
RE_PARAGRAPHE = re.compile(r"paragraphe\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
RE_ANNEXE = re.compile(r"annexe\s+([IVXLCDM]+|[A-Z])", re.IGNORECASE)
RE_FOOTNOTE_CALL = re.compile(r"\[(\d+)\]|note\s+(\d+)", re.IGNORECASE)
RE_TITRE = re.compile(r"titre\s+([IVXLCDM]+)", re.IGNORECASE)
RE_CHAPITRE = re.compile(r"chapitre\s+([IVXLCDM]+)", re.IGNORECASE)
RE_SECTION = re.compile(r"section\s+(\d+)", re.IGNORECASE)

# Références externes (lois, décrets, codes)
RE_EXTERNAL_LAW = re.compile(
    r"(?:loi|décret|ordonnance)\s+n[°o]\s*([\d\-]+)", re.IGNORECASE
)
RE_EXTERNAL_CODE = re.compile(
    r"code\s+(?:de\s+)?(?:la\s+|l['\u2019])?([\w\s]+?)(?:\s*[;,\.]|$)",
    re.IGNORECASE,
)


def extract_references_regex(text: str) -> List[Dict]:
    """
    Étage 1 : extraction par regex de toutes les références dans un texte.
    Retourne une liste de dicts {type, value, raw, confidence}.
    """
    refs = []

    # 1. Plages d'articles (priorité haute sur article single)
    for m in RE_ARTICLE_RANGE.finditer(text):
        refs.append({
            "type": "article_range",
            "value": (int(m.group(1)), int(m.group(2))),
            "raw": m.group(0),
            "confidence": 0.95,
        })

    # 2. Articles individuels (éviter les doublons avec les plages)
    range_spans = set()
    for m in RE_ARTICLE_RANGE.finditer(text):
        range_spans.add((m.start(), m.end()))

    for m in RE_ARTICLE_SINGLE.finditer(text):
        # Vérifier que ce match n'est pas dans une plage déjà capturée
        in_range = any(s <= m.start() < e for s, e in range_spans)
        if not in_range:
            qualifier = None
            qual_match = re.search(r"(ci[-\s]dessus|ci[-\s]après|suivant|précédent)", m.group(0), re.IGNORECASE)
            if qual_match:
                qualifier = qual_match.group(1).lower().replace("\u2011", "-").replace(" ", "-")
            refs.append({
                "type": "article",
                "value": int(m.group(1)),
                "raw": m.group(0),
                "confidence": 0.9,
                "qualifier": qualifier,
            })

    # 3. Alinéas
    for m in RE_ALINEA.finditer(text):
        refs.append({
            "type": "alinea",
            "value": int(m.group(1)),
            "raw": m.group(0),
            "confidence": 0.85,
        })

    # 4. Paragraphes
    for m in RE_PARAGRAPHE.finditer(text):
        refs.append({
            "type": "paragraphe",
            "value": m.group(1),
            "raw": m.group(0),
            "confidence": 0.85,
        })

    # 5. Annexes
    for m in RE_ANNEXE.finditer(text):
        refs.append({
            "type": "annexe",
            "value": m.group(1),
            "raw": m.group(0),
            "confidence": 0.9,
        })

    # 6. Notes de bas de page
    for m in RE_FOOTNOTE_CALL.finditer(text):
        num = m.group(1) or m.group(2)
        refs.append({
            "type": "footnote",
            "value": int(num),
            "raw": m.group(0),
            "confidence": 0.9,
        })

    # 7. Titres
    for m in RE_TITRE.finditer(text):
        refs.append({
            "type": "titre",
            "value": m.group(1),
            "raw": m.group(0),
            "confidence": 0.85,
        })

    # 8. Chapitres
    for m in RE_CHAPITRE.finditer(text):
        refs.append({
            "type": "chapitre",
            "value": m.group(1),
            "raw": m.group(0),
            "confidence": 0.85,
        })

    # 9. Références externes – lois
    for m in RE_EXTERNAL_LAW.finditer(text):
        refs.append({
            "type": "external_law",
            "value": m.group(1),
            "raw": m.group(0),
            "confidence": 0.8,
        })

    # 10. Références externes – codes
    for m in RE_EXTERNAL_CODE.finditer(text):
        code_name = m.group(1).strip()
        if len(code_name) > 3:
            refs.append({
                "type": "external_code",
                "value": code_name,
                "raw": m.group(0),
                "confidence": 0.7,
            })

    return refs


# ══════════════════════════════════════════════════════════════════════
# Étage 2 – Heuristiques de graphe pour anaphores
# ══════════════════════════════════════════════════════════════════════

ANAPHORE_KEYWORDS = {
    "précédent": "previous",
    "ci-dessus": "above",
    "ci dessus": "above",
    "ci-avant": "above",
    "ci après": "below",
    "ci-après": "below",
    "ci-apres": "below",
    "suivant": "next",
    "susvisé": "aforementioned",
    "susvise": "aforementioned",
    "présent": "current",
    "même": "same",
    "audit": "aforementioned",
    "ladite": "aforementioned",
    "icelui": "aforementioned",
}

# Expressions déclencheuses pour le fallback LLM
# Seules les clauses contenant ces expressions seront envoyées au LLM
LLM_TRIGGER_EXPRESSIONS = [
    "précédent", "ci-dessus", "ci-après", "susvisé", "susvise",
    "audit", "ladite", "icelui", "même article", "présent article",
    "modalités", "conditions prévues", "dispositions du présent",
    "renvoyant à", "voir également", "conformément à",
]


def find_ancestor_of_type(
    G: nx.DiGraph, node_id: str, target_type: str
) -> Optional[str]:
    """Remonte la hiérarchie pour trouver l'ancêtre d'un type donné."""
    current = node_id
    visited = set()
    while current and current not in visited:
        visited.add(current)
        data = G.nodes.get(current, {})
        if data.get("node_type") == target_type:
            return current
        # Remonter via les arêtes hiérarchiques
        parents = list(G.predecessors(current))
        hierarchy_parents = [
            p for p in parents
            if G.edges[p, current].get("edge_type") == "hierarchy"
        ]
        if hierarchy_parents:
            current = hierarchy_parents[0]
        else:
            break
    return None


def find_sibling_by_offset(
    G: nx.DiGraph,
    node_id: str,
    seq_index: Dict[str, int],
    offset: int,
    same_type: bool = True,
) -> Optional[str]:
    """
    Trouve un frère (même parent) décalé de `offset` positions.
    offset=-1 → précédent, offset=1 → suivant.
    """
    data = G.nodes.get(node_id, {})
    node_type = data.get("node_type")

    # Trouver le parent
    parents = [
        p for p in G.predecessors(node_id)
        if G.edges[p, node_id].get("edge_type") == "hierarchy"
    ]
    if not parents:
        return None

    parent = parents[0]
    siblings = list(G.successors(parent))
    siblings = [
        s for s in siblings
        if G.edges[parent, s].get("edge_type") == "hierarchy"
    ]
    # Trier par position séquentielle
    siblings.sort(key=lambda s: seq_index.get(s, 0))

    try:
        idx = siblings.index(node_id)
        target_idx = idx + offset
        if 0 <= target_idx < len(siblings):
            candidate = siblings[target_idx]
            if same_type:
                if G.nodes[candidate].get("node_type") == node_type:
                    return candidate
            else:
                return candidate
    except ValueError:
        pass

    return None


def resolve_anaphore(
    G: nx.DiGraph,
    seq_index: Dict[str, int],
    article_index: Dict[str, List[str]],
    context_node_id: str,
    phrase: str,
) -> Optional[str]:
    """
    Résout une anaphore relative dans `phrase` en utilisant le contexte
    du noeud `context_node_id`.
    """
    phrase_lower = phrase.lower()

    # "le présent article / chapitre / titre"
    for struct_type in ["article", "chapitre", "titre", "section"]:
        if f"présent {struct_type}" in phrase_lower or f"present {struct_type}" in phrase_lower:
            ancestor = find_ancestor_of_type(G, context_node_id, struct_type)
            if ancestor:
                return ancestor

    # "l'article précédent / suivant"
    if "précédent" in phrase_lower or "precedent" in phrase_lower:
        # D'abord vérifier si un numéro d'article est mentionné
        m = re.search(r"article\s+(\d+)", phrase_lower)
        if m:
            num = m.group(1)
            candidates = article_index.get(num, [])
            # Filtrer par même document
            src = G.nodes[context_node_id].get("source_file", "")
            for c in candidates:
                if G.nodes[c].get("source_file") == src:
                    return c
        # Sinon, frère précédent
        return find_sibling_by_offset(G, context_node_id, seq_index, -1)

    if "suivant" in phrase_lower:
        return find_sibling_by_offset(G, context_node_id, seq_index, 1)

    # "ci-dessus" / "ci-avant" → article frère précédent ou ancêtre article
    if "ci-dessus" in phrase_lower or "ci-avant" in phrase_lower or "ci dessus" in phrase_lower:
        m = re.search(r"article\s+(\d+)", phrase_lower)
        if m:
            num = m.group(1)
            src = G.nodes[context_node_id].get("source_file", "")
            for c in article_index.get(num, []):
                if G.nodes[c].get("source_file") == src:
                    return c
        return find_ancestor_of_type(G, context_node_id, "article")

    # "ci-après" / "ci-après" → article frère suivant
    if "ci-après" in phrase_lower or "ci après" in phrase_lower or "ci-apres" in phrase_lower:
        m = re.search(r"article\s+(\d+)", phrase_lower)
        if m:
            num = m.group(1)
            src = G.nodes[context_node_id].get("source_file", "")
            for c in article_index.get(num, []):
                if G.nodes[c].get("source_file") == src:
                    return c
        return find_sibling_by_offset(G, context_node_id, seq_index, 1)

    # "susvisé" / "ladite" → chercher le terme mentionné plus haut dans le document
    if "susvisé" in phrase_lower or "susvise" in phrase_lower:
        m = re.search(r"article\s+(\d+)", phrase_lower)
        if m:
            num = m.group(1)
            src = G.nodes[context_node_id].get("source_file", "")
            for c in article_index.get(num, []):
                if G.nodes[c].get("source_file") == src:
                    return c

    return None


# ══════════════════════════════════════════════════════════════════════
# Étage 3 – Recherche externe (ChromaDB)
# ══════════════════════════════════════════════════════════════════════

def resolve_external_reference(
    ref: Dict,
    external_collection=None,
) -> Optional[str]:
    """
    Étage 3 : résout une référence externe via ChromaDB.
    Retourne un identifiant externe ou None.
    """
    if external_collection is None:
        return None

    query_text = ref.get("raw", "")
    ref_type = ref.get("type", "")

    if ref_type == "external_law":
        query_text = f"loi n° {ref['value']}"
    elif ref_type == "external_code":
        query_text = f"code {ref['value']}"

    try:
        results = external_collection.query(
            query_texts=[query_text],
            n_results=3,
        )
        if results and results["distances"] and results["distances"][0]:
            if results["distances"][0][0] < (1 - config.EXTERNAL_SEARCH_THRESHOLD):
                ext_id = results["ids"][0][0]
                return f"EXTERNAL::{ext_id}"
    except Exception as e:
        logger.warning(f"Erreur recherche externe : {e}")

    return None


# ══════════════════════════════════════════════════════════════════════
# Étage 4 – Fallback NER / LLM
# ══════════════════════════════════════════════════════════════════════

def resolve_with_llm(text: str, context_clause_id: str = "") -> List[Dict]:
    """
    Étage 4 : fallback utilisant un LLM (Ollama ou OpenAI) pour
    extraire les références non résolues par les étages précédents.
    """
    prompt = f"""Tu es un expert en analyse de documents juridiques francophones.
Identifie TOUTES les références à d'autres articles, lois, décrets, codes, annexes ou sections dans le texte suivant.
Pour chaque référence, indique son type (article, loi, décret, code, annexe, section, chapitre, titre) et sa valeur exacte.

Texte à analyser :
\"\"\"
{text}
\"\"\"

Réponds au format JSON : une liste d'objets {{"type": "...", "value": "...", "raw": "..."}}.
Si aucune référence n'est trouvée, retourne une liste vide [].
Réponds UNIQUEMENT avec le JSON, sans autre texte."""

    refs = []
    try:
        if config.LLM_PROVIDER == "ollama":
            import requests
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=60,
            )
            result = response.json().get("response", "")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Tu extrais les références juridiques. Réponds uniquement en JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            result = response.choices[0].message.content

        # Parser la réponse JSON
        import json
        # Extraire le JSON de la réponse (peut être enveloppé dans des backticks)
        json_match = re.search(r"\[.*\]", result, re.DOTALL)
        if json_match:
            refs = json.loads(json_match.group())
            for r in refs:
                r["confidence"] = 0.6
                r["source"] = "llm_fallback"
    except Exception as e:
        logger.warning(f"Erreur LLM fallback : {e}")

    return refs


# ══════════════════════════════════════════════════════════════════════
# Résolveur hybride complet
# ══════════════════════════════════════════════════════════════════════

class ReferenceResolver:
    """
    Résolveur hybride 4 étages pour les références juridiques.
    Inclut cache LLM et limitation du fallback aux expressions déclencheuses.
    """

    def __init__(
        self,
        G_lex: nx.DiGraph,
        article_index: Dict[str, List[str]],
        seq_index: Dict[str, int],
        external_collection=None,
        use_llm_fallback: bool = True,
    ):
        self.G_lex = G_lex
        self.article_index = article_index
        self.seq_index = seq_index
        self.external_collection = external_collection
        self.use_llm_fallback = use_llm_fallback
        self.resolution_log: List[Dict] = []
        # Cache pour les résultats du LLM : {text_hash → List[Dict]}
        self._llm_cache: Dict[str, List[Dict]] = {}

    def resolve_clause(self, node_id: str) -> Dict:
        """
        Résout toutes les références d'une clause donnée.
        Retourne {node_id, internal_refs, external_refs, unresolved, log}.
        """
        data = self.G_lex.nodes.get(node_id, {})
        text = data.get("full_text", "")
        source_file = data.get("source_file", "")

        if not text:
            return {"node_id": node_id, "internal_refs": [], "external_refs": [], "unresolved": []}

        internal_refs = []
        external_refs = []
        unresolved = []

        # ── Étage 1 : Regex ──
        regex_refs = extract_references_regex(text)
        logger.debug(f"[Étage1] {node_id}: {len(regex_refs)} refs regex")

        for ref in regex_refs:
            resolved = self._resolve_single_ref(ref, node_id, source_file)
            if resolved:
                if resolved.get("type") == "external":
                    external_refs.append(resolved)
                else:
                    internal_refs.append(resolved)
                self.resolution_log.append({
                    "node_id": node_id,
                    "ref": ref,
                    "resolved_to": resolved,
                    "stage": "regex",
                })
            else:
                unresolved.append(ref)

        # ── Étage 2 : Anaphores (heuristiques graphe) ──
        # Chercher des mots-clés anaphoriques dans le texte
        text_lower = text.lower()
        anaphore_found = False
        for keyword in ANAPHORE_KEYWORDS:
            if keyword in text_lower:
                resolved_id = resolve_anaphore(
                    self.G_lex, self.seq_index, self.article_index,
                    node_id, text,
                )
                if resolved_id:
                    internal_refs.append({
                        "type": "anaphore",
                        "target_node_id": resolved_id,
                        "keyword": keyword,
                        "confidence": 0.7,
                    })
                    anaphore_found = True
                    self.resolution_log.append({
                        "node_id": node_id,
                        "ref": {"type": "anaphore", "keyword": keyword},
                        "resolved_to": resolved_id,
                        "stage": "heuristic",
                    })
                break  # Une seule anaphore par clause

        # ── Étage 3 : Références externes ──
        for ref in regex_refs:
            if ref.get("type") in ("external_law", "external_code"):
                ext_id = resolve_external_reference(ref, self.external_collection)
                if ext_id:
                    external_refs.append({
                        "type": "external",
                        "ext_id": ext_id,
                        "raw": ref["raw"],
                        "confidence": 0.7,
                    })
                    self.resolution_log.append({
                        "node_id": node_id,
                        "ref": ref,
                        "resolved_to": ext_id,
                        "stage": "external",
                    })

        # ── Étage 4 : Fallback LLM (seulement si des refs non résolues ET expressions déclencheuses) ──
        if self.use_llm_fallback and unresolved:
            # Limiter le fallback aux clauses contenant des expressions déclencheuses
            should_trigger = any(
                expr in text_lower for expr in LLM_TRIGGER_EXPRESSIONS
            )
            if should_trigger:
                # Vérifier le cache
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self._llm_cache:
                    llm_refs = self._llm_cache[text_hash]
                    logger.debug(f"[Étage4] Cache hit pour {node_id}")
                else:
                    llm_refs = resolve_with_llm(text, data.get("clause_id", ""))
                    self._llm_cache[text_hash] = llm_refs
                    logger.debug(f"[Étage4] LLM appelé pour {node_id}")
                for ref in llm_refs:
                    resolved = self._resolve_single_ref(ref, node_id, source_file)
                    if resolved:
                        if resolved.get("type") == "external":
                            external_refs.append(resolved)
                        else:
                            internal_refs.append(resolved)
                        self.resolution_log.append({
                            "node_id": node_id,
                            "ref": ref,
                            "resolved_to": resolved,
                            "stage": "llm_fallback",
                        })
            else:
                logger.debug(f"[Étage4] Skippé pour {node_id} (pas d'expression déclencheuse)")

        return {
            "node_id": node_id,
            "internal_refs": internal_refs,
            "external_refs": external_refs,
            "unresolved": unresolved,
        }

    def _resolve_single_ref(
        self, ref: Dict, context_node_id: str, source_file: str
    ) -> Optional[Dict]:
        """Tente de résoudre une référence unique en un node_id."""
        ref_type = ref.get("type", "")
        value = ref.get("value")

        if ref_type == "article" and isinstance(value, int):
            key = str(value)
            candidates = self.article_index.get(key, [])
            # Filtrer par même document source
            for c in candidates:
                if self.G_lex.nodes[c].get("source_file") == source_file:
                    return {
                        "type": "internal",
                        "target_node_id": c,
                        "article_num": value,
                        "confidence": ref.get("confidence", 0.9),
                    }
            # Si pas dans le même document, prendre le premier
            if candidates:
                return {
                    "type": "internal",
                    "target_node_id": candidates[0],
                    "article_num": value,
                    "confidence": ref.get("confidence", 0.7),
                }

        elif ref_type == "article_range" and isinstance(value, tuple):
            start, end = value
            resolved_range = []
            for num in range(start, end + 1):
                key = str(num)
                candidates = self.article_index.get(key, [])
                for c in candidates:
                    if self.G_lex.nodes[c].get("source_file") == source_file:
                        resolved_range.append(c)
                        break
            if resolved_range:
                return {
                    "type": "internal_range",
                    "target_node_ids": resolved_range,
                    "article_range": (start, end),
                    "confidence": ref.get("confidence", 0.9),
                }

        elif ref_type in ("external_law", "external_code"):
            # Sera géré par l'Étage 3
            return None

        elif ref_type == "footnote":
            # Chercher les footnotes du même document
            # Méthode 1 : via is_footnote=True + numéro dans clause_id
            for nid, data in self.G_lex.nodes(data=True):
                if (
                    data.get("is_footnote")
                    and data.get("source_file") == source_file
                ):
                    fn_match = re.search(r"(\d+)", data.get("clause_id", ""))
                    if fn_match and int(fn_match.group(1)) == value:
                        return {
                            "type": "internal",
                            "target_node_id": nid,
                            "footnote_num": value,
                            "confidence": 0.85,
                        }
            # Méthode 2 : via footnote_text non-null du même document
            for nid, data in self.G_lex.nodes(data=True):
                if (
                    data.get("footnote_text")
                    and data.get("source_file") == source_file
                    and not data.get("is_footnote")
                ):
                    # La clause elle-même contient une footnote_text
                    fn_match = re.search(r"note\s+(\d+)|(\d+)", data.get("clause_id", ""))
                    if fn_match:
                        fn_num = fn_match.group(1) or fn_match.group(2)
                        if int(fn_num) == value:
                            return {
                                "type": "internal",
                                "target_node_id": nid,
                                "footnote_num": value,
                                "confidence": 0.8,
                            }
            # Méthode 3 : chercher dans le texte de la clause source si une footnote est inline
            source_data = self.G_lex.nodes.get(context_node_id, {})
            if source_data.get("footnote_text"):
                return {
                    "type": "internal",
                    "target_node_id": context_node_id,
                    "footnote_num": value,
                    "confidence": 0.75,
                    "note": "footnote_text inline dans la clause source",
                }

        elif ref_type in ("titre", "chapitre", "section", "annexe"):
            # Chercher par type et identifiant dans le même document
            for nid, data in self.G_lex.nodes(data=True):
                if data.get("source_file") == source_file and data.get("node_type") == ref_type:
                    cid_lower = data.get("clause_id", "").lower()
                    val_lower = str(value).lower()
                    if val_lower in cid_lower:
                        return {
                            "type": "internal",
                            "target_node_id": nid,
                            "confidence": ref.get("confidence", 0.8),
                        }

        return None

    def resolve_all(self) -> Dict[str, Dict]:
        """
        Résout les références de toutes les clauses du graphe.
        Retourne {node_id: resolution_dict}.
        """
        results = {}
        total = self.G_lex.number_of_nodes()
        for i, node_id in enumerate(self.G_lex.nodes):
            data = self.G_lex.nodes[node_id]
            if data.get("node_type") == "placeholder":
                continue
            result = self.resolve_clause(node_id)
            results[node_id] = result
            if (i + 1) % 100 == 0:
                logger.info(f"Résolution : {i+1}/{total} clauses traitées")

        total_internal = sum(len(r["internal_refs"]) for r in results.values())
        total_external = sum(len(r["external_refs"]) for r in results.values())
        total_unresolved = sum(len(r["unresolved"]) for r in results.values())
        logger.info(
            f"Résolution terminée : {total_internal} internes, "
            f"{total_external} externes, {total_unresolved} non résolues"
        )
        return results
