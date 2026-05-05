"""
Phase 4 – Agents LangGraph : Initial Search, Definition, Router,
Recursive Retrieval, External Law, Supervisor, Answering.
"""
import logging
import re
from typing import Dict, List, Optional, TypedDict

from . import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# État partagé des agents
# ══════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    query: str
    initial_docs: List[Dict]
    definitions: Dict[str, str]
    retrieved_graph_nodes: List[str]
    retrieved_texts: List[str]
    external_context: List[str]
    pass_count: int
    failures: List[str]
    final_answer: str
    router_decision: str  # "STOP", "RECURSE", "EXTERNAL"
    node_links_to_fetch: List[str]
    node_footers_to_fetch: Dict[str, str]


# ══════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Appelle le LLM configuré (Ollama ou OpenAI)."""
    if config.LLM_PROVIDER == "ollama":
        import requests
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        return response.json().get("response", "")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════
# Agent 1 – Initial Search
# ══════════════════════════════════════════════════════════════════════

def initial_search_agent(state: AgentState, hybrid_retriever=None) -> AgentState:
    """
    Recherche initiale via le retriever hybride (Vector + BM25).
    Récupère les clauses les plus pertinentes pour la requête.
    """
    query = state.get("query", "")

    if hybrid_retriever is not None:
        docs = hybrid_retriever.query(query, top_k=config.TOP_K_INITIAL)
    else:
        docs = []

    state["initial_docs"] = docs
    state["retrieved_graph_nodes"] = []
    state["retrieved_texts"] = []
    state["definitions"] = {}
    state["external_context"] = []
    state["pass_count"] = 0
    state["failures"] = []
    state["node_links_to_fetch"] = []
    state["node_footers_to_fetch"] = {}

    # Ajouter les node_ids des documents initiaux
    for doc in docs:
        node_id = doc.get("metadata", {}).get("node_id", "")
        if node_id:
            state["retrieved_graph_nodes"].append(node_id)
        state["retrieved_texts"].append(doc.get("text", ""))

    logger.info(f"[InitialSearch] {len(docs)} documents trouvés pour : {query[:80]}")
    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 2 – Definition
# ══════════════════════════════════════════════════════════════════════

def definition_agent(state: AgentState, definition_store=None) -> AgentState:
    """
    Cherche les définitions des termes de la requête
    dans la collection ChromaDB des définitions.
    """
    query = state.get("query", "")
    definitions = state.get("definitions", {})

    if definition_store is not None:
        try:
            results = definition_store.query(
                query, collection_name=config.CHROMA_COLLECTION_DEFINITIONS,
                n_results=5,
            )
            for r in results:
                term = r.get("metadata", {}).get("term", "")
                text = r.get("text", "")
                if term and text:
                    definitions[term] = text
        except Exception as e:
            logger.warning(f"[Definition] Erreur : {e}")

    state["definitions"] = definitions
    logger.info(f"[Definition] {len(definitions)} définitions trouvées")
    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 3 – Router
# ══════════════════════════════════════════════════════════════════════

def router_agent(state: AgentState, use_llm: bool = False) -> AgentState:
    """
    Décide si le processus doit s'arrêter ou continuer.
    Mode par défaut : heuristique regex (rapide, déterministe).
    Mode LLM : si use_llm=True ou si l'heuristique ne trouve rien,
    utilise le LLM pour une analyse plus fine.
    """
    query = state.get("query", "")
    docs = state.get("initial_docs", [])
    retrieved_texts = state.get("retrieved_texts", [])

    # ── Mode heuristique (par défaut, rapide) ──
    from .reference_resolver import extract_references_regex
    all_refs = []
    for text in retrieved_texts:
        refs = extract_references_regex(text)
        all_refs.extend(refs)

    # Séparer les refs internes et externes
    internal_refs = [r for r in all_refs if r["type"] not in ("external_law", "external_code")]
    external_refs = [r for r in all_refs if r["type"] in ("external_law", "external_code")]

    if internal_refs:
        state["router_decision"] = "RECURSE"
        article_nums = []
        for r in internal_refs:
            if r["type"] == "article" and isinstance(r.get("value"), int):
                article_nums.append(str(r["value"]))
            elif r["type"] == "article_range" and isinstance(r.get("value"), tuple):
                start, end = r["value"]
                article_nums.extend(str(n) for n in range(start, end + 1))
        state["node_links_to_fetch"] = article_nums
        state["node_footers_to_fetch"] = {}
        logger.info(f"[Router] Heuristique : RECURSE, {len(article_nums)} articles à récupérer")
        return state
    elif external_refs:
        state["router_decision"] = "EXTERNAL"
        state["node_links_to_fetch"] = [r.get("raw", "") for r in external_refs]
        logger.info(f"[Router] Heuristique : EXTERNAL, {len(external_refs)} réf. externes")
        return state

    # ── Mode LLM (si heuristique n'a rien trouvé ET use_llm=True) ──
    if use_llm:
        context = "\n---\n".join(retrieved_texts[-5:])
        system_prompt = """Tu es un agent intelligent supervisant un processus de récupération multi-agent dans un graphe de connaissances juridiques.

Tu dois déterminer si les documents récupérés suffisent pour répondre à la requête, ou s'il faut récupérer des informations supplémentaires.

Analyse les documents ci-dessous et décide :
- STOP : les documents suffisent pour répondre à la requête
- RECURSE : des références internes (articles, paragraphes, alinéas, notes de bas de page) doivent être récupérées. Liste les clause_id ou node_id à récupérer.
- EXTERNAL : des références à des lois ou codes externes doivent être recherchées. Liste les références externes.

Réponds au format JSON :
{"decision": "STOP"|"RECURSE"|"EXTERNAL", "node_ids": [...], "footer_queries": {...}, "external_refs": [...]}"""

        user_prompt = f"""Requête : {query}

Documents récupérés :
{context}

Définitions disponibles : {list(state.get('definitions', {}).keys())}

Analyse les documents et prends une décision. Réponds UNIQUEMENT en JSON."""

        try:
            response = call_llm(system_prompt, user_prompt, temperature=0.1)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                decision = json.loads(json_match.group())
                state["router_decision"] = decision.get("decision", "STOP")
                state["node_links_to_fetch"] = decision.get("node_ids", [])
                state["node_footers_to_fetch"] = decision.get("footer_queries", {})
            else:
                state["router_decision"] = "STOP"
        except Exception as e:
            logger.warning(f"[Router] Erreur LLM : {e}")
            state["router_decision"] = "STOP"
    else:
        state["router_decision"] = "STOP"

    logger.info(f"[Router] Décision : {state.get('router_decision', 'STOP')}")
    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 4 – Recursive Retrieval
# ══════════════════════════════════════════════════════════════════════

def recursive_retrieval_agent(
    state: AgentState,
    G_ref=None,
    G_lex=None,
    article_index: Dict = None,
) -> AgentState:
    """
    Parcourt le graphe de références pour récupérer les clauses liées.
    Évite les cycles (noeuds déjà visités).
    """
    node_links = state.get("node_links_to_fetch", [])
    footer_queries = state.get("node_footers_to_fetch", {})
    visited = set(state.get("retrieved_graph_nodes", []))
    new_texts = []
    new_nodes = []

    # Résoudre les node_ids / numéros d'articles
    if G_ref is not None and article_index is not None:
        for link in node_links:
            # Nettoyer les backticks éventuels
            link = link.strip("`")

            # Si c'est un node_id direct
            if link in G_ref.nodes:
                if link not in visited:
                    new_nodes.append(link)
                    text = G_ref.nodes[link].get("full_text", "")
                    if text:
                        new_texts.append(f"### {G_ref.nodes[link].get('clause_id', link)}\n{text}")
            # Si c'est un numéro d'article
            elif link.isdigit() and article_index:
                candidates = article_index.get(link, [])
                for c in candidates:
                    if c not in visited and c in G_ref.nodes:
                        new_nodes.append(c)
                        text = G_ref.nodes[c].get("full_text", "")
                        if text:
                            new_texts.append(f"### {G_ref.nodes[c].get('clause_id', c)}\n{text}")

    # Recherche par footer/keyword queries
    if footer_queries and G_lex is not None:
        for node_id, search_query in footer_queries.items():
            # Recherche textuelle basique dans les noeuds du graphe
            found = False
            query_lower = search_query.lower()
            for nid, data in G_lex.nodes(data=True):
                if data.get("node_type") == "placeholder":
                    continue
                if query_lower in data.get("full_text", "").lower() and nid not in visited:
                    new_nodes.append(nid)
                    new_texts.append(f"### {data.get('clause_id', nid)}\n{data.get('full_text', '')}")
                    found = True
                    break
            if not found:
                state["failures"].append(f"Footer non trouvé : {search_query}")

    # Mettre à jour l'état
    state["retrieved_graph_nodes"].extend(new_nodes)
    state["retrieved_texts"].extend(new_texts)
    state["pass_count"] = state.get("pass_count", 0) + 1
    state["node_links_to_fetch"] = []
    state["node_footers_to_fetch"] = {}

    if not new_nodes:
        state["failures"].append(f"Pass {state['pass_count']}: aucun nouveau noeud trouvé")

    logger.info(f"[RecursiveRetrieval] {len(new_nodes)} nouveaux noeuds, pass {state['pass_count']}")
    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 5 – External Law
# ══════════════════════════════════════════════════════════════════════

def external_law_agent(state: AgentState, chroma_store=None) -> AgentState:
    """
    Recherche les références externes dans la collection external_laws.
    Désactivé proprement si la collection est vide ou indisponible.
    """
    external_refs = state.get("node_links_to_fetch", [])
    external_context = state.get("external_context", [])

    if not external_refs:
        logger.info("[ExternalLaw] Aucune réf. externe à chercher")
        state["external_context"] = external_context
        state["node_links_to_fetch"] = []
        return state

    if chroma_store is None:
        logger.warning("[ExternalLaw] ChromaStore non disponible, réf. externes ignorées")
        for ref in external_refs:
            state["failures"].append(f"Index externe indisponible pour : {ref}")
        state["external_context"] = external_context
        state["node_links_to_fetch"] = []
        return state

    # Vérifier que la collection externe existe et n'est pas vide
    try:
        collection = chroma_store.get_or_create_collection(config.CHROMA_COLLECTION_EXTERNAL)
        if collection.count() == 0:
            logger.warning("[ExternalLaw] Collection externe vide, réf. externes ignorées")
            for ref in external_refs:
                state["failures"].append(f"Index externe vide pour : {ref}")
            state["external_context"] = external_context
            state["node_links_to_fetch"] = []
            return state
    except Exception as e:
        logger.warning(f"[ExternalLaw] Erreur accès collection : {e}")
        state["external_context"] = external_context
        state["node_links_to_fetch"] = []
        return state

    for ref in external_refs:
        try:
            results = chroma_store.query(
                ref, collection_name=config.CHROMA_COLLECTION_EXTERNAL,
                n_results=3,
            )
            for r in results:
                external_context.append(
                    f"[Réf. externe : {r.get('metadata', {}).get('clause_id', r.get('id', ''))}] "
                    f"{r.get('text', '')}"
                )
        except Exception as e:
            state["failures"].append(f"Recherche externe échouée pour '{ref}': {e}")

    state["external_context"] = external_context
    state["node_links_to_fetch"] = []
    logger.info(f"[ExternalLaw] {len(external_context)} contextes externes")
    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 6 – Supervisor
# ══════════════════════════════════════════════════════════════════════

def supervisor_agent(state: AgentState) -> AgentState:
    """
    Surveille le processus : arrêt après N passes,
    gestion des échecs, élagage du contexte si trop long.
    """
    pass_count = state.get("pass_count", 0)
    failures = state.get("failures", [])

    # Condition d'arrêt : trop de passes
    if pass_count >= config.MAX_AGENT_PASSES:
        logger.info(f"[Supervisor] Arrêt : max passes atteint ({pass_count})")
        state["router_decision"] = "STOP"
        return state

    # Condition d'arrêt : trop d'échecs récents
    recent_failures = failures[-3:] if len(failures) >= 3 else failures
    if len(recent_failures) >= 3 and all("aucun nouveau noeud" in f for f in recent_failures):
        logger.info("[Supervisor] Arrêt : échecs répétés")
        state["router_decision"] = "STOP"
        return state

    # Élagage du contexte si trop long
    retrieved_texts = state.get("retrieved_texts", [])
    total_chars = sum(len(t) for t in retrieved_texts)
    max_chars = config.MAX_CONTEXT_TOKENS * 4  # ~4 chars/token

    if total_chars > max_chars:
        # Garder les textes les plus récents (plus pertinents)
        kept = []
        running = 0
        for t in reversed(retrieved_texts):
            if running + len(t) > max_chars:
                break
            kept.insert(0, t)
            running += len(t)
        state["retrieved_texts"] = kept
        logger.info(f"[Supervisor] Élagage : {len(kept)}/{len(retrieved_texts)} textes conservés")

    return state


# ══════════════════════════════════════════════════════════════════════
# Agent 7 – Answering
# ══════════════════════════════════════════════════════════════════════

def answering_agent(state: AgentState) -> AgentState:
    """
    Synthétise toutes les informations collectées en une réponse finale
    avec citations (clause_id, titre, extrait).
    """
    query = state.get("query", "")
    retrieved_texts = state.get("retrieved_texts", [])
    definitions = state.get("definitions", {})
    external_context = state.get("external_context", [])

    # Construire le contexte
    context_parts = []

    if definitions:
        defs_text = "\n".join(f"- **{t}** : {d}" for t, d in definitions.items())
        context_parts.append(f"## Définitions pertinentes\n{defs_text}")

    if retrieved_texts:
        docs_text = "\n\n".join(retrieved_texts)
        context_parts.append(f"## Clauses récupérées\n{docs_text}")

    if external_context:
        ext_text = "\n\n".join(external_context)
        context_parts.append(f"## Références externes\n{ext_text}")

    full_context = "\n\n".join(context_parts)

    system_prompt = """Tu es un agent de synthèse juridique. Tu dois répondre à la requête de l'utilisateur en te basant EXCLUSIVEMENT sur les documents, définitions et références fournies.

Instructions :
1. Réponds de manière structurée et complète
2. Cite systématiquement les articles/clauses sources (ex: Art. 16, Article 7.2, etc.)
3. Si des définitions sont fournies dans la section Définitions pertinentes, utilise-les pour contextualiser ta réponse
4. Si des références externes sont mentionnées dans la section Références externes, intègre-les dans ta réponse et indique clairement leur origine
5. Ne pas inventer d'informations non présentes dans le contexte
6. Réponds en français
7. Si le contexte est insuffisant pour répondre complètement, indique-le explicitement"""

    user_prompt = f"""Requête : {query}

{full_context}

Rédige une réponse complète et structurée avec citations des sources."""

    try:
        answer = call_llm(system_prompt, user_prompt, temperature=0.3)
        state["final_answer"] = answer
    except Exception as e:
        logger.error(f"[Answering] Erreur LLM : {e}")
        # Réponse de fallback : concaténer les textes récupérés
        state["final_answer"] = (
            f"Réponse (mode fallback) :\n\n"
            f"{'---'.join(retrieved_texts[:5])}"
        )

    logger.info(f"[Answering] Réponse générée ({len(state.get('final_answer', ''))} chars)")
    return state
