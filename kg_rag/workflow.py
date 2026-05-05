"""
Phase 4 – Workflow LangGraph : orchestration des agents en graphe d'état.
"""
import logging
from typing import Dict, Optional

from langgraph.graph import StateGraph, END

from .agents import (
    AgentState,
    initial_search_agent,
    definition_agent,
    router_agent,
    recursive_retrieval_agent,
    external_law_agent,
    supervisor_agent,
    answering_agent,
)
from . import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Construction du workflow
# ══════════════════════════════════════════════════════════════════════

class KGRAGWorkflow:
    """
    Workflow LangGraph pour le retrieval multi-agents KG-RAG.
    """

    def __init__(
        self,
        hybrid_retriever=None,
        definition_store=None,
        chroma_store=None,
        G_ref=None,
        G_lex=None,
        article_index: Dict = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.definition_store = definition_store
        self.chroma_store = chroma_store
        self.G_ref = G_ref
        self.G_lex = G_lex
        self.article_index = article_index or {}
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construit le graphe LangGraph."""

        # Créer le graphe avec l'état
        workflow = StateGraph(AgentState)

        # Ajouter les noeuds (agents)
        workflow.add_node("initial_search", self._initial_search)
        workflow.add_node("definition", self._definition)
        workflow.add_node("router", self._router)
        workflow.add_node("recursive_retrieval", self._recursive_retrieval)
        workflow.add_node("external_law", self._external_law)
        workflow.add_node("supervisor", self._supervisor)
        workflow.add_node("answering", self._answering)

        # Définir le point d'entrée
        workflow.set_entry_point("initial_search")

        # Définir les arêtes
        workflow.add_edge("initial_search", "definition")
        workflow.add_edge("definition", "router")

        # Router → conditionnel
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "RECURSE": "recursive_retrieval",
                "EXTERNAL": "external_law",
                "STOP": "answering",
            },
        )

        # Après recursive retrieval → supervisor
        workflow.add_edge("recursive_retrieval", "supervisor")

        # Après external law → supervisor
        workflow.add_edge("external_law", "supervisor")

        # Supervisor → conditionnel (continuer ou arrêter)
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_decision,
            {
                "CONTINUE": "router",
                "STOP": "answering",
            },
        )

        # Answering → END
        workflow.add_edge("answering", END)

        # Compiler
        return workflow.compile()

    # ── Wrappers pour injecter les dépendances ────────────────────────

    def _initial_search(self, state: AgentState) -> AgentState:
        return initial_search_agent(state, hybrid_retriever=self.hybrid_retriever)

    def _definition(self, state: AgentState) -> AgentState:
        return definition_agent(state, definition_store=self.definition_store)

    def _router(self, state: AgentState) -> AgentState:
        return router_agent(state)

    def _recursive_retrieval(self, state: AgentState) -> AgentState:
        return recursive_retrieval_agent(
            state,
            G_ref=self.G_ref,
            G_lex=self.G_lex,
            article_index=self.article_index,
        )

    def _external_law(self, state: AgentState) -> AgentState:
        return external_law_agent(state, chroma_store=self.chroma_store)

    def _supervisor(self, state: AgentState) -> AgentState:
        return supervisor_agent(state)

    def _answering(self, state: AgentState) -> AgentState:
        return answering_agent(state)

    # ── Fonctions de routage conditionnel ──────────────────────────────

    def _route_decision(self, state: AgentState) -> str:
        decision = state.get("router_decision", "STOP")
        logger.info(f"[Workflow] Route : {decision}")
        return decision

    def _supervisor_decision(self, state: AgentState) -> str:
        decision = state.get("router_decision", "STOP")
        # Le supervisor peut forcer l'arrêt
        if state.get("pass_count", 0) >= config.MAX_AGENT_PASSES:
            return "STOP"
        # Si le router a dit STOP, on arrête
        if decision == "STOP":
            return "STOP"
        return "CONTINUE"

    # ── Exécution ─────────────────────────────────────────────────────

    def run(self, query: str) -> Dict:
        """
        Exécute le workflow complet pour une requête.
        """
        initial_state = {
            "query": query,
            "initial_docs": [],
            "definitions": {},
            "retrieved_graph_nodes": [],
            "retrieved_texts": [],
            "external_context": [],
            "pass_count": 0,
            "failures": [],
            "final_answer": "",
            "router_decision": "RECURSE",
            "node_links_to_fetch": [],
            "node_footers_to_fetch": {},
        }

        logger.info(f"═══ Workflow démarré pour : {query[:80]} ═══")

        result = self.graph.invoke(initial_state)

        logger.info(
            f"═══ Workflow terminé : "
            f"{result.get('pass_count', 0)} passes, "
            f"{len(result.get('retrieved_graph_nodes', []))} noeuds, "
            f"{len(result.get('failures', []))} échecs ═══"
        )

        return result
