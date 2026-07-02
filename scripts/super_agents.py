import logging
import json
import re
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from groq import Groq
import scripts.retrieval_fusion as rf

logger = logging.getLogger(__name__)

load_dotenv("legal_rag/.env")

# --- Modèles d'état adaptables au Super-Index ---
class DocumentNode(BaseModel):
    node_id: str
    content: str
    node_type: str = "article"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def print_node_prompt(self) -> str:
        return f"[{self.node_id}] : {self.content}\n"

class AgentState(BaseModel):
    query: str = ""
    country: Optional[str] = None
    previous_nodes: List[DocumentNode] = Field(default_factory=list)
    last_fetched_context_nodes: List[DocumentNode] = Field(default_factory=list)
    node_links_to_fetch: List[str] = Field(default_factory=list)
    node_footers_to_fetch: Dict[str, str] = Field(default_factory=dict)
    search_failures: List[str] = Field(default_factory=list)
    pass_count: int = 0
    definitions: Dict[str, str] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    supervisor_decision: Optional[str] = None
    
    def get_all_context(self) -> str:
        context = "Relevant Document Sections:\n"
        for node in self.previous_nodes + self.last_fetched_context_nodes:
            context += node.print_node_prompt()
        if self.definitions:
            context += "\nRelevant Definitions:\n"
            for term, definition in self.definitions.items():
                context += f"{term}: {definition}\n"
        return context

# --- Le Système Multi-Agents ---
class SuperLegalAgents:
    """Système Multi-Agents adapté au Super-Index (FAISS/BM25)"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.max_recursion = 2
        
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
        
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("initial_search", self.initial_search_agent)
        workflow.add_node("definitions_search", self.definitions_agent)
        workflow.add_node("router", self.router_agent)
        workflow.add_node("recursive_retrieval", self.recursive_retrieval_agent)
        workflow.add_node("supervisor", self.supervisor_agent)
        workflow.add_node("answering", self.answering_agent)
        
        workflow.set_entry_point("initial_search")
        workflow.add_edge("initial_search", "definitions_search")
        workflow.add_edge("definitions_search", "router")
        workflow.add_edge("recursive_retrieval", "router")
        
        workflow.add_conditional_edges(
            "router",
            self.should_continue_retrieval,
            {"continue": "recursive_retrieval", "end": "supervisor"}
        )
        workflow.add_conditional_edges(
            "supervisor",
            self.should_end_process,
            {"continue": "recursive_retrieval", "end": "answering"}
        )
        workflow.add_edge("answering", END)
        return workflow
    
    def _dict_to_docnode(self, node_dict: dict) -> DocumentNode:
        return DocumentNode(
            node_id=node_dict.get("node_id", "Unknown"),
            content=node_dict.get("text", ""),
            metadata=node_dict.get("metadata", {})
        )

    def initial_search_agent(self, state: AgentState) -> AgentState:
        """Agent de recherche initiale via Super-Index"""
        try:
            hits, trace = rf.retrieve(state.query, top_k_final=5, jurisdiction_filter=state.country)
            state.last_fetched_context_nodes = [self._dict_to_docnode(h) for h in hits]
        except Exception as e:
            state.search_failures.append(f"Initial search failed: {e}")
        return state

    def definitions_agent(self, state: AgentState) -> AgentState:
        """Agent des définitions : interroge le contexte s'il y a des termes techniques"""
        # Simplifié pour le Super-Index : on extrait les termes clés et on cherche s'il y a des définitions
        try:
            prompt = f"Extrait uniquement les concepts juridiques clés de cette requête qui nécessitent une définition stricte : '{state.query}'. Renvoie-les sous forme de liste JSON."
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            # Dans la vraie implémentation, on interrogerait le graphe de définitions ici.
            # Pour l'instant on laisse le dictionnaire vide car le RAG Fusion a déjà l'expansion lexicale.
            state.definitions = {} 
        except Exception as e:
            state.search_failures.append(f"Definitions search failed: {e}")
        return state

    def router_agent(self, state: AgentState) -> AgentState:
        """Agent Routeur pour vérifier s'il manque de l'information (liens ou renvois)"""
        state.node_links_to_fetch = []
        state.node_footers_to_fetch = {}
        context = state.get_all_context()
        
        link_prompt = f"""Tu es un agent routeur. Voici le contexte récupéré pour répondre à : {state.query}
        {context}
        Détermine si tu dois récupérer d'autres articles de loi cités dans ce contexte (ex: "conformément à l'article X").
        Réponds uniquement par une liste de références à chercher (ex: `Article X`), ou `None` si l'info est suffisante."""
        
        try:
            res = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": link_prompt}]
            )
            text = res.choices[0].message.content.strip()
            if text.lower() != "none":
                matches = re.findall(r'`([^`]+)`', text)
                state.node_links_to_fetch.extend(matches)
        except Exception as e:
            state.search_failures.append(f"Router failed: {e}")
            
        return state

    def recursive_retrieval_agent(self, state: AgentState) -> AgentState:
        """Recherche récursive pour récupérer les articles manquants pointés par le Routeur"""
        for n in state.last_fetched_context_nodes:
            state.previous_nodes.append(n)
        state.last_fetched_context_nodes = []
        
        # On relance rf.retrieve sur les liens demandés
        for link_query in state.node_links_to_fetch:
            hits, _ = rf.retrieve(link_query, top_k_final=2, jurisdiction_filter=state.country)
            state.last_fetched_context_nodes.extend([self._dict_to_docnode(h) for h in hits])
            
        state.node_links_to_fetch = []
        state.pass_count += 1
        return state

    def supervisor_agent(self, state: AgentState) -> AgentState:
        """Agent superviseur : décide de stopper ou continuer"""
        context = state.get_all_context()
        prompt = f"""Tu es superviseur. Question : {state.query}. 
        Voici le contexte : {context}. 
        Si on a assez d'infos, réponds END. Sinon CONTINUE."""
        
        try:
            res = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            state.supervisor_decision = "END" if "END" in res.choices[0].message.content.upper() else "CONTINUE"
        except:
            state.supervisor_decision = "END"
        return state

    def answering_agent(self, state: AgentState) -> AgentState:
        """Agent rédacteur final (avec anti-hallucination)"""
        context = state.get_all_context()
        prompt = f"""Tu es un expert juridique. Réponds EXCLUSIVEMENT à partir des sections suivantes. 
        Cite les articles de loi. Si un article parle de substances nocives ou de pollution, applique-le aux hydrocarbures.
        Contexte :
        {context}
        
        Question : {state.query}"""
        
        try:
            res = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            state.final_answer = res.choices[0].message.content.strip()
        except Exception as e:
            state.final_answer = f"Erreur de génération : {e}"
        return state

    def should_continue_retrieval(self, state: AgentState) -> str:
        has_links = len(state.node_links_to_fetch) > 0
        if has_links and state.pass_count < self.max_recursion:
            return "continue"
        return "end"

    def should_end_process(self, state: AgentState) -> str:
        decision = getattr(state, 'supervisor_decision', 'END')
        if decision == 'CONTINUE' and state.pass_count < self.max_recursion:
            return "continue"
        return "end"

    def run_query(self, query: str, country: str = None) -> AgentState:
        initial_state = AgentState(query=query, country=country)
        return self.app.invoke(initial_state)
