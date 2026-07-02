from typing import List, Dict, Any, Optional, Annotated, Union
from pydantic import BaseModel, Field
from datetime import datetime
import operator

class DocumentNode(BaseModel):
    """Represents a node in the document hierarchy graph"""
    node_id: str
    content: str
    node_type: str
    page_number: int
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    links_to: List[str] = Field(default_factory=list)
    footnotes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def print_node_prompt(self) -> str:
        return f"[{self.node_id}] {self.node_type}: {self.content[:200]}...\n"

class DefinitionNode(BaseModel):
    """Represents a legal definition"""
    term: str
    definition: str
    source_page: int
    context: Optional[str] = None
    embedding: Optional[List[float]] = None

# Utilisation de TypedDict pour LangGraph pour une meilleure gestion des mises à jour concurrentes
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """State passed between agents in the workflow"""
    query: str
    # previous_nodes s'accumulent pour garder l'historique
    previous_nodes: Annotated[List[DocumentNode], operator.add]
    # Les autres champs sont remplacés à chaque étape
    last_fetched_context_nodes: List[DocumentNode]
    node_links_to_fetch: List[str]
    node_footers_to_fetch: Dict[str, str]
    # search_failures s'accumulent pour le debug
    search_failures: Annotated[List[str], operator.add]
    pass_count: int
    definitions: Dict[str, str]
    final_answer: Optional[str]
    supervisor_decision: Optional[str]

def get_state_context(state: AgentState) -> str:
    """Helper to get context from TypedDict state"""
    context = "Relevant Document Sections:\n"
    nodes = state.get("previous_nodes", []) + state.get("last_fetched_context_nodes", [])
    for node in nodes:
        context += node.print_node_prompt()
    
    definitions = state.get("definitions", {})
    if definitions:
        context += "\nRelevant Definitions:\n"
        for term, definition in definitions.items():
            context += f"{term}: {definition}\n"
    
    return context

class GraphTriple(BaseModel):
    """Represents a triple in the knowledge graph"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultiAgentSearchLocalNode(BaseModel):
    """Local node representation for multi-agent search"""
    node_id: str
    content: str
    node_type: str
    page_number: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def print_node_prompt(self) -> str:
        return f"[{self.node_id}] {self.node_type}: {self.content[:200]}...\n"

class DocumentStructure(BaseModel):
    """Represents the parsed structure of a legal document"""
    document_id: str
    title: str
    pages: List[Dict[str, Any]]
    hierarchy: List[DocumentNode]
    definitions: List[DefinitionNode]
    created_at: datetime = Field(default_factory=datetime.now)
