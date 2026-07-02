import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import networkx as nx
from .retrieval_system import LegalRetrievalSystem
from .graph_builder import GraphBuilder
from .models import DocumentNode, AgentState
from .config import settings
from groq import Groq

logger = logging.getLogger(__name__)

class NaiveGraphRAG:
    """
    Simplified Graph RAG (Naive version) for comparison.
    1. Performs Hybrid Retrieval (Vector + BM25)
    2. Expands context by fetching immediate neighbors from the graph
    3. Performs one-shot answering (no multi-agent loop)
    """
    
    def __init__(self, retrieval_system: LegalRetrievalSystem, graph_builder: GraphBuilder):
        self.retrieval_system = retrieval_system
        self.graph_builder = graph_builder
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for NaiveGraphRAG. Set it in legal_rag/.env or the environment.")
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Load local graphs for expansion
        self.lexical_graph = self._load_graph("lexical_graph.json")
        self.definitions_graph = self._load_graph("definitions_graph.json")
        
    def _load_graph(self, filename: str) -> nx.DiGraph:
        """Load a graph from a JSON file into NetworkX"""
        G = nx.DiGraph()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for node in data['nodes']:
                    G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
                for edge in data['edges']:
                    G.add_edge(edge['source'], edge['target'], **{k: v for k, v in edge.items() if k not in ['source', 'target']})
            logger.info(f"Loaded graph {filename} with {len(G.nodes)} nodes")
        except FileNotFoundError:
            logger.warning(f"Graph file {filename} not found, using empty graph")
        return G

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform Naive Graph RAG query"""
        logger.info(f"Naive Graph RAG query: {question}")
        
        # 1. Hybrid Retrieval (Vector + BM25)
        # This respects the user's requirement for hybrid search with BGE-M3 and BM25
        retrieved_nodes, retrieval_info = self.retrieval_system.retrieve_with_fusion(question, top_k=top_k)
        
        # 2. Graph Expansion (Naive 1-hop)
        expanded_nodes = list(retrieved_nodes)
        expanded_ids = {node.node_id for node in retrieved_nodes}
        
        for node in retrieved_nodes:
            if node.node_id in self.lexical_graph:
                # Get immediate neighbors (children/references)
                neighbors = list(self.lexical_graph.neighbors(node.node_id))
                for neighbor_id in neighbors:
                    if neighbor_id not in expanded_ids:
                        neighbor_node = self.retrieval_system.get_node_by_id(neighbor_id)
                        if neighbor_node:
                            expanded_nodes.append(neighbor_node)
                            expanded_ids.add(neighbor_id)
                            
        # 3. Definition Lookup
        # Query definitions graph for keywords in the query
        definitions = self.graph_builder.query_definitions_graph(question, top_k=5)
        definition_context = ""
        if definitions:
            definition_context = "\nDéfinitions juridiques :\n" + "\n".join([
                f"- {d.get('term', d.get('node_id', '')): <15}: {d.get('content', d.get('definition', ''))}" 
                for d in definitions
            ])

        # 4. Context Construction
        context_parts = []
        for i, node in enumerate(expanded_nodes):
            source = node.metadata.get('source_file', 'Document')
            clause = node.metadata.get('clause_id', 'N/A')
            context_parts.append(f"[Source {i+1}: {source} - Art. {clause}]\n{node.content}")
            
        full_context = "\n\n".join(context_parts) + "\n" + definition_context
        
        # 5. One-shot Answering
        prompt = f"""
STRICT INSTRUCTION: Vous êtes un assistant juridique expert. Votre réponse doit être précise et UNIQUEMENT basée sur le contexte fourni. 

CONTEXTE :
{full_context}

QUESTION :
{question}

RÈGLES :
1. Répondez de manière structurée.
2. Citez les Articles exacts pour chaque point.
3. Si l'information n'est pas dans le contexte, dites-le.
4. Terminez par "Sources consultées :".
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Assistant juridique expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in answering: {e}")
            answer = f"Error generating answer: {str(e)}"
            
        return {
            "answer": answer,
            "retrieved_nodes": len(retrieved_nodes),
            "expanded_nodes": len(expanded_nodes),
            "definitions_found": len(definitions),
            "context_used": full_context
        }
