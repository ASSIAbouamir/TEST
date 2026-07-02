"""
Legal Document RAG: Multi-Graph Multi-Agent Recursive Retrieval System

This system implements the concept described in the article "Legal Document RAG: 
Multi-Graph Multi-Agent Recursive Retrieval through Legal Clauses" using:
- Document hierarchy graph (lexical graph)
- Legal definitions graph  
- Multi-agent system with LangGraph
- Hybrid retrieval (Vector, BM25, Keyword)
- Recursive navigation through legal clauses
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .config import settings
from .models import DocumentStructure, DocumentNode, MultiAgentSearchLocalNode
from .document_parser import DocumentParser
from .graph_builder import GraphBuilder
from .retrieval_system import LegalRetrievalSystem
from .agents import LegalDocumentAgents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalRAGSystem:
    """Main Legal RAG System orchestrating all components"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.graph_builder = GraphBuilder()
        self.retrieval_system = LegalRetrievalSystem()
        self.agents = None
        self.document_structure = None
        self.local_nodes_map = {}
        
    def load_document(self, document_path: str, document_id: str = None) -> DocumentStructure:
        """Load and parse a legal document"""
        logger.info(f"Loading document: {document_path}")
        
        if not document_id:
            document_id = Path(document_path).stem
        
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Parse document
        self.document_structure = self.parser.parse_document(document_text, document_id)
        
        logger.info(f"Document parsed: {len(self.document_structure.hierarchy)} nodes, {len(self.document_structure.definitions)} definitions")
        return self.document_structure
    
    def build_graphs(self) -> tuple[str, str]:
        """Build lexical and definitions graphs"""
        if not self.document_structure:
            raise ValueError("No document loaded. Call load_document first.")
        
        logger.info("Building knowledge graphs...")
        
        # Create lexical graph
        lexical_graph_id = self.graph_builder.create_lexical_graph(self.document_structure)
        
        # Create definitions graph
        definitions_graph_id = self.graph_builder.create_definitions_graph(self.document_structure)
        
        logger.info(f"Graphs built: lexical={lexical_graph_id}, definitions={definitions_graph_id}")
        return lexical_graph_id, definitions_graph_id
    
    def setup_retrieval(self):
        """Setup the retrieval system"""
        if not self.document_structure:
            raise ValueError("No document loaded. Call load_document first.")
        
        logger.info("Setting up retrieval system...")
        
        # Index document nodes
        self.retrieval_system.index_documents(self.document_structure.hierarchy)
        
        # Create local nodes map for agents
        self.local_nodes_map = {}
        for node in self.document_structure.hierarchy:
            self.local_nodes_map[node.node_id] = MultiAgentSearchLocalNode(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                page_number=node.page_number,
                metadata=node.metadata
            )
        
        logger.info("Retrieval system setup complete")
    
    def setup_agents(self):
        """Setup the multi-agent system"""
        logger.info("Setting up multi-agent system...")
        
        self.agents = LegalDocumentAgents(self.retrieval_system, self.graph_builder)
        self.agents.set_local_nodes_map(self.local_nodes_map)
        
        logger.info("Multi-agent system setup complete")
    
    def initialize_system(self, document_path: str, document_id: str = None):
        """Initialize the complete system with a document"""
        logger.info("Initializing Legal RAG System...")
        
        # Load and parse document
        self.load_document(document_path, document_id)
        
        # Build graphs
        self.build_graphs()
        
        # Setup retrieval
        self.setup_retrieval()
        
        # Setup agents
        self.setup_agents()
        
        logger.info("System initialization complete")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the system with a legal question"""
        if not self.agents:
            raise ValueError("System not initialized. Call initialize_system first.")
        
        logger.info(f"Processing query: {question}")
        
        # Run multi-agent workflow
        result_state = self.agents.run_query(question)
        
        # Prepare response
        response = {
            'question': question,
            'answer': result_state.final_answer,
            'retrieved_nodes': len(result_state.previous_nodes) + len(result_state.last_fetched_context_nodes),
            'passes': result_state.pass_count,
            'definitions_found': len(result_state.definitions),
            'search_failures': result_state.search_failures,
            'retrieved_sections': [
                {
                    'node_id': node.node_id,
                    'type': node.node_type,
                    'page': node.page_number,
                    'content': node.content[:200] + "..." if len(node.content) > 200 else node.content
                }
                for node in result_state.previous_nodes + result_state.last_fetched_context_nodes
            ],
            'definitions': result_state.definitions
        }
        
        return response
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.document_structure:
            return {"status": "not_initialized"}
        
        return {
            'status': 'initialized',
            'document_id': self.document_structure.document_id,
            'total_nodes': len(self.document_structure.hierarchy),
            'total_definitions': len(self.document_structure.definitions),
            'lexical_graph_id': settings.LEXICAL_GRAPH_ID,
            'definitions_graph_id': settings.DEFINITIONS_GRAPH_ID,
            'indexed_nodes': len(self.local_nodes_map)
        }

def main():
    """Main function for testing the system"""
    # Example usage
    system = LegalRAGSystem()
    
    # Check if document exists
    document_path = "sample_legal_document.txt"
    if not os.path.exists(document_path):
        print(f"Sample document not found: {document_path}")
        print("Please create a sample legal document to test the system.")
        return
    
    try:
        # Initialize system
        system.initialize_system(document_path)
        
        # Print system stats
        stats = system.get_system_stats()
        print("\\nSystem Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example queries
        test_queries = [
            "How can the Board and the CCO manage control functions?",
            "What are the responsibilities of the Chief Compliance Officer?",
            "What is the definition of 'Senior Management'?"
        ]
        
        print("\\n" + "="*70)
        print("TESTING LEGAL RAG SYSTEM")
        print("="*70)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\\nQuery {i}: {query}")
            print("-" * 50)
            
            try:
                result = system.query(query)
                
                print(f"Answer: {result['answer']}")
                print(f"Retrieved {result['retrieved_nodes']} sections in {result['passes']} passes")
                print(f"Found {result['definitions_found']} definitions")
                
                if result['search_failures']:
                    print(f"Search failures: {len(result['search_failures'])}")
                
                print("\\nRetrieved Sections:")
                for section in result['retrieved_sections'][:5]:  # Show first 5
                    print(f"  [{section['node_id']}] Page {section['page']}: {section['content']}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
