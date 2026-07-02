"""
Main integration script for Legal RAG System with processed data from data_processed directory
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .config import settings
from .models import DocumentStructure
from .document_parser import DocumentParser
from .graph_builder import GraphBuilder
from .retrieval_system import LegalRetrievalSystem
from .agents import LegalDocumentAgents
from .data_loader import ProcessedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalRAGDataIntegration:
    """Legal RAG System integrated with processed legal documents"""
    
    def __init__(self, data_processed_path: str = None):
        self.data_loader = ProcessedDataLoader(data_processed_path)
        self.graph_builder = GraphBuilder()
        self.retrieval_system = LegalRetrievalSystem()
        self.agents = None
        self.current_document = None
        self.local_nodes_map = {}
        self.meta_index = None # Index of document summaries
        self.doc_registry = {} # Map of doc_id to theme/country
        self.law_title_map = {} # Map of source_file to real law title
        
    def show_available_data(self):
        """Display available themes and countries"""
        summary = self.data_loader.get_document_summary()
        
        print("=" * 70)
        print("AVAILABLE LEGAL DOCUMENTS")
        print("=" * 70)
        
        print(f"Total Themes: {summary['total_themes']}")
        print(f"Total Documents: {summary['total_documents']}")
        print()
        
        for theme, info in summary['themes'].items():
            print(f"📁 {theme.replace('_', ' ').title()}")
            print(f"   Countries: {info['countries']}")
            print(f"   List: {', '.join(info['country_list'])}")
            print()
    
    def load_document_by_theme_country(self, theme: str, country: str) -> DocumentStructure:
        """Load a specific document by theme and country"""
        logger.info(f"Loading document: {theme} - {country}")
        
        document = self.data_loader.load_document(theme, country)
        
        if document:
            self.current_document = document
            logger.info(f"Loaded document with {len(document.hierarchy)} nodes and {len(document.definitions)} definitions")
        else:
            logger.error(f"Failed to load document: {theme} - {country}")
        
        return document
    
    def load_all_documents(self):
        """Load meta-information and build a meta-index for fast global routing"""
        logger.info("Building meta-index for global routing...")
        
        meta_nodes = []
        summary = self.data_loader.get_document_summary()
        
        from llama_index.core.schema import TextNode
        
        for theme, info in summary['themes'].items():
            for country in info['country_list']:
                # Create a summary node for this document
                doc_id = f"{theme}_{country}"
                # Try to get the real title from the document file
                real_title = f"Lois sur {theme} ({country})"
                try:
                    file_path = os.path.join(self.data_loader.data_processed_path, f"{theme}_{country}_processed.json")
                    if os.path.exists(file_path):
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            nodes = data.get("nodes", [])
                            if nodes:
                                # First node usually contains the title
                                first_text = nodes[0].get("text", "")
                                if "# LOI" in first_text.upper() or "# CODE" in first_text.upper():
                                    real_title = first_text.replace("#", "").strip().split("\n")[0]
                                elif len(nodes) > 1 and ("# LOI" in nodes[1].get("text", "").upper() or "# CODE" in nodes[1].get("text", "").upper()):
                                    real_title = nodes[1].get("text", "").replace("#", "").strip().split("\n")[0]
                                
                                # Store in map for agents to use
                                source_file = nodes[0].get("law_name", "")
                                if source_file:
                                    self.law_title_map[source_file] = real_title
                except Exception as e:
                    logger.warning(f"Could not extract title for {theme}_{country}: {e}")

                content = f"Lois et réglementations sur le thème {theme} pour le pays {country}. Titre: {real_title}"
                
                node = TextNode(
                    text=content,
                    id_=doc_id,
                    metadata={'theme': theme, 'country': country, 'title': real_title}
                )
                meta_nodes.append(node)
                self.doc_registry[doc_id] = (theme, country)
        
        # Build the meta-index (very fast, only ~42 nodes)
        from llama_index.core import VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        
        embed_model = self.retrieval_system.embed_model
        if embed_model is None:
            logger.info("Using local offline MockEmbedding for global routing index.")
            embed_model = MockEmbedding(embed_dim=384)
            
        self.meta_index = VectorStoreIndex(nodes=meta_nodes, embed_model=embed_model)
        
        logger.info("Meta-index complete. Ready for global routing.")
        return {
            'total_documents': len(meta_nodes),
            'mode': 'Meta-RAG'
        }

    def global_query(self, question: str) -> dict:
        """Find the best document and query it"""
        if not self.meta_index:
            return self.query(question)
            
        best_doc_id = None
        
        # 1. Exact text matching for countries and themes (fixes vector embedding confusion)
        lower_q = question.lower()
        is_baleine_query = any(k in lower_q for k in ["baleine", "cétacé", "cetace", "mammifère"])
        is_hydro_query = any(k in lower_q for k in ["hydro", "rejet", "pollution"])
        
        for doc_id, (d_theme, d_country) in self.doc_registry.items():
            if d_country.lower() in lower_q:
                if is_baleine_query and d_theme.lower() == "baleine":
                    best_doc_id = doc_id
                    break
                elif is_hydro_query and "hydro" in d_theme.lower():
                    best_doc_id = doc_id
                    break
                    
        # Fallback if no precise theme match
        if not best_doc_id:
            for doc_id, (d_theme, d_country) in self.doc_registry.items():
                if d_country.lower() in lower_q:
                    best_doc_id = doc_id
                    break
                
        # 2. Fallback to vector search if no country explicitly mentioned
        if not best_doc_id:
            retriever = self.meta_index.as_retriever(similarity_top_k=1)
            results = retriever.retrieve(question)
            
            if not results:
                return {"answer": "Aucun document pertinent trouvé.", "sources": []}
                
            best_doc_id = results[0].node.id_
            
        theme, country = self.doc_registry[best_doc_id]
        
        logger.info(f"Global Router: Redirecting to {theme} - {country}")
        
        # 2. Load and setup if not already current
        if not self.current_document or f"{theme}_{country}" != self.current_document.document_id:
            doc = self.load_document_by_theme_country(theme, country)
            self.setup_system_for_document(doc)
            
        # 3. Perform the actual query
        return self.query(question)

    def setup_system_for_document(self, document: DocumentStructure = None):
        """Setup the complete RAG system for a document"""
        if document is None:
            document = self.current_document
        
        if not document:
            raise ValueError("No document loaded. Call load_document_by_theme_country first.")
        
        logger.info(f"Setting up RAG system for {document.document_id}")
        
        # Build graphs
        lexical_graph_id = self.graph_builder.create_lexical_graph(document)
        definitions_graph_id = self.graph_builder.create_definitions_graph(document)
        
        # Setup retrieval
        self.retrieval_system.index_documents(document.hierarchy)
        
        # Create local nodes map
        self.local_nodes_map = {}
        for node in document.hierarchy:
            from .models import MultiAgentSearchLocalNode
            self.local_nodes_map[node.node_id] = MultiAgentSearchLocalNode(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                page_number=node.page_number,
                metadata=node.metadata
            )
        
        # Setup agents
        self.agents = LegalDocumentAgents(self.retrieval_system, self.graph_builder)
        self.agents.set_law_title_map(self.law_title_map)
        self.agents.set_local_nodes_map(self.local_nodes_map)
        
        logger.info("System setup complete")
        return {
            'lexical_graph_id': lexical_graph_id,
            'definitions_graph_id': definitions_graph_id,
            'total_nodes': len(document.hierarchy),
            'total_definitions': len(document.definitions)
        }
    
    def query(self, question: str) -> dict:
        """Query the current loaded document"""
        if not self.agents:
            raise ValueError("System not setup. Call setup_system_for_document first.")
        
        logger.info(f"Querying: {question}")
        
        result_state = self.agents.run_query(question)
        
        # Post-process answer to replace technical law names with real titles
        raw_answer = result_state['final_answer'] or ""
        for tech_name, real_title in self.law_title_map.items():
            if tech_name in raw_answer:
                raw_answer = raw_answer.replace(tech_name, real_title)
                
        # --- NETTOYAGE ET FORMATAGE STRICT (Oui/Non en premier) ---
        # Extraire le verdict
        verdict_val = "Non"
        if "Verdict : Oui" in raw_answer or "Verdict: Oui" in raw_answer:
            verdict_val = "Oui"
        elif "Verdict : Non" in raw_answer or "Verdict: Non" in raw_answer:
            verdict_val = "Non"
        elif "Oui" in raw_answer[:30]:
            verdict_val = "Oui"
            
        # Extraire le texte de reflexion
        import re
        cleaned_text = raw_answer
        reflexion_match = re.search(r"<reflexion>(.*?)</reflexion>", raw_answer, re.DOTALL)
        if reflexion_match:
            cleaned_text = reflexion_match.group(1).strip()
        else:
            cleaned_text = cleaned_text.replace("<reflexion>", "").replace("</reflexion>", "").strip()
            
        # Nettoyer les résidus de verdict à la fin
        cleaned_text = re.sub(r"Verdict\s*:\s*(Oui|Non)", "", cleaned_text, flags=re.IGNORECASE).strip()
        
        # Enlever les references aux parties A et B pour la proprete
        cleaned_text = re.sub(r"Partie [A|B]\s*\(.*?\)\s*:\s*(Oui|Non)", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"Partie [A|B]\s*:\s*(Oui|Non|non|oui)", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\**Partie [A|B]\**.*?:?", "", cleaned_text, flags=re.IGNORECASE)
        
        # S'assurer que la reponse commence par le bon verdict
        first_word = "Oui" if verdict_val == "Oui" else "Non"
        
        # Check if LLM response already starts with Oui/Non (with any punctuation)
        prefix_match = re.match(r"^(Oui|Non)([\s,.]*)", cleaned_text, flags=re.IGNORECASE)
        if prefix_match:
            original_separator = prefix_match.group(2)
            if not original_separator or not any(c in original_separator for c in [",", "."]):
                original_separator = ". "
            rest = re.sub(r"^(Oui|Non)[\s,.]*", "", cleaned_text, flags=re.IGNORECASE).strip()
            if rest:
                final_answer = f"{first_word}{original_separator}{rest}"
            else:
                final_answer = f"{first_word}."
        else:
            final_answer = cleaned_text
        
        response = {
            'question': question,
            'answer': final_answer,
            'retrieved_nodes': len(result_state['previous_nodes']) + len(result_state['last_fetched_context_nodes']),
            'passes': result_state['pass_count'],
            'definitions_found': len(result_state['definitions']),
            'search_failures': result_state['search_failures'],
            'retrieved_sections': [
                {
                    'node_id': node.node_id,
                    'type': node.node_type,
                    'page': node.page_number,
                    'country': node.metadata.get('country', 'Unknown'),
                    'clause_id': node.metadata.get('clause_id', ''),
                    'law_name': node.metadata.get('source_file', node.metadata.get('law_name', 'Unknown')),
                    'real_title': self.law_title_map.get(node.metadata.get('source_file', node.metadata.get('law_name', '')), ''),
                    'content': node.content[:200] + "..." if len(node.content) > 200 else node.content
                }
                for node in result_state['previous_nodes'] + result_state['last_fetched_context_nodes']
            ],
            'definitions': result_state['definitions']
        }
        
        return response
    
    def test_baleine_benin_document(self):
        """Test with the Baleine_Bénin document"""
        print("=" * 70)
        print("TESTING WITH BALEINE_BÉNIN DOCUMENT")
        print("=" * 70)
        
        # Load the document
        document = self.load_document_by_theme_country("Baleine", "Bénin")
        
        if not document:
            print("❌ Failed to load Baleine_Bénin document")
            return
        
        print(f"✅ Loaded document: {document.title}")
        print(f"   Nodes: {len(document.hierarchy)}")
        print(f"   Definitions: {len(document.definitions)}")
        
        # Setup system
        setup_info = self.setup_system_for_document(document)
        print(f"✅ System setup complete")
        print(f"   Lexical Graph: {setup_info['lexical_graph_id']}")
        print(f"   Definitions Graph: {setup_info['definitions_graph_id']}")
        
        # Test queries related to whale protection
        test_queries = [
            "Quelles sont les dispositions générales de la loi sur les baleines?",
            "Qu'est-ce qui est interdit concernant les baleines au Bénin?",
            "Quelles sont les sanctions pour chasse illégale de baleines?",
            "Qui est responsable de la protection des cétacés?",
            "Quelles autorisations sont nécessaires pour les activités liées aux baleines?"
        ]
        
        print("\n" + "=" * 70)
        print("TESTING QUERIES")
        print("=" * 70)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 50)
            
            try:
                result = self.query(query)
                
                print(f"✅ Processed - {result['retrieved_nodes']} nodes, {result['passes']} passes")
                print(f"   Answer: {result['answer'][:300]}...")
                
                if result['definitions_found'] > 0:
                    print(f"   Definitions found: {list(result['definitions'].keys())}")
                
                # Show top retrieved sections
                print("   Top sections:")
                for j, section in enumerate(result['retrieved_sections'][:3], 1):
                    print(f"     {j}. [{section['node_id']}] {section['clause_id']}: {section['content'][:100]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_multiple_themes(self):
        """Test with multiple themes and countries"""
        print("=" * 70)
        print("TESTING MULTIPLE THEMES")
        print("=" * 70)
        
        themes = self.data_loader.get_available_themes()
        
        for theme in themes[:3]:  # Test first 3 themes
            print(f"\n📁 Testing theme: {theme}")
            countries = self.data_loader.get_countries_for_theme(theme)
            
            # Test first 2 countries for each theme
            for country in countries[:2]:
                print(f"  🇺🇸 {country}")
                
                try:
                    document = self.load_document_by_theme_country(theme, country)
                    if document:
                        setup_info = self.setup_system_for_document(document)
                        
                        # Test a simple query
                        query = f"Quelles sont les principales dispositions sur {theme.replace('_', ' ')}?"
                        result = self.query(query)
                        
                        print(f"    ✅ {result['retrieved_nodes']} nodes retrieved")
                    else:
                        print(f"    ❌ Failed to load")
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
    
    def interactive_mode(self):
        """Interactive mode for querying documents"""
        print("=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        
        # Show available data
        self.show_available_data()
        
        while True:
            print("\n" + "-" * 50)
            print("Options:")
            print("1. Load document by theme and country")
            print("2. Query current document")
            print("3. Show current document info")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                theme = input("Enter theme (e.g., Baleine, Oiseaux marins): ").strip()
                country = input("Enter country (e.g., Bénin, Cameroun): ").strip()
                
                document = self.load_document_by_theme_country(theme, country)
                if document:
                    self.setup_system_for_document(document)
                    print(f"✅ Loaded: {document.title}")
                else:
                    print("❌ Failed to load document")
            
            elif choice == "2":
                if not self.agents:
                    print("❌ No document loaded. Please load a document first.")
                    continue
                
                query = input("Enter your question: ").strip()
                if query:
                    result = self.query(query)
                    print(f"\n📝 Answer: {result['answer']}")
                    print(f"\n📊 Retrieved {result['retrieved_nodes']} sections in {result['passes']} passes")
                    
                    if result['definitions_found'] > 0:
                        print(f"\n📚 Definitions:")
                        for term, definition in result['definitions'].items():
                            print(f"  {term}: {definition}")
            
            elif choice == "3":
                if self.current_document:
                    print(f"Current document: {self.current_document.title}")
                    print(f"Nodes: {len(self.current_document.hierarchy)}")
                    print(f"Definitions: {len(self.current_document.definitions)}")
                    print(f"System ready: {'Yes' if self.agents else 'No'}")
                else:
                    print("No document loaded")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")

def main():
    """Main function"""
    print("🐋 LEGAL RAG SYSTEM - DATA INTEGRATION")
    print("=" * 70)
    
    # Initialize the system
    system = LegalRAGDataIntegration()
    
    # Check if data is available
    if not system.data_loader.available_files:
        print("❌ No processed data found. Please ensure data_processed directory exists.")
        return
    
    print("✅ Data loaded successfully")
    
    # Show available data
    system.show_available_data()
    
    # Test with Baleine_Bénin document
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED TEST")
    print("=" * 70)
    
    try:
        system.test_baleine_benin_document()
        
        # Ask if user wants interactive mode
        response = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            system.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
