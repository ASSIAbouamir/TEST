"""
Test script to verify data loading and graph creation without requiring API keys
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .data_loader import ProcessedDataLoader
from .document_parser import DocumentParser
from .graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test the data loading functionality"""
    print("=" * 70)
    print("TESTING DATA LOADING AND GRAPH CREATION")
    print("=" * 70)
    
    # Initialize data loader
    loader = ProcessedDataLoader()
    
    # Show available data
    summary = loader.get_document_summary()
    print(f"Available themes: {summary['total_themes']}")
    print(f"Total documents: {summary['total_documents']}")
    
    for theme, info in summary['themes'].items():
        print(f"\n📁 {theme}: {info['countries']} countries")
    
    # Test loading a specific document
    print("\n" + "=" * 70)
    print("TESTING DOCUMENT LOADING")
    print("=" * 70)
    
    # Load Baleine_Bénin document
    document = loader.load_document("Baleine", "Bénin")
    
    if document:
        print(f"✅ Successfully loaded: {document.title}")
        print(f"   Document ID: {document.document_id}")
        print(f"   Total nodes: {len(document.hierarchy)}")
        print(f"   Total definitions: {len(document.definitions)}")
        
        # Show sample nodes
        print(f"\n📄 Sample nodes:")
        for i, node in enumerate(document.hierarchy[:5]):
            print(f"   {i+1}. [{node.node_id}] {node.node_type} (Page {node.page_number})")
            print(f"      {node.content[:100]}...")
        
        # Show sample definitions
        if document.definitions:
            print(f"\n📚 Sample definitions:")
            for i, definition in enumerate(document.definitions[:3]):
                print(f"   {i+1}. {definition.term}: {definition.definition[:80]}...")
        
    else:
        print("❌ Failed to load document")
        return False
    
    return True

def test_graph_creation():
    """Test graph creation without API dependencies"""
    print("\n" + "=" * 70)
    print("TESTING GRAPH CREATION")
    print("=" * 70)
    
    # Load document
    loader = ProcessedDataLoader()
    document = loader.load_document("Baleine", "Bénin")
    
    if not document:
        print("❌ No document loaded")
        return False
    
    # Create graphs
    graph_builder = GraphBuilder()
    
    try:
        # Create lexical graph
        lexical_id = graph_builder.create_lexical_graph(document)
        print(f"✅ Lexical graph created: {lexical_id}")
        
        # Create definitions graph
        definitions_id = graph_builder.create_definitions_graph(document)
        print(f"✅ Definitions graph created: {definitions_id}")
        
        # Test graph querying (local)
        print(f"\n🔍 Testing local graph queries...")
        
        # Test lexical graph query
        lexical_results = graph_builder.query_lexical_graph("baleine", top_k=5)
        print(f"   Lexical graph results: {len(lexical_results)} nodes")
        
        # Test definitions graph query
        definition_results = graph_builder.query_definitions_graph("définition", top_k=5)
        print(f"   Definitions graph results: {len(definition_results)} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return False

def test_multiple_documents():
    """Test loading multiple documents"""
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE DOCUMENTS")
    print("=" * 70)
    
    loader = ProcessedDataLoader()
    
    # Test loading multiple whale documents
    whale_countries = ["Bénin", "Cameroun", "Sénégal"]
    loaded_docs = []
    
    for country in whale_countries:
        doc = loader.load_document("Baleine", country)
        if doc:
            loaded_docs.append(doc)
            print(f"✅ Loaded {country}: {len(doc.hierarchy)} nodes, {len(doc.definitions)} definitions")
        else:
            print(f"❌ Failed to load {country}")
    
    print(f"\n📊 Summary: {len(loaded_docs)}/{len(whale_countries)} whale documents loaded")
    
    # Test different themes
    themes = ["Baleine", "Oiseaux marins", "Rejet hydrocarbure"]
    
    for theme in themes:
        countries = loader.get_countries_for_theme(theme)
        print(f"📁 {theme}: {len(countries)} countries available")
        
        # Load first document from each theme
        if countries:
            doc = loader.load_document(theme, countries[0])
            if doc:
                print(f"   ✅ Sample {countries[0]}: {len(doc.hierarchy)} nodes")
    
    return True

def test_document_analysis():
    """Test document structure analysis"""
    print("\n" + "=" * 70)
    print("TESTING DOCUMENT STRUCTURE ANALYSIS")
    print("=" * 70)
    
    loader = ProcessedDataLoader()
    document = loader.load_document("Baleine", "Bénin")
    
    if not document:
        print("❌ No document loaded")
        return False
    
    # Analyze node types
    node_types = {}
    for node in document.hierarchy:
        node_type = node.node_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"📊 Node type distribution:")
    for node_type, count in sorted(node_types.items()):
        print(f"   {node_type}: {count}")
    
    # Analyze links and footnotes
    total_links = sum(len(node.links_to) for node in document.hierarchy)
    total_footnotes = sum(len(node.footnotes) for node in document.hierarchy)
    
    print(f"\n🔗 Cross-references: {total_links} links across {len(document.hierarchy)} nodes")
    print(f"📝 Footnotes: {total_footnotes} footnotes")
    
    # Show nodes with most links
    nodes_with_links = [(node, len(node.links_to)) for node in document.hierarchy if node.links_to]
    nodes_with_links.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔗 Most referenced nodes:")
    for node, link_count in nodes_with_links[:5]:
        print(f"   [{node.node_id}] {link_count} links: {node.content[:50]}...")
    
    return True

def main():
    """Run all tests"""
    print("🧪 LEGAL RAG SYSTEM - DATA LOADING TESTS")
    print("=" * 70)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Graph Creation", test_graph_creation),
        ("Multiple Documents", test_multiple_documents),
        ("Document Analysis", test_document_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All data loading tests passed!")
        print("\n📝 Next steps:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Run: python main_data_integration.py")
        print("3. Try the interactive mode for querying documents")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
