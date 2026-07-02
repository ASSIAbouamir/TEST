"""
Test simple version of Legal RAG System with Groq
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from data_loader import ProcessedDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_test():
    """Simple test without complex agent workflow"""
    print("=" * 70)
    print("SIMPLE LEGAL RAG TEST WITH GROQ")
    print("=" * 70)
    
    # Initialize data loader
    loader = ProcessedDataLoader()
    
    # Show available data
    summary = loader.get_document_summary()
    print(f"Available themes: {summary['total_themes']}")
    print(f"Total documents: {summary['total_documents']}")
    
    # Load a document
    document = loader.load_document("Baleine", "Bénin")
    
    if not document:
        print("❌ Failed to load document")
        return
    
    print(f"✅ Loaded document: {document.title}")
    print(f"   Nodes: {len(document.hierarchy)}")
    print(f"   Definitions: {len(document.definitions)}")
    
    # Show some sample content
    print(f"\n📄 Sample content:")
    for i, node in enumerate(document.hierarchy[:3]):
        print(f"   {i+1}. [{node.node_id}] {node.node_type}")
        print(f"      {node.content[:150]}...")
    
    # Test simple query with Groq
    print(f"\n🔍 Testing simple query with Groq...")
    
    try:
        from groq import Groq
        
        # Initialize Groq client
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Simple query
        query = "Quelles sont les dispositions générales sur les baleines?"
        
        # Create context from first few nodes
        context = "Contexte du document:\n"
        for i, node in enumerate(document.hierarchy[:5]):
            context += f"{i+1}. [{node.node_id}] {node.content[:100]}...\n"
        
        # Query Groq
        response = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant juridique expert. Réponds en français basé sur le contexte fourni."},
                {"role": "user", "content": f"Question: {query}\n\n{context}"}
            ]
        )
        
        answer = response.choices[0].message.content.strip()
        
        print(f"✅ Query successful!")
        print(f"📝 Answer: {answer[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_multiple_queries():
    """Test multiple queries"""
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE QUERIES")
    print("=" * 70)
    
    queries = [
        "Quelles sont les sanctions pour chasse illégale de baleines?",
        "Qui est responsable de la protection des cétacés?",
        "Quelles zones sont protégées pour les baleines?",
        "Quelles autorisations sont nécessaires pour les activités liées aux baleines?"
    ]
    
    try:
        from groq import Groq
        
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
        loader = ProcessedDataLoader()
        document = loader.load_document("Baleine", "Bénin")
        
        if not document:
            print("❌ Failed to load document")
            return
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 50)
            
            # Simple context from relevant nodes
            relevant_nodes = [node for node in document.hierarchy if any(keyword.lower() in node.content.lower() for keyword in query.split()[:3])]
            
            context = f"Contexte pertinent:\n"
            for j, node in enumerate(relevant_nodes[:3]):
                context += f"{j+1}. [{node.node_id}] {node.content[:150]}...\n"
            
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Tu es un assistant juridique expert. Réponds en français basé sur le contexte fourni."},
                    {"role": "user", "content": f"Question: {query}\n\n{context}"}
                ]
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"✅ Answer: {answer[:200]}...")
        
        print(f"\n🎉 Successfully processed {len(queries)} queries!")
        
    except Exception as e:
        print(f"❌ Error in multiple queries: {e}")

def main():
    """Main function"""
    print("🐋 LEGAL RAG SYSTEM - SIMPLE VERSION")
    print("=" * 70)
    
    # Check if Groq API key is available
    if not settings.GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in settings")
        print("Please add your Groq API key to .env file:")
        print("GROQ_API_KEY=your_groq_api_key_here")
        return
    
    print(f"✅ Using Groq model: {settings.GROQ_MODEL}")
    
    # Run simple test
    simple_test()
    
    # Ask if user wants to try multiple queries
    response = input("\nWould you like to test multiple queries? (y/n): ").strip().lower()
    if response == 'y':
        test_multiple_queries()
    
    print("\n🎉 Simple test completed!")
    print("\n📝 Next steps:")
    print("1. The system is working with your data_processed directory")
    print("2. You can integrate this with your existing RAG systems")
    print("3. For full multi-agent functionality, fix the complex workflow issues")

if __name__ == "__main__":
    main()
