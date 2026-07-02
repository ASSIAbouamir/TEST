import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Insert the project root to python path
sys.path.insert(0, os.path.abspath("."))

from legal_rag.main_data_integration import LegalRAGDataIntegration

rag = LegalRAGDataIntegration(data_processed_path="data_processed")
info = rag.load_all_documents()

theme, country = "Rejet hydrocarbure", "Bénin"
print(f"\nLoading document: {theme} - {country}")
doc = rag.load_document_by_theme_country(theme, country)
rag.setup_system_for_document(doc)

# Query
question = "Est-ce que le rejet d'hydrocarbures au Bénin entraîne une peine de prison ?"
print(f"Running query: '{question}'...")
res = rag.query(question)
print("\n" + "="*50)
print("ANSWER:")
print(res.get("answer"))
print("="*50)
print("RETRIEVED SECTIONS:")
for sec in res.get("retrieved_sections", []):
    print(f"- Node ID: {sec['node_id']} | Law: {sec['law_name']} | Clause: {sec.get('clause_id')} | Title: {sec.get('real_title')}")
    print(f"  Content: {sec['content'][:300]}...")
print("="*50)
