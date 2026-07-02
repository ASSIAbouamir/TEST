import json
import os

def process_mauritanie():
    input_file = "Mau164741_repaired.json"
    output_file = os.path.join("data_processed", "Rejet hydrocarbure_Mauritanie_processed.json")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    clauses = raw_data.get('data', {}).get('clauses', [])
    metadata = raw_data.get('data', {}).get('document_metadata', {})
    
    processed_data = {
        "theme": "Rejet hydrocarbure",
        "country": "Mauritanie",
        "nodes": []
    }
    
    for i, clause in enumerate(clauses):
        node = {
            "node_id": f"Mau_clause_node_{i+1}",
            "text": clause.get('full_text', ''),
            "summary": clause.get('title_or_summary', ''),
            "country": "Mauritanie",
            "theme": "Rejet hydrocarbure",
            "metadata": {
                "title": metadata.get('title', ''),
                "file_name": metadata.get('filename', '')
            }
        }
        processed_data['nodes'].append(node)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Formatage terminé ! {len(processed_data['nodes'])} nœuds créés dans {output_file}")

if __name__ == "__main__":
    process_mauritanie()
