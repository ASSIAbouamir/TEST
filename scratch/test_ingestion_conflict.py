import os
import sys
import shutil
from pathlib import Path

# Ajouter le répertoire racine au chemin de recherche Python
sys.path.insert(0, str(Path(__file__).parent.parent))

# Charger les variables d'environnement depuis le fichier .env
from dotenv import load_dotenv
load_dotenv()

from src.parser import LegalDocumentParser

def run_test():
    # Définir les chemins
    base_dir = Path(__file__).parent.parent
    archive_pdf = base_dir / "data" / "archive" / "Mad182928.pdf"
    input_pdf = base_dir / "data" / "input" / "Mad182928_test.pdf"
    
    if not archive_pdf.exists():
        print(f"[ERREUR] Le fichier source {archive_pdf} n'existe pas.")
        return
        
    # S'assurer que le dossier d'entrée existe et y copier le fichier
    input_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(archive_pdf), str(input_pdf))
    print(f"[TEST] Fichier copié de {archive_pdf} vers {input_pdf}")
    
    # Lancer le parser
    parser = LegalDocumentParser()
    print("[TEST] Début de l'ingestion avec détection de conflits...")
    
    def progress_cb(percent, message):
        print(f"[{percent}%] {message}")
        
    md_path, json_path, conflicts = parser.parse_file(
        file_path=input_pdf,
        theme="TBT",
        country="Madagascar",
        progress_callback=progress_cb
    )
    
    print("\n--- RÉSULTATS DU PARSER ---")
    print(f"Markdown généré : {md_path}")
    print(f"Nodes RAG générés : {json_path}")
    print(f"Nombre de conflits détectés : {len(conflicts)}")
    for i, cf in enumerate(conflicts):
        print(f"Conflit {i+1} :")
        print(f"  - Loi A (Nouvelle) : {cf.get('new_law')} ({cf.get('new_node_id')})")
        print(f"  - Loi B (Ancienne) : {cf.get('old_law')} ({cf.get('old_node_id')})")
        print(f"  - Explication : {cf.get('explanation')}")
        
    # Vérifier si des fichiers d'alertes ont été générés dans data/alerts ou data/reports
    print("\n--- VÉRIFICATION DES RAPPORTS D'ALERTES ---")
    alerts_dir = base_dir / "data" / "alerts"
    reports_dir = base_dir / "data" / "reports"
    
    print("Alertes locales générées :")
    if alerts_dir.exists():
        for f in alerts_dir.glob("rapport_*.md"):
            print(f"  - {f.name}")
    else:
        print("  Aucun dossier d'alertes.")
        
    print("Rapports HTML de veille générés :")
    if reports_dir.exists():
        for f in reports_dir.glob("rapport_juridique_*.html"):
            print(f"  - {f.name}")
    else:
        print("  Aucun dossier de rapports.")

if __name__ == "__main__":
    run_test()
