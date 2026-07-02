import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Ajouter le répertoire racine au chemin de recherche Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.notifier import LegalNotifier

def run_simulation():
    print("[SIMULATION] Démarrage de la simulation de conflit...")
    base_dir = Path(__file__).parent.parent
    
    # Créer de faux conflits pour simuler l'ingestion
    theme = "TBT"
    country = "Madagascar"
    filename = "Mad182928_test.pdf"
    
    conflicts = [
        {
            "new_node_id": "clause_002_tbt_mad",
            "old_node_id": "art_14_baleine_mad",
            "new_law": "Loi TBT Madagascar 2026",
            "old_law": "Loi Baleines Madagascar 2018",
            "explanation": "La nouvelle réglementation TBT autorise les rejets acoustiques sous-marins industriels à Madagascar alors que la Loi Baleines de 2018 les proscrit formellement dans les eaux territoriales."
        }
    ]
    
    # Simuler le bloc de notification de server.py
    try:
        notifier = LegalNotifier(
            alerts_dir=base_dir / "data" / "alerts",
            reports_dir=base_dir / "data" / "reports",
            company_name="Legal AI Hub"
        )
        
        updates = []
        for idx, cf in enumerate(conflicts):
            updates.append({
                "id": f"conflict-{theme.lower()}-{country.lower()}-{idx}-{int(time.time())}",
                "title": f"Conflit de clauses : {cf.get('new_law', 'Nouvelle loi')} vs {cf.get('old_law', 'Loi existante')}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": f"Importateur automatique ({filename})",
                "url": "#",
                "excerpt": f"Contradiction identifiée entre la clause {cf.get('new_node_id')} et {cf.get('old_node_id')}.\n\nDescription du conflit : {cf.get('explanation')}",
                "topics": [theme.lower(), "conformité", "conflit"],
                "country": country,
                "is_new": True,
                "is_critical": True
            })
        
        print(f"[SIMULATION] Envoi de la notification avec {len(updates)} alerte(s)...")
        notif_res = notifier.notify(updates)
        print(f"[SIMULATION] Notification complétée. Résultats : {notif_res}")
        
        # Vérifier si les fichiers ont bien été créés
        alert_md = Path(notif_res.get("local")) if notif_res.get("local") else None
        report_html = Path(notif_res.get("html")) if notif_res.get("html") else None
        
        print("\n--- VÉRIFICATION DES FICHIERS ---")
        if alert_md and alert_md.exists():
            print(f"[SUCCÈS] Rapport local Markdown créé : {alert_md.name}")
        else:
            print("[ÉCHEC] Rapport local Markdown manquant.")
            
        if report_html and report_html.exists():
            print(f"[SUCCÈS] Rapport HTML de veille créé : {report_html.name}")
        else:
            print("[ÉCHEC] Rapport HTML de veille manquant.")
            
    except Exception as e:
        print(f"[ERREUR] Échec de la simulation : {e}")

if __name__ == "__main__":
    run_simulation()
