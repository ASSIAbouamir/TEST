import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au chemin de recherche Python
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.notifier import LegalNotifier

def test_smtp_connection():
    print("[SMTP TEST] Démarrage du test SMTP...")
    base_dir = Path(__file__).parent.parent
    
    try:
        notifier = LegalNotifier(
            alerts_dir=base_dir / "data" / "alerts",
            reports_dir=base_dir / "data" / "reports",
            company_name="Legal AI Hub"
        )
        
        print(f"SMTP Host: {notifier.email_cfg.get('smtp_host')}")
        print(f"SMTP Port: {notifier.email_cfg.get('smtp_port')}")
        print(f"Expéditeur (From): {notifier.email_cfg.get('from_addr')}")
        print(f"Destinataires (To): {notifier.email_cfg.get('to')}")
        
        print("\n[SMTP TEST] Tentative d'envoi de l'e-mail de test...")
        success = notifier.test_email()
        
        if success:
            print("\n[SUCCÈS] L'e-mail de test a été envoyé avec succès ! Vérifiez votre boîte de réception.")
        else:
            print("\n[ÉCHEC] Échec de l'envoi de l'e-mail de test. Vérifiez les paramètres SMTP et la connexion réseau.")
            
    except Exception as e:
        print(f"\n[ERREUR] Une erreur est survenue lors du test SMTP : {e}")

if __name__ == "__main__":
    test_smtp_connection()
