"""
main.py
=======
Point d'entrée principal du système Legal AI Automation.

Commandes disponibles :
  python main.py                    → Parse initial + surveillance continue de data/input/
  python main.py --once             → Parse en une seule passe
  python main.py --watch            → Surveillance continue de data/input/
  python main.py --monitor          → Une passe de veille Internet
  python main.py --monitor --notify → Veille Internet + alertes (email, rapport HTML/MD)
  python main.py --monitor --rag    → Veille Internet + import automatique dans le RAG
  python main.py --schedule         → Veille Internet continue (toutes les N heures)
  python main.py --test-email       → Teste la configuration email SMTP
"""

import argparse
import io
import sys
import logging
from pathlib import Path

from src.config import INPUT_DIR, LLAMA_CLOUD_API_KEY, BASE_DIR

# ---------------------------------------------------------------------------
# Configuration globale du système de log
# Force UTF-8 sur le StreamHandler pour éviter UnicodeEncodeError (cp1252)
# ---------------------------------------------------------------------------
_utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(_utf8_stdout),
        logging.FileHandler(BASE_DIR / "legal_ai.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("LegalAI-Automation")


# ---------------------------------------------------------------------------
# Parsing des documents locaux (data/input/)
# ---------------------------------------------------------------------------

def parse_all_existing():
    """Recherche et traite tous les documents déjà présents dans data/input/"""
    from src.parser import LegalDocumentParser

    if not LLAMA_CLOUD_API_KEY:
        logger.warning(
            "La clé API LLAMA_CLOUD_API_KEY n'est pas configurée dans votre fichier .env. "
            "Seuls les fichiers pré-structurés au format .json pourront être traités localement."
        )

    logger.info(f"Recherche de documents existants dans le dossier : {INPUT_DIR.resolve()}")

    files_to_parse = []
    for ext in ["*.pdf", "*.docx", "*.txt", "*.json"]:
        files_to_parse.extend(list(INPUT_DIR.glob(ext)))

    if not files_to_parse:
        logger.info("Aucun document existant trouvé dans data/input/. Le dossier est vide.")
        return

    logger.info(f"Trouvé {len(files_to_parse)} documents à traiter...")

    try:
        parser = LegalDocumentParser()
    except Exception as e:
        logger.error(f"Erreur d'initialisation du parseur : {e}")
        sys.exit(1)

    success_count = 0
    for file_path in files_to_parse:
        try:
            parser.parse_file(file_path)
            success_count += 1
        except Exception as e:
            logger.error(f"Erreur lors du parsing de {file_path.name} : {e}")

    logger.info(
        f"Traitement initial terminé. Documents importés avec succès : {success_count}/{len(files_to_parse)}"
    )


# ---------------------------------------------------------------------------
# Veille Internet
# ---------------------------------------------------------------------------

def run_monitor(notify: bool = False, rag: bool = False):
    """Exécute une passe unique de veille juridique Internet."""
    from src.scheduler import run_monitoring_cycle

    logger.info("=" * 60)
    logger.info("  Legal AI -- Veille Juridique Internet")
    logger.info("=" * 60)

    result = run_monitoring_cycle(
        base_dir=BASE_DIR,
        notify=notify,
        only_critical=False,
        import_to_rag=rag,
    )

    # Résumé final
    logger.info("")
    logger.info("[RESUME] CYCLE DE VEILLE")
    logger.info(f"  Sources analysees     : {result['sources_checked']}")
    logger.info(f"  Mises a jour totales  : {result['total_updates']}")
    logger.info(f"  Alertes critiques     : {result['critical_updates']}")
    if rag:
        logger.info(f"  Documents -> RAG      : {result['rag_imports']}")
    if result.get("notifications", {}).get("html"):
        logger.info(f"  Rapport HTML          : {result['notifications']['html']}")
    if result.get("notifications", {}).get("local"):
        logger.info(f"  Rapport Markdown      : {result['notifications']['local']}")
    logger.info("")


def run_schedule(notify: bool = True, rag: bool = False):
    """Lance la surveillance continue (boucle infinie)."""
    from src.scheduler import run_continuous_monitoring

    logger.info("═" * 60)
    logger.info("  ⚖️  Legal AI — Surveillance Continue Activée")
    logger.info("═" * 60)

    run_continuous_monitoring(
        base_dir=BASE_DIR,
        notify=notify,
        only_critical=False,
        import_to_rag=rag,
    )


def test_email_config():
    """Envoie un email de test pour valider la configuration SMTP."""
    import os
    from src.notifier import LegalNotifier
    from src.config import BASE_DIR

    company = os.getenv("COMPANY_NAME", "Votre Entreprise")
    notifier = LegalNotifier(
        alerts_dir=BASE_DIR / "data" / "alerts",
        reports_dir=BASE_DIR / "data" / "reports",
        company_name=company,
    )
    success = notifier.test_email()
    if success:
        logger.info("✅ Email de test envoyé avec succès !")
    else:
        logger.error(
            "❌ Échec de l'envoi. Vérifiez SMTP_HOST, SMTP_USER, SMTP_PASSWORD, "
            "ALERT_EMAIL_TO dans votre .env"
        )


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Legal AI — Automatisation et veille juridique d'entreprise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                      Parse data/input/ + surveillance continue
  python main.py --once               Parse en une seule passe
  python main.py --watch              Surveillance de data/input/ en temps réel
  python main.py --monitor            Veille Internet (une passe, sans alerte)
  python main.py --monitor --notify   Veille Internet + alertes email/HTML/MD
  python main.py --monitor --rag      Veille Internet + import dans le RAG
  python main.py --schedule           Veille Internet continue (toutes les N heures)
  python main.py --schedule --rag     Veille + import RAG en continu
  python main.py --test-email         Test de la configuration email SMTP
        """,
    )

    # Modes de traitement local
    parser.add_argument(
        "--once",
        action="store_true",
        help="Traite en une seule passe les fichiers de data/input/",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Surveillance continue de data/input/ sans traitement initial",
    )

    # Modes de veille Internet
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Exécute une passe de veille juridique Internet",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Lance la surveillance continue Internet (toutes les N heures)",
    )

    # Options communes
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Envoie les alertes (email + rapport HTML + Markdown)",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Importe automatiquement les mises à jour critiques dans le RAG",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        dest="test_email",
        help="Envoie un email de test pour vérifier la configuration SMTP",
    )

    args = parser.parse_args()

    # --- Routing ---
    if args.test_email:
        test_email_config()

    elif args.schedule:
        logger.info("--- Lancement de la surveillance continue Internet ---")
        run_schedule(notify=args.notify or True, rag=args.rag)

    elif args.monitor:
        logger.info("--- Exécution d'une passe de veille juridique Internet ---")
        run_monitor(notify=args.notify, rag=args.rag)

    elif args.once:
        logger.info("--- Exécution du traitement en une passe (mode --once) ---")
        parse_all_existing()

    elif args.watch:
        logger.info("--- Lancement de la surveillance continue de data/input/ ---")
        from src.watcher import start_watching
        start_watching()

    else:
        # Mode par défaut : parse initial + surveillance continue de data/input/
        logger.info("--- Démarrage de Legal AI Automation Pipeline ---")
        parse_all_existing()
        from src.watcher import start_watching
        start_watching()


if __name__ == "__main__":
    main()
