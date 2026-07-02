import os
from pathlib import Path
from dotenv import load_dotenv

# Dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Charger les variables d'environnement
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# LlamaParse
# ---------------------------------------------------------------------------
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", os.getenv("LLAMA_PARSE_API_KEY", ""))

# ---------------------------------------------------------------------------
# Chemins des dossiers
# ---------------------------------------------------------------------------
INPUT_DIR          = BASE_DIR / os.getenv("INPUT_DIR", "data/input")
PARSED_DIR         = BASE_DIR / os.getenv("PARSED_DIR", "data/parsed")
ARCHIVE_DIR        = BASE_DIR / os.getenv("ARCHIVE_DIR", "data/archive")
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
STRUCTURED_DIR     = BASE_DIR / "data" / "structured"   # JSON structurés (clauses) issus du LLM
ALERTS_DIR         = BASE_DIR / "data" / "alerts"
REPORTS_DIR        = BASE_DIR / "data" / "reports"

# ---------------------------------------------------------------------------
# Paramètres de parsing
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "fr")  # Français par défaut

# ---------------------------------------------------------------------------
# Groq LLM (extraction structurée)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Paramètres de surveillance Internet
# ---------------------------------------------------------------------------
COMPANY_NAME             = os.getenv("COMPANY_NAME", "Votre Entreprise")
MONITOR_INTERVAL_HOURS   = float(os.getenv("MONITOR_INTERVAL_HOURS", "6"))

# ---------------------------------------------------------------------------
# Configuration Email / SMTP
# ---------------------------------------------------------------------------
ALERT_EMAIL_TO   = os.getenv("ALERT_EMAIL_TO", "")    # ex: rh@entreprise.com,dg@entreprise.com
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "legal-ai-bot@automatisation.local")
SMTP_HOST        = os.getenv("SMTP_HOST", "")
SMTP_PORT        = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER        = os.getenv("SMTP_USER", "")
SMTP_PASSWORD    = os.getenv("SMTP_PASSWORD", "")
ALERT_MIN_TOPICS = int(os.getenv("ALERT_MIN_TOPICS", "1"))


def ensure_dirs():
    """S'assure que tous les dossiers nécessaires existent."""
    for d in [INPUT_DIR, PARSED_DIR, ARCHIVE_DIR, DATA_PROCESSED_DIR, STRUCTURED_DIR, ALERTS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# S'assurer que les dossiers existent dès le chargement de la config
ensure_dirs()
