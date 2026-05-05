"""
Configuration du système KG-RAG Multi-Graph Multi-Agent.
Tous les paramètres centralisés ici.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR.parent / "data_old"))
MD_DIR = Path(os.getenv("MD_DIR", BASE_DIR.parent / "md"))
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# ── Catégories de documents ────────────────────────────────────────────
CATEGORIES = {
    "Baleine": DATA_DIR / "Baleine",
    "Oiseaux marins": DATA_DIR / "Oiseaux marins",
    "Rejet hydrocarbure": DATA_DIR / "Rejet hydrocarbure",
}

# ── LLM ────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Embeddings ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# ── ChromaDB ───────────────────────────────────────────────────────────
CHROMA_COLLECTION_ENRICHED = "enriched_articles"
CHROMA_COLLECTION_EXTERNAL = "external_laws"
CHROMA_COLLECTION_DEFINITIONS = "definitions"

# ── Retrieval ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "5"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))

# ── Expansion ──────────────────────────────────────────────────────────
MAX_EXPANSION_DEPTH = int(os.getenv("MAX_EXPANSION_DEPTH", "3"))

# ── Agents ─────────────────────────────────────────────────────────────
MAX_AGENT_PASSES = int(os.getenv("MAX_AGENT_PASSES", "5"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))

# ── Résolution de références ───────────────────────────────────────────
EXTERNAL_SEARCH_THRESHOLD = float(os.getenv("EXTERNAL_SEARCH_THRESHOLD", "0.7"))
NER_CONFIDENCE_THRESHOLD = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.6"))

# ── Logging ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "kg_rag.log"
