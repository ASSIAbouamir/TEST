import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class Settings(BaseModel):
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Groq Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    # OpenAI Configuration (for embeddings only)
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # WhyHow Configuration
    WHYHOW_API_KEY: str = ""
    WHYHOW_API_URL: str = "https://api.whyhow.ai"
    
    # Reducto Configuration
    REDUCTO_API_KEY: str = ""
    
    # Graph Configuration
    LEXICAL_GRAPH_ID: Optional[str] = None
    DEFINITIONS_GRAPH_ID: Optional[str] = None
    
    # Retrieval Configuration
    TOP_K_VECTOR: int = 10
    TOP_K_BM25: int = 10
    TOP_K_KEYWORD: int = 10
    TOP_K_FINAL: int = 5
    
    # Agent Configuration
    MAX_RECURSION_DEPTH: int = 3
    MAX_CONTEXT_TOKENS: int = 4000
    SIMILARITY_THRESHOLD: float = 0.6
    
    # Document Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    @classmethod
    def from_env(cls):
        """Load settings from environment variables"""
        import os

        # Optional dependency: python-dotenv. In restricted environments (like this sandbox)
        # we still want the system to run in offline mode without crashing on import.
        try:
            from dotenv import load_dotenv  # type: ignore

            # Load the package-local .env first, then allow process env overrides.
            load_dotenv(Path(__file__).with_name(".env"), override=True)
            load_dotenv(override=True)
        except ModuleNotFoundError:
            # Best-effort minimal .env loader (KEY=VALUE, ignores comments/blank lines).
            env_path = Path(__file__).with_name(".env")
            if env_path.exists():
                try:
                    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ.setdefault(key, value)
                except Exception:
                    # If parsing fails, just proceed with process env.
                    pass
        
        return cls(
            OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY", ""),
            OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            OPENROUTER_BASE_URL=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
            GROQ_MODEL=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            OPENAI_EMBEDDING_MODEL=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            WHYHOW_API_KEY=os.getenv("WHYHOW_API_KEY", ""),
            WHYHOW_API_URL=os.getenv("WHYHOW_API_URL", "https://api.whyhow.ai"),
            REDUCTO_API_KEY=os.getenv("REDUCTO_API_KEY", ""),
            LEXICAL_GRAPH_ID=os.getenv("LEXICAL_GRAPH_ID"),
            DEFINITIONS_GRAPH_ID=os.getenv("DEFINITIONS_GRAPH_ID"),
            TOP_K_VECTOR=int(os.getenv("TOP_K_VECTOR", "10")),
            TOP_K_BM25=int(os.getenv("TOP_K_BM25", "10")),
            TOP_K_KEYWORD=int(os.getenv("TOP_K_KEYWORD", "10")),
            TOP_K_FINAL=int(os.getenv("TOP_K_FINAL", "5")),
            MAX_RECURSION_DEPTH=int(os.getenv("MAX_RECURSION_DEPTH", "5")),
            MAX_CONTEXT_TOKENS=int(os.getenv("MAX_CONTEXT_TOKENS", "8000")),
            SIMILARITY_THRESHOLD=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "512")),
            CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "50"))
        )

settings = Settings.from_env()
