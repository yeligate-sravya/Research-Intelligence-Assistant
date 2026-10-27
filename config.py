from __future__ import annotations
 
import os
from dataclasses import dataclass, field
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR))
LOG_DIR = Path(os.getenv("RAG_LOG_DIR", DATA_DIR / "logs"))
EVAL_DIR = Path(os.getenv("RAG_EVAL_DIR", DATA_DIR / "eval_results"))
 
 
def _env(name: str, default, cast=str):
    value = os.getenv(name)
    return cast(value) if value is not None else default
 
 
@dataclass(frozen=True)
class Settings:
    # --- storage -------------------------------------------------------
    db_path: Path = DATA_DIR / "chromaDB"
    chunks_path: Path = DATA_DIR / "output" / "chunks.json"
    collection_name: str = "research_papers"
 
    # --- models --------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3.2:3b"))
    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434")
    )
 
    # --- retrieval -----------------------------------------------------
    candidates: int = 30            # raw vector hits fetched before filtering
    top_k: int = field(default_factory=lambda: _env("RAG_TOP_K", 4, int))
    max_distance: float = field(
        default_factory=lambda: _env("RAG_MAX_DISTANCE", 0.55, float)
    )
    # When a PDF is named in the question the search is already scoped, so
    # the absolute gate is relaxed (this fixed "what does X.pdf say about ...").
    filtered_max_distance: float = 0.90
    # Drop weak tail results: keep chunks within this margin of the best hit.
    relative_margin: float = 0.15
    max_per_source: int = 2         # diversity cap when no PDF is named
    lexical_weight: float = 0.15    # hybrid rerank: bonus for keyword overlap
    max_chars_per_chunk: int = 700  # context budget per passage
 
    # --- generation (Ollama) ------------------------------------------
    keep_alive: str = "30m"
    num_ctx: int = 2048
    num_predict: int = 300
    temperature: float = 0.2
 
    # --- grounding check ----------------------------------------------
    # Cosine similarity between an answer sentence and its best evidence
    # sentence. Calibrate with evaluate.py; 0.5 is a sensible start.
    support_threshold: float = 0.5
 