import os
from pathlib import Path

from dotenv import load_dotenv


# =====================================
# PROJECT DIRECTORY
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================

ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=False
)


# =====================================
# OPENAI CONFIGURATION
# =====================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =====================================
# MODEL CONFIGURATION
# =====================================

LLM_MODEL = "gpt-4.1"

EMBEDDING_MODEL = "text-embedding-3-small"


# =====================================
# VECTOR DATABASE CONFIGURATION
# =====================================

CHROMA_BASE_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pdf_rag_collection"


# =====================================
# TEXT SPLITTING CONFIGURATION
# =====================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# =====================================
# RETRIEVAL CONFIGURATION
# =====================================

RETRIEVER_K = 4    