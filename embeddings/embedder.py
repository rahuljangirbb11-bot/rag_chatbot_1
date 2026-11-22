from langchain_openai import OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)


# ==========================================
# CREATE OPENAI EMBEDDING MODEL
# ==========================================

def create_embeddings():

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing.")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    return embeddings