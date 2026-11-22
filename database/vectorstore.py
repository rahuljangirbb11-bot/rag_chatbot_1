import os
import hashlib

from langchain_chroma import Chroma

from config import (
    CHROMA_BASE_DIR,
    COLLECTION_NAME
)

from ingestion import (
    load_pdf,
    split_documents
)

from embeddings import create_embeddings


# ==========================================
# 1. GENERATE PDF HASH
# ==========================================

def generate_file_hash(file_path):

    with open(file_path, "rb") as pdf_file:

        file_hash = hashlib.sha256(
            pdf_file.read()
        ).hexdigest()

    return file_hash


# ==========================================
# 2. CREATE VECTOR DATABASE
# ==========================================

def create_vectorstore(file_path):

    if not os.path.isfile(file_path):

        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    # Generate PDF hash

    file_hash = generate_file_hash(file_path)

    # Create separate directory for each PDF

    persist_directory = os.path.join(
        str(CHROMA_BASE_DIR),
        file_hash
    )

    os.makedirs(
        persist_directory,
        exist_ok=True
    )

    # Load PDF

    documents = load_pdf(file_path)

    # Split documents

    chunks = split_documents(documents)

    # Initialize embeddings

    embeddings = create_embeddings()

    # Check existing database

    database_path = os.path.join(
        persist_directory,
        "chroma.sqlite3"
    )

    if os.path.exists(database_path):

        vectorstore = Chroma(

            collection_name=COLLECTION_NAME,

            embedding_function=embeddings,

            persist_directory=persist_directory

        )

        if vectorstore._collection.count() > 0:

            return (
                vectorstore,
                len(documents),
                len(chunks)
            )

    # Create new vector database

    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        collection_name=COLLECTION_NAME,

        persist_directory=persist_directory

    )

    return (
        vectorstore,
        len(documents),
        len(chunks)
    )