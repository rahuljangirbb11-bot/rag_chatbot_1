from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ==========================================
# 1. LOAD PDF DOCUMENT
# ==========================================

def load_pdf(file_path):

    if not file_path:
        raise ValueError("PDF file path is missing.")

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in the PDF.")

    return documents


# ==========================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# ==========================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP

    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    return chunks