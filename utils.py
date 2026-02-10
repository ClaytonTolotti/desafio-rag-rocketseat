import hashlib

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from constants import DB_VECTOR_OS_SERTOES
from constants import PDF_PATH


def get_collection(type_rag):
    client = chromadb.PersistentClient()
    return client, client.get_or_create_collection(name=f"{DB_VECTOR_OS_SERTOES}_{type_rag}")


def generate_hash_do_pdf(path: PDF_PATH = None):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_pdf_has_embeddings(type_rag):
    hash_pdf = generate_hash_do_pdf()
    _, result = get_collection(type_rag)

    if result.get(where={'document_id': hash_pdf})['ids']:
        print('arquivo processado')
        return result

    return None


def load_and_extract_content_pdf():
    return PyPDFLoader(PDF_PATH).load()


def get_embeddings_model(model=None):
    return GoogleGenerativeAIEmbeddings(model=model)
