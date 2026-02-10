# Imports

import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import EMBEDDING_001_GEMINI
from utils import load_and_extract_content_pdf, get_embeddings_model


def create_chroma_db():
    return get_client_chromadb().get_or_create_collection(name="pdf_parent_documents")


def get_client_chromadb():
    return chromadb.PersistentClient()


def get_retriever():
    embeddings_model = get_embeddings_model(EMBEDDING_001_GEMINI)
    return Chroma(
        client=get_client_chromadb(),
        collection_name="pdf_parent_documents",
        embedding_function=embeddings_model
    )


def create_parent_and_child_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    return parent_splitter, child_splitter


def create_retriever():
    store = InMemoryStore()
    parent_splitter, child_splitter = create_parent_and_child_splitters()
    retriever = ParentDocumentRetriever(
        vectorstore=get_retriever(),
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    return retriever


def add_documents_in_retriever(retriever, documents):
    retriever.add_documents(documents=documents)


def get_relevant_documents_from_query(retriever, ask):
    return retriever.invoke(ask)


def create_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = ChatPromptTemplate.from_template("""
        Você é um bibliotecário. Responda as perguntas baseadas no contexto fornecido.
                                              
        Context: {context}
                                              
        Pergunta: {input}
    """)
    return create_stuff_documents_chain(llm, prompt)


if __name__ == '__main__':

    asks = [
        "Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
        "Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
        "Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
        "Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
        "Quais são os principais aspectos da crítica social e política presentes em \"Os Sertões\"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"
    ]
    _documents = load_and_extract_content_pdf()
    _retriever = create_retriever()
    add_documents_in_retriever(retriever=_retriever, documents=_documents)
    for _ask in asks:
        _retrieved_docs = get_relevant_documents_from_query(retriever=_retriever, ask=_ask)
        _chain = create_chain()
        response = _chain.invoke({"input": _ask, "context": _retrieved_docs})

        print(f"Pergunta: {_ask}\nResposta: {response}\n\n")
