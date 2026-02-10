# Imports

import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import EMBEDDING_001_GEMINI, MODEL_GEMINI_1_5_FLASH, TEMPLATE_PROMPT
from utils import load_and_extract_content_pdf, get_embeddings_model, get_collection


def create_chunks_and_get_list_documents(documents):
    return CharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(documents)


def create_and_return_embeddings(list_of_documents):
    embeddings_model = get_embeddings_model(EMBEDDING_001_GEMINI)
    text_contents = [doc.page_content for doc in list_of_documents]
    return embeddings_model.embed_documents(text_contents), text_contents, embeddings_model


def create_db_chroma(texts, text_contents, embeddings):
    client, collection = get_collection('naive')
    ids = [str(index) for index in range(len(texts))]
    collection.add(
        embeddings=embeddings,
        documents=text_contents,
        ids=ids
    )
    return client


def create_and_get_retriever(client, embeddings_model, k=5):
    vectorstore = Chroma(
        client=client,
        collection_name="pdf_documents",
        embedding_function=embeddings_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def init_gemini():
    return ChatGoogleGenerativeAI(model=MODEL_GEMINI_1_5_FLASH, temperature=0.2)


def create_chain(retriever):
    llm = init_gemini()
    prompt = ChatPromptTemplate.from_template(TEMPLATE_PROMPT)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def execute_chain(retrieval_chain, ask):
    return retrieval_chain.invoke({"input": ask})


if __name__ == '__main__':
    asks = [
        "Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
        "Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
        "Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
        "Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
        "Quais são os principais aspectos da crítica social e política presentes em \"Os Sertões\"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"
    ]
    _documents = load_and_extract_content_pdf()
    _list_of_documents = create_chunks_and_get_list_documents(documents=_documents)
    _embeddings, _text_contents, _embed_model = create_and_return_embeddings(list_of_documents=_list_of_documents)
    _client = create_db_chroma(texts=_list_of_documents, text_contents=_text_contents, embeddings=_embeddings)
    _retriever = create_and_get_retriever(client=_client, embeddings_model=_embed_model)
    _retrieval_chain = create_chain(retriever=_retriever)

    for _ask in asks:
        _response = execute_chain(retrieval_chain=_retrieval_chain, ask=_ask)
        print(f"Sua pergunta: {_ask}\n")
        print(f"Resposta: {_response['answer']}\n")
