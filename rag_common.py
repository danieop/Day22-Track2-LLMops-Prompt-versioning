from __future__ import annotations

from typing import Iterable

from config import get_embeddings, get_int_env, get_llm, load_knowledge_base

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


SYSTEM_V1 = (
    "You are a careful VinFast support assistant. Answer using ONLY the provided context. "
    "Keep the answer concise, factual, and no longer than 4 sentences. "
    "If the context does not contain the answer, say you do not have enough information.\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are a precise VinFast documentation analyst. Use ONLY the provided context.\n\n"
    "Instructions:\n"
    "1. Identify the relevant warranty or specification facts.\n"
    "2. Answer directly with numbers, units, and conditions when present.\n"
    "3. Mention whether a limit is based on time, distance, or whichever comes first.\n"
    "4. If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_V1),
        ("human", "{question}"),
    ]
)

PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_V2),
        ("human", "{question}"),
    ]
)

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


def format_docs(docs: Iterable) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore() -> FAISS:
    text = load_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_int_env("CHUNK_SIZE", 700),
        chunk_overlap=get_int_env("CHUNK_OVERLAP", 100),
    )
    chunks = splitter.split_text(text)
    print(f"Split knowledge base into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, get_embeddings())


def build_rag_chain(vectorstore: FAISS, prompt=PROMPT_V1):
    retriever = vectorstore.as_retriever(search_kwargs={"k": get_int_env("RETRIEVER_K", 3)})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain, retriever


def answer_with_context(retriever, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | get_llm() | StrOutputParser()).invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return {"answer": answer, "contexts": contexts}
