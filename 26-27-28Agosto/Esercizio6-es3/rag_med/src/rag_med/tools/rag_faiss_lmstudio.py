"""Utilities to build a minimal RAG pipeline backed by FAISS and Azure OpenAI.

This module contains utilities to create embeddings, load/build a FAISS index,
assemble a retrieval-augmented generation (RAG) chain, and execute queries.
Where possible, functions include small doctest examples. Environment variables
are used to configure Azure OpenAI credentials and deployments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader

from openai import AzureOpenAI

import faiss
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    """Runtime configuration for the RAG components.

    Attributes:
        persist_dir (str): Directory where FAISS artifacts are saved.
        chunk_size (int): Maximum characters per text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        search_type (str): Retrieval mode, "mmr" or "similarity".
        k (int): Number of results returned by the retriever.
        fetch_k (int): Candidate pool size for MMR.
        mmr_lambda (float): Trade-off between relevance and diversity in MMR.
        hf_model_name (str): Default HF embedding model (not used with Azure).
        lmstudio_model_env (str): Env var name that holds the chat model deployment.
    """
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 300
    # Retriever (MMR)
    search_type: str = "similarity"        # "mmr" o "similarity"
    k: int = 1                      # risultati finali
    fetch_k: int = 1              # candidati iniziali (per MMR)
    mmr_lambda: float = 1         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # nome del modello in LM Studio, via env var



SETTINGS = Settings()


# =========================
# Componenti di base
# =========================

def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings client from environment variables.

    Args:
        settings (Settings): Runtime configuration (unused but kept for symmetry).

    Returns:
        AzureOpenAIEmbeddings: Configured embeddings client.

    Raises:
        RuntimeError: If required environment variables are missing.
    """

    return AzureOpenAIEmbeddings(
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        model=os.getenv("EMBEDDING_DEPLOYMENT"),
    )


def get_llm_from_lmstudio(settings: Settings):
    """Initialize an Azure OpenAI chat model from environment variables.

    Args:
        settings (Settings): Runtime configuration.

    Returns:
        AzureChatOpenAI: Configured chat LLM.

    Raises:
        RuntimeError: If required environment variables are not set.
    """
    base_url = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    model_name = os.getenv("CHAT_DEPLOYMENT")

    if not base_url or not api_key:
        raise RuntimeError(
            "OPENAI_BASE_URL e OPENAI_API_KEY devono essere impostate per LM Studio."
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile {settings.lmstudio_model_env} con il nome del modello caricato in LM Studio."
        )

    # model_provider="openai" perché l'endpoint è OpenAI-compatible
    return AzureChatOpenAI(
        model=os.getenv("CHAT_DEPLOYMENT"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


def simulate_corpus() -> List[Document]:
    """Create a small English corpus with metadata and `source` for citations.

    Returns:
        List[Document]: A list of toy documents used for examples.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval.

    Args:
        docs (List[Document]): The documents to split.
        settings (Settings): Chunking configuration.

    Returns:
        List[Document]: The resulting chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAI, persist_dir: str) -> FAISS:
    """Build a FAISS vector store from chunks and persist it.

    Args:
        chunks (List[Document]): Pre-split documents.
        embeddings (AzureOpenAIEmbeddings): Embedding model.
        persist_dir (str): Directory to store FAISS artifacts.

    Returns:
        FAISS: The created vector store.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings: AzureOpenAI, docs: List[Document]) -> FAISS:
    """Load a persisted FAISS index or build it from documents if missing.

    Args:
        settings (Settings): Configuration including persistence directory.
        embeddings (AzureOpenAIEmbeddings): Embedding model.
        docs (List[Document]): Source documents.

    Returns:
        FAISS: Loaded or newly built vector store.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """Configure a retriever in MMR or pure similarity mode.

    Args:
        vector_store (FAISS): The vector store backing the retriever.
        settings (Settings): Retrieval configuration.

    Returns:
        Any: A retriever compatible with LangChain invoke interface.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format retrieved docs into a string with [source:...] citations.

    Args:
        docs (List[Document]): Retrieved documents to include in the prompt.

    Returns:
        str: Formatted context string.
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm, retriever):
    """Build the RAG chain: retrieval -> prompt -> LLM -> string output.

    Args:
        llm: The chat model to generate answers.
        retriever: The retriever providing relevant context for the prompt.

    Returns:
        Runnable: A chain that maps a question string to an answer string.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi SOLO con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'"
         "4) Non contraddire assolutamente il CONTENUTO fornito nel contesto.")
    ])

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """Execute the RAG chain for a single question.

    Args:
        question (str): Natural language question.
        chain: A chain created by `build_rag_chain`.

    Returns:
        str: The generated answer.

    Examples:
        >>> def _fake_chain(q):
        ...     return f"echo: {q}"
        >>> rag_answer("test", _fake_chain)
        'echo: test'
    """
    return chain.invoke(question)


def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the text of the top-k retrieved chunks used as context.

    Args:
        retriever: Retriever with an `invoke` method that returns Documents.
        question (str): The user query.
        k (int): Number of contexts to return.

    Returns:
        List[str]: The page contents of the top-k retrieved documents.
    """
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """Run RAG for each question and return a dataset suitable for Ragas.

    Args:
        questions (List[str]): Questions to evaluate.
        retriever: Retriever used to obtain contexts.
        chain: Chain used to produce answers.
        k (int): Number of contexts per question.
        ground_truth (dict[str, str] | None): Optional references keyed by question.

    Returns:
        list[dict]: Each row contains user_input, retrieved_contexts, response, and optional reference.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


# =========================
# Esecuzione dimostrativa
# =========================

def setup():
    """Create and return a ready-to-use RAG chain using default settings.

    Returns:
        Any: A chain that accepts a question and returns an answer string.

    Examples:
        >>> chain = setup()  # doctest: +SKIP
        >>> isinstance(chain, object)  # doctest: +SKIP
        True
    """
    settings = SETTINGS

    # 1) Componenti
    embeddings = get_embeddings(settings)
    llm = get_llm_from_lmstudio(settings)

    # 2) Dati simulati e indicizzazione (load or build)
    docs = simulate_corpus()
    # loader = DirectoryLoader("db", glob="**/*.md")
    # docs = loader.load()
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    return chain

    # ans = rag_answer(q, chain)
