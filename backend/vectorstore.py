# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FIX: on importe aussi PINECONE_INDEX_NAME et PINECONE_ENVIRONMENT depuis
# config.py au lieu de les re-definir en dur ici (avant: "langgraph-rag-index"
# codee en dur, differente du nom "rag-index" attendu par le README -> le code
# creait/utilisait silencieusement un index Pinecone different de celui que
# l'utilisateur cree a la main)
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Hugging Face embedding model
# This will download the model the first time it's used.
# The default model for HuggingFaceEmbeddings is 'sentence-transformers/all-MiniLM-L6-v2'
# which has a dimension of 384.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name (coherent avec config.py / README: "rag-index")
INDEX_NAME = PINECONE_INDEX_NAME


# --- Retriever (Existing function) ---
def get_retriever(k: int = 5):
    """Initializes and returns the Pinecone vector store retriever.

    FIX: le nombre de resultats (k) est maintenant configure ici via
    `search_kwargs`, au lieu d'etre passe a `.invoke(query, k=5)` dans
    agent.py, ce qui n'est pas l'API supportee par les retrievers LangChain
    (le k doit etre fixe a la creation du retriever, pas a l'invocation).
    """
    # Ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # dimension pour 'sentence-transformers/all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(text_content: str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # Create Langchain Document objects from the raw text
    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
