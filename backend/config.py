# rag_agent_app/backend/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# FIX: nom de modele externalise dans une variable pour pouvoir le changer
# facilement si Groq retire/renomme un modele (a verifier sur
# https://console.groq.com/docs/models)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths (adjust as needed)
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR", "data")
