from pinecone import Pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized successfully")
except Exception as e:
    print("Failed to initialize Pinecone:", e)
    raise e
