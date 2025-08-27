from pinecone import Pinecone, ServerlessSpec
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


pc = None
index = None

try:

    pc = Pinecone(api_key=PINECONE_API_KEY)
    

    existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=512,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    

    index = pc.Index(INDEX_NAME)
    print("Pinecone client initialized successfully")
    
except Exception as e:
    print("Failed to initialize Pinecone:", e)
    print("The application will continue without vector database functionality")
    pc = None
    index = None
