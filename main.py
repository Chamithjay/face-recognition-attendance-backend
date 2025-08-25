from fastapi import FastAPI
from database import engine, Base
from vector_db import pc  # Ensure Pinecone client is initialized

app = FastAPI()

Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")

@app.get("/")
def root():
    return {"message": "Face Recognition Attendance System API is running!"}
