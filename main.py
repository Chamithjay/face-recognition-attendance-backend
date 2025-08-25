from fastapi import FastAPI
from database import SessionLocal

app = FastAPI()


def get_db():
    """
    Dependency to get a SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@app.get("/")
def root():
    return {"message": "Hello, FastAPI is working!"}
