
"""
Main FastAPI application entry point.
Configures CORS, initializes database, and registers route handlers.
"""

from fastapi import FastAPI
from database import engine, Base
from routers.student_router import router as student_router
from routers.attendance_router import router as attendance_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Face Recognition Attendance System")

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")

app.include_router(student_router)
app.include_router(attendance_router)


@app.get("/")
def root():
    """API health check endpoint."""
    return {"message": "Face Recognition Attendance System API is running!"}
