import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import engine, Base
from vector_db import pc  
from routers.student_router import router as student_router
from routers.attendance_router import router as attendance_router
from routers.stream_router import router as stream_router
from services.session_service import cleanup_sessions_periodically
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    try:
        yield
    finally:
        # Cancel cleanup task on shutdown
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)

origins=["http://localhost:5173"]

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
app.include_router(stream_router)



@app.get("/")
def root():
    return {"message": "Face Recognition Attendance System API is running!"}
