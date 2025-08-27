from fastapi import FastAPI
from database import engine, Base
from vector_db import pc  
from routers.student_router import router as student_router
from routers.attendance_router import router as attendance_router


app = FastAPI()

Base.metadata.create_all(bind=engine)
print("Database tables created successfully!")

app.include_router(student_router)
app.include_router(attendance_router)



@app.get("/")
def root():
    return {"message": "Face Recognition Attendance System API is running!"}
