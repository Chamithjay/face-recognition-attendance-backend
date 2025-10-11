"""
Student management endpoints for registration, deletion, and retrieval.
"""

from fastapi import APIRouter, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from services.student_service import register_student, delete_student, get_all_students, get_student_by_id
from database import get_db
import os
import traceback

router = APIRouter(prefix="/students", tags=["Students"])


@router.post("/register")
async def register_student_route(
    student_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    video: UploadFile = None,
    db: Session = Depends(get_db)
):
    """
    Register new student with face embeddings from video.
    
    Args:
        student_id: Unique student identifier
        name: Student's full name
        email: Student's email address
        video: Registration video containing student's face
        db: Database session
        
    Returns:
        Success message with student details
    """
    video_path = None
    try:
        print(f"Registering student: {student_id}, {name}, {email}")
        
        if not video:
            raise HTTPException(status_code=400, detail="Video file is required")
        
        video_path = f"temp_{video.filename}"
        print(f"Saving video to: {video_path}")
        
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        print(f"Video saved successfully, size: {len(content)} bytes")
        
        student = register_student(db, student_id, name, email, video_path)
        
        print(f"Student registered successfully: {student.student_id}")
        return {
            "message": "Student registered successfully", 
            "student_id": student.student_id, 
            "id": student.id
        }
        
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to register student: {str(e)}")
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up temporary file: {video_path}")
            except Exception as cleanup_error:
                print(f"Failed to clean up file {video_path}: {cleanup_error}")

@router.delete("/delete/{student_id}")
async def delete_student_route(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete student and all associated face embeddings.
    
    Args:
        student_id: Unique identifier of student to delete
        db: Database session
        
    Returns:
        Success message
    """
    try:
        print(f"Deleting student: {student_id}")
        
        result = delete_student(db, student_id)
        
        print(f"Student deleted successfully: {student_id}")
        return result
        
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete student: {str(e)}")


@router.get("/")
async def get_students_route(db: Session = Depends(get_db)):
    """
    Retrieve list of all registered students.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with students array and count
    """
    try:
        students = get_all_students(db)
        return {
            "students": [
                {
                    "id": student.id,
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email,
                    "created_at": student.created_at
                }
                for student in students
            ],
            "count": len(students)
        }
    except Exception as e:
        print(f"Error getting students: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get students: {str(e)}")


@router.get("/{student_id}")
async def get_student_route(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve specific student details by ID.
    
    Args:
        student_id: Unique identifier of student
        db: Database session
        
    Returns:
        Student details dictionary
    """
    try:
        student = get_student_by_id(db, student_id)
        return {
            "id": student.id,
            "student_id": student.student_id,
            "name": student.name,
            "email": student.email,
            "created_at": student.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error getting student: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get student: {str(e)}")
