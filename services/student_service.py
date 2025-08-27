import uuid
import cv2
import numpy as np
from deepface import DeepFace
from sqlalchemy.orm import Session
from models.models import Student
from vector_db import index

def extract_frames_from_video(video_path, frame_skip=10):
    """
    Extract frames from video
    """
    try:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames, frame_count = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        
        print(f"Successfully extracted {len(frames)} frames from {frame_count} total frames")
        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        raise e

def get_face_embeddings(frames):
    """
    Generate embeddings from frames
    """
    embeddings = []
    successful_detections = 0
    
    for i, frame in enumerate(frames):
        try:
            print(f"Processing frame {i+1}/{len(frames)}")
            result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)
            if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
                embeddings.append(np.array(result[0]["embedding"]))
                successful_detections += 1
                print(f"Successfully extracted face embedding from frame {i+1}")
        except Exception as e:
            print(f"Failed to process frame {i+1}: {str(e)}")
            continue
    
    print(f"Successfully processed {successful_detections} faces from {len(frames)} frames")
    return embeddings

def save_embeddings_to_pinecone(student_id, embeddings):
    """
    Save embeddings to Pinecone
    """
    if index is None:
        print("Pinecone not available, skipping embedding storage")
        return
        
    vectors = []
    for emb in embeddings:
        vector_id = str(uuid.uuid4())
        vectors.append({
            "id": vector_id,
            "values": emb.tolist(),
            "metadata": {"student_id": student_id}
        })
    index.upsert(vectors)

def register_student(db: Session, student_id: str, name: str, email: str, video_path: str):
    """
    Register student in PostgreSQL and Pinecone
    """
    try:
        print(f"Starting student registration for: {student_id}")
        
        # Check if student_id already exists
        existing_student = db.query(Student).filter(Student.student_id == student_id).first()
        if existing_student:
            raise ValueError(f"Student with ID '{student_id}' already exists")
        
        # Check if email already exists
        existing_email = db.query(Student).filter(Student.email == email).first()
        if existing_email:
            raise ValueError(f"Student with email '{email}' already exists")
        
        print("Creating student record in database...")
        student = Student(student_id=student_id, name=name, email=email)
        db.add(student)
        db.commit()
        db.refresh(student)
        print(f"Student record created with ID: {student.id}")

        print(f"Extracting frames from video: {video_path}")
        frames = extract_frames_from_video(video_path)
        print(f"Extracted {len(frames)} frames")
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted from the video")
        
        print("Generating face embeddings...")
        embeddings = get_face_embeddings(frames)
        print(f"Generated {len(embeddings)} face embeddings")
        
        if len(embeddings) == 0:
            raise ValueError("No faces could be detected in the video")
        
        print("Saving embeddings to Pinecone...")
        save_embeddings_to_pinecone(student.student_id, embeddings)
        print("Registration completed successfully")

        return student
        
    except Exception as e:
        print(f"Error in register_student: {str(e)}")
        # Rollback database changes if something goes wrong
        db.rollback()
        raise e

def delete_student(db: Session, student_id: str):
    """
    Delete a student and their associated embeddings from both PostgreSQL and Pinecone
    """
    try:
        print(f"Starting student deletion for: {student_id}")
        
        # Find the student
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            raise ValueError(f"Student with ID '{student_id}' not found")
        
        student_db_id = student.id
        print(f"Found student: {student.name} (DB ID: {student_db_id})")
        
        # Delete embeddings from Pinecone
        if index is not None:
            try:
                print("Deleting embeddings from Pinecone...")
                # Query Pinecone to find all vectors for this student
                query_response = index.query(
                    vector=[0] * 128,  # Dummy vector for metadata filtering
                    filter={"student_id": student_db_id},
                    top_k=1000,  # Get all matches
                    include_metadata=True
                )
                
                if query_response.matches:
                    vector_ids = [match.id for match in query_response.matches]
                    index.delete(ids=vector_ids)
                    print(f"Deleted {len(vector_ids)} embeddings from Pinecone")
                else:
                    print("No embeddings found in Pinecone for this student")
                    
            except Exception as e:
                print(f"Warning: Could not delete embeddings from Pinecone: {str(e)}")
        else:
            print("Pinecone not available, skipping embedding deletion")
        
        # Delete student from PostgreSQL
        print("Deleting student from database...")
        db.delete(student)
        db.commit()
        print(f"Student '{student_id}' deleted successfully")
        
        return {"message": f"Student '{student_id}' deleted successfully"}
        
    except Exception as e:
        print(f"Error in delete_student: {str(e)}")
        db.rollback()
        raise e

def get_all_students(db: Session):
    """
    Get all students from the database
    """
    try:
        students = db.query(Student).all()
        return students
    except Exception as e:
        print(f"Error getting students: {str(e)}")
        raise e

def get_student_by_id(db: Session, student_id: str):
    """
    Get a specific student by student_id
    """
    try:
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            raise ValueError(f"Student with ID '{student_id}' not found")
        return student
    except Exception as e:
        print(f"Error getting student: {str(e)}")
        raise e
