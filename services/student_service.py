"""
Student registration and management service.
Handles video frame extraction, face emb            if isinstance(frame, np.ndarray):
                frame = np.ascontiguousarray(frame)
            
            import uuid
            temp_filename = f"temp_frame_{uuid.uuid4().hex[:8]}_{i}.jpg"
            try:
                success = cv2.imwrite(temp_filename, frame)
                if not success:
                    continue
                
                result = None
                
                try:on, and database operations.
"""

import uuid
import cv2
import numpy as np
import os
from deepface import DeepFace
from sqlalchemy.orm import Session
from models.models import Student
from vector_db import index


MAX_EMBEDDINGS_PER_STUDENT = 20


def extract_frames_from_video(video_path, frame_skip=10):
    """
    Extract frames from video file at specified intervals.
    
    Args:
        video_path: Path to video file
        frame_skip: Extract every Nth frame (default: 10)
        
    Returns:
        List of extracted frames as numpy arrays
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
    Generate face embeddings from video frames using ArcFace model.
    Uses RetinaFace detector with MTCNN fallback.
    
    Args:
        frames: List of video frames as numpy arrays
        
    Returns:
        List of normalized face embeddings
    """
    embeddings = []
    
    for i, frame in enumerate(frames):
        try:
            if isinstance(frame, tuple):
                if len(frame) == 2:
                    ret, actual_frame = frame
                    if ret and isinstance(actual_frame, np.ndarray):
                        frame = actual_frame
                    else:
                        continue
                else:
                    continue
            
            # Ensure frame is numpy array
            if not isinstance(frame, np.ndarray):
                continue
                
            # Ensure frame has correct shape and dtype
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                continue
                
            # Ensure uint8 dtype
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Ensure frame is contiguous in memory for DeepFace
            frame = np.ascontiguousarray(frame)
            
            # Try RetinaFace first, fallback to MTCNN
            import uuid
            temp_filename = f"temp_frame_{uuid.uuid4().hex[:8]}_{i}.jpg"
            try:
                success = cv2.imwrite(temp_filename, frame)
                if not success:
                    continue
                
                result = None
                
                # Try RetinaFace with resized image (known to work better)
                try:
                    h, w = frame.shape[:2]
                    max_dim = 800
                    if max(h, w) > max_dim:
                        if h > w:
                            new_h = max_dim
                            new_w = int(w * (max_dim / h))
                        else:
                            new_w = max_dim
                            new_h = int(h * (max_dim / w))
                        resized_frame = cv2.resize(frame, (new_w, new_h))
                        cv2.imwrite(temp_filename, resized_frame)
                    
                    result = DeepFace.represent(
                        img_path=temp_filename, 
                        model_name="ArcFace", 
                        detector_backend="retinaface", 
                        enforce_detection=False
                    )
                    
                except Exception:
                    cv2.imwrite(temp_filename, frame)
                    result = DeepFace.represent(
                        img_path=temp_filename, 
                        model_name="ArcFace", 
                        detector_backend="mtcnn", 
                        enforce_detection=False
                    )
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as temp_error:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise temp_error
            
            if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
                embeddings.append(np.array(result[0]["embedding"]))
                
        except Exception:
            continue
    
    return embeddings


def save_embeddings_to_pinecone(student_id, embeddings):
    """
    Save face embeddings to Pinecone vector database.
    Normalizes embeddings and limits storage to MAX_EMBEDDINGS_PER_STUDENT.
    
    Args:
        student_id: Unique identifier for the student
        embeddings: List of face embedding vectors
    """
    if index is None or len(embeddings) == 0:
        print("No embeddings or Pinecone not available")
        return

    embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

    if len(embeddings) > MAX_EMBEDDINGS_PER_STUDENT:
        indices = np.linspace(0, len(embeddings)-1, MAX_EMBEDDINGS_PER_STUDENT, dtype=int)
        embeddings = [embeddings[i] for i in indices]

    vectors_to_upsert = []
    for emb in embeddings:
        vector_id = str(uuid.uuid4())
        vectors_to_upsert.append({
            "id": vector_id,
            "values": emb.tolist(),
            "metadata": {"student_id": student_id}
        })

    index.upsert(vectors_to_upsert)
    print(f"Saved {len(embeddings)} embeddings for student {student_id} to Pinecone")


def register_student(db: Session, student_id: str, name: str, email: str, video_path: str):
    """
    Register a new student with face embeddings.
    Extracts frames from video, generates embeddings, and stores in database.
    
    Args:
        db: Database session
        student_id: Unique student identifier
        name: Student's full name
        email: Student's email address
        video_path: Path to registration video file
        
    Returns:
        Created Student object
    """
    try:
        print(f"Starting student registration for: {student_id}")
        
        existing_student = db.query(Student).filter(Student.student_id == student_id).first()
        if existing_student:
            raise ValueError(f"Student with ID '{student_id}' already exists")
        
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
        db.rollback()
        raise e


def delete_student(db: Session, student_id: str):
    """
    Delete student and associated face embeddings from database and Pinecone.
    
    Args:
        db: Database session
        student_id: Unique identifier of student to delete
        
    Returns:
        Success message dictionary
    """
    try:
        print(f"Starting student deletion for: {student_id}")
        
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            raise ValueError(f"Student with ID '{student_id}' not found")
        
        student_db_id = student.student_id
        print(f"Found student: {student.name} (DB ID: {student_db_id})")
        
        if index is not None:
            try:
                print("Deleting embeddings from Pinecone...")
                query_response = index.query(
                    vector=[0] * 512,
                    filter={"student_id": student_db_id},
                    top_k=1000,
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
    Retrieve all registered students from database.
    
    Args:
        db: Database session
        
    Returns:
        List of Student objects
    """
    try:
        students = db.query(Student).all()
        return students
    except Exception as e:
        print(f"Error getting students: {str(e)}")
        raise e


def get_student_by_id(db: Session, student_id: str):
    """
    Retrieve specific student by their unique identifier.
    
    Args:
        db: Database session
        student_id: Unique student identifier
        
    Returns:
        Student object if found
    """
    try:
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            raise ValueError(f"Student with ID '{student_id}' not found")
        return student
    except Exception as e:
        print(f"Error getting student: {str(e)}")
        raise e
