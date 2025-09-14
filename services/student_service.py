import uuid
import cv2
import numpy as np
import os
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
    Generate embeddings from frames using retinaface detector
    """
    embeddings = []
    successful_detections = 0
    
    for i, frame in enumerate(frames):
        try:
            # Handle tuple frames from cv2.VideoCapture.read() which returns (ret, frame)
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
                    # Fallback to MTCNN
                    cv2.imwrite(temp_filename, frame)  # Use original frame
                    result = DeepFace.represent(
                        img_path=temp_filename, 
                        model_name="ArcFace", 
                        detector_backend="mtcnn", 
                        enforce_detection=False
                    )
                
                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as temp_error:
                # Clean up temp file if it exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise temp_error
            
            if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
                embeddings.append(np.array(result[0]["embedding"]))
                successful_detections += 1
                
        except Exception:
            continue
    
    return embeddings

def save_embeddings_to_pinecone(student_id, embeddings):
    """
    Save average embedding to Pinecone
    """
    if index is None:
        print("Pinecone not available, skipping embedding storage")
        return
    
    if len(embeddings) == 0:
        print("No embeddings to save")
        return

    average_embedding = np.mean(embeddings, axis=0)
    print(f"Calculated average embedding from {len(embeddings)} individual embeddings")

    vector_id = str(uuid.uuid4())
    vector = {
        "id": vector_id,
        "values": average_embedding.tolist(),
        "metadata": {"student_id": student_id}
    }
    
    index.upsert([vector])
    print(f"Saved average embedding for student {student_id} to Pinecone")

def register_student(db: Session, student_id: str, name: str, email: str, video_path: str):
    """
    Register student in PostgreSQL and Pinecone
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
    Delete a student and their associated embeddings from both PostgreSQL and Pinecone
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
