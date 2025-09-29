import numpy as np
import cv2
import os
import uuid
from deepface import DeepFace
from vector_db import index
from sqlalchemy.orm import Session
from models.models import Student


def detect_faces_with_bboxes(frame, use_mediapipe=True):
    """
    Detect faces in a frame and return bounding boxes with embeddings.
    
    Args:
        frame: Input frame as numpy array
        use_mediapipe: Whether to use MediaPipe detector (default: True)
    
    Returns:
        List of dicts containing:
        - bbox: [x, y, width, height] bounding box coordinates
        - embedding: Face embedding vector
        - confidence: Detection confidence (if available)
    """
    faces_data = []
    
    # Ensure frame is contiguous in memory
    frame = np.ascontiguousarray(frame)
    
    # Use temporary file approach for DeepFace
    temp_filename = f"temp_detection_{uuid.uuid4().hex[:8]}.jpg"
    
    try:
        success = cv2.imwrite(temp_filename, frame)
        if not success:
            return faces_data
        
        # Try MediaPipe first, then MTCNN fallback
        detector_backend = "mediapipe" if use_mediapipe else "mtcnn"
        
        try:
            result = DeepFace.represent(
                img_path=temp_filename,
                model_name="ArcFace",
                detector_backend=detector_backend,
                enforce_detection=False
            )
        except Exception:
            # Fallback to MTCNN if MediaPipe fails
            result = DeepFace.represent(
                img_path=temp_filename,
                model_name="ArcFace", 
                detector_backend="mtcnn",
                enforce_detection=False
            )
        
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        if isinstance(result, list) and len(result) > 0:
            for face_data in result:
                if "embedding" in face_data and "facial_area" in face_data:
                    facial_area = face_data["facial_area"]
                    
                    # Extract bounding box coordinates
                    # DeepFace returns: {"x": x, "y": y, "w": width, "h": height}
                    bbox = [
                        int(facial_area["x"]),
                        int(facial_area["y"]),
                        int(facial_area["w"]),
                        int(facial_area["h"])
                    ]
                    
                    faces_data.append({
                        "bbox": bbox,
                        "embedding": np.array(face_data["embedding"]),
                        "confidence": face_data.get("confidence", 1.0)
                    })
    
    except Exception as e:
        print(f"Face detection error: {e}")
        # Clean up temp file on error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return faces_data


def process_frame_with_tracking(frame, db: Session, threshold=0.5):
    """
    Process a single frame to detect faces with bounding boxes and recognition.
    
    Args:
        frame: Input frame as numpy array
        db: Database session
        threshold: Recognition threshold
    
    Returns:
        dict containing:
        - detections: List of face detections with bboxes
        - recognitions: List of recognized students
    """
    detections = []
    recognitions = []
    
    try:
        # Detect faces with bounding boxes
        faces_data = detect_faces_with_bboxes(frame)
        
        if not faces_data:
            return {"detections": detections, "recognitions": recognitions}
        
        # Process each detected face
        for face_data in faces_data:
            bbox = face_data["bbox"]
            embedding = face_data["embedding"]
            confidence = face_data["confidence"]
            
            # Add detection info
            detection = {
                "bbox": bbox,
                "confidence": confidence
            }
            
            # Try to recognize the face
            if index is not None:
                try:
                    search_result = index.query(
                        vector=embedding.tolist(),
                        top_k=3,
                        include_metadata=True
                    )
                    
                    if search_result and len(search_result.matches) > 0:
                        best_match = search_result.matches[0]
                        if best_match.score >= threshold:
                            student_id = best_match.metadata["student_id"]
                            student = db.query(Student).filter(Student.student_id == student_id).first()
                            
                            if student:
                                recognition = {
                                    "id": student.id,
                                    "name": student.name,
                                    "student_id": student.student_id,
                                    "confidence": best_match.score,
                                    "bbox": bbox
                                }
                                recognitions.append(recognition)
                                detection["recognized"] = True
                                detection["student_name"] = student.name
                                detection["student_id"] = student.student_id
                            else:
                                detection["recognized"] = False
                        else:
                            detection["recognized"] = False
                    else:
                        detection["recognized"] = False
                except Exception as e:
                    print(f"Recognition error: {e}")
                    detection["recognized"] = False
            else:
                detection["recognized"] = False
            
            detections.append(detection)
    
    except Exception as e:
        print(f"Frame processing error: {e}")
    
    return {"detections": detections, "recognitions": recognitions}


def recognize_students_from_frames(frames, db: Session, threshold=0.5):
    """
    Recognize students from video frames using face embeddings and Pinecone vector search.
    
    Args:
        frames: List of video frames as numpy arrays
        db: Database session
        threshold: Minimum similarity score for recognition (default: 0.5)
    
    Returns:
        List of dictionaries containing recognized student information
    """
    recognized_students = set()
    all_embeddings = []
    total_faces_detected = 0

    print(f"Starting recognition for {len(frames)} frames with threshold {threshold}")

    for i, frame in enumerate(frames):
        try:
            print(f"Processing frame {i+1}/{len(frames)}")
            
            # Handle tuple frames from cv2.VideoCapture.read() which returns (ret, frame)
            if isinstance(frame, tuple):
                if len(frame) == 2:
                    ret, actual_frame = frame
                    if ret and isinstance(actual_frame, np.ndarray):
                        frame = actual_frame
                    else:
                        print(f"Frame {i+1} read failed or invalid frame")
                        continue
                else:
                    print(f"Frame {i+1} is unexpected tuple format: {len(frame)} elements")
                    continue
            
            # Ensure we have a numpy array
            if not isinstance(frame, np.ndarray):
                print(f"Frame {i+1} is not numpy array: {type(frame)}")
                continue
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Frame {i+1} has invalid shape: {frame.shape}")
                continue
                
            # Ensure uint8 dtype
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Ensure frame is contiguous in memory
            frame = np.ascontiguousarray(frame)
            
            # Use RetinaFace first, then MTCNN fallback
            import uuid
            temp_filename = f"temp_recognition_frame_{uuid.uuid4().hex[:8]}_{i}.jpg"
            try:
                success = cv2.imwrite(temp_filename, frame)
                
                if not success:
                    continue
                
                result = None
                
                # Use MediaPipe for real-time face detection (faster than RetinaFace)
                try:
                    result = DeepFace.represent(
                        img_path=temp_filename, 
                        model_name="ArcFace", 
                        detector_backend="mediapipe", 
                        enforce_detection=False
                    )
                    
                except Exception:
                    # Fallback to MTCNN if MediaPipe fails
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
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise temp_error
            
            if isinstance(result, list) and len(result) > 0:
                # Process ALL faces detected in the frame
                for face_result in result:
                    if "embedding" in face_result:
                        embedding = np.array(face_result["embedding"])
                        all_embeddings.append(embedding)
                        total_faces_detected += 1
                
                print(f"Frame {i+1}: detected {len(result)} faces, collected {len(result)} embeddings")
            else:
                print(f"No face detected in frame {i+1}")
                
        except Exception as e:
            print(f"Frame {i+1} failed: {e}")
            continue

    print(f"Collected {len(all_embeddings)} face embeddings from {total_faces_detected} detections")

    if len(all_embeddings) > 0:
        print("Analyzing individual face embeddings...")
        print(f"Total face embeddings collected: {len(all_embeddings)}")

        if index is None:
            print("Pinecone index not available")
            return []
        
        # Instead of averaging, analyze each embedding individually for better accuracy
        student_scores = {}  # student_id -> list of scores
        
        for idx, embedding in enumerate(all_embeddings):
            try:
                search_result = index.query(
                    vector=embedding.tolist(),
                    top_k=3,  # Get top 3 matches per face
                    include_metadata=True
                )

                if search_result and len(search_result.matches) > 0:
                    best_match = search_result.matches[0]
                    if best_match.score >= threshold:
                        student_id = best_match.metadata["student_id"]
                        if student_id not in student_scores:
                            student_scores[student_id] = []
                        student_scores[student_id].append(best_match.score)
                        
            except Exception as e:
                print(f"Failed to query embedding {idx}: {e}")
                continue
        
        # Determine recognized students based on multiple detections
        print("Student detection summary:")
        for student_id, scores in student_scores.items():
            avg_score = np.mean(scores)
            detection_count = len(scores)
            print(f"  Student {student_id}: {detection_count} detections, avg score: {avg_score:.4f}")
            
            # Require multiple detections for higher confidence
            if detection_count >= 2 and avg_score >= threshold:
                student = db.query(Student).filter(Student.student_id == student_id).first()
                if student:
                    recognized_students.add((student.id, student.name, student.student_id))
                    print(f"  ✓ Student recognized: {student.name} ({student.student_id})")
            elif detection_count == 1 and avg_score >= (threshold + 0.1):  # Higher threshold for single detection
                student = db.query(Student).filter(Student.student_id == student_id).first()
                if student:
                    recognized_students.add((student.id, student.name, student.student_id))
                    print(f"  ✓ Student recognized (high confidence): {student.name} ({student.student_id})")
            else:
                print(f"  ✗ Student {student_id}: insufficient confidence or detections")
    else:
        print("No face embeddings collected, cannot perform recognition")

    print("Recognition completed:")
    print(f"  Frames processed: {len(frames)}")
    print(f"  Faces detected: {total_faces_detected}")
    print(f"  Students recognized: {len(recognized_students)}")

    return [
        {"id": sid, "name": name, "student_id": stud_id}
        for sid, name, stud_id in recognized_students
    ]
