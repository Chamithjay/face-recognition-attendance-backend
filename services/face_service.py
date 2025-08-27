import numpy as np
from deepface import DeepFace
from vector_db import index
from sqlalchemy.orm import Session
from models.models import Student

def recognize_students_from_frames(frames, db: Session, threshold=0.5):
    recognized_students = set()
    all_embeddings = []
    total_faces_detected = 0

    print(f"🔍 Starting recognition for {len(frames)} frames with threshold {threshold}")

    # First pass: Extract all embeddings from all frames
    for i, frame in enumerate(frames):
        try:
            print(f"🎬 Processing frame {i+1}/{len(frames)}")
            
            # Extract embedding using ArcFace
            result = DeepFace.represent(frame, model_name="ArcFace", enforce_detection=False)

            if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"])
                all_embeddings.append(embedding)
                total_faces_detected += 1
                print(f"   👥 Face detected in frame {i+1}, embedding dimension: {len(embedding)}")
            else:
                print(f"   ❌ No face detected in frame {i+1}")

        except Exception as e:
            print(f"   ❌ Frame {i+1} failed: {e}")
            continue

    print(f"📊 Collected {len(all_embeddings)} face embeddings from {total_faces_detected} detections")

    # Calculate average embedding if we have any embeddings
    if len(all_embeddings) > 0:
        print("🧮 Calculating average embedding...")
        average_embedding = np.mean(all_embeddings, axis=0)
        print(f"   ✅ Average embedding calculated with dimension: {len(average_embedding)}")

        # Search in Pinecone with average embedding
        if index is None:
            print("   ❌ Pinecone index not available")
            return []
            
        try:
            print("🔍 Querying Pinecone with average embedding...")
            search_result = index.query(
                vector=average_embedding.tolist(),
                top_k=3,  # Get top 3 for debugging
                include_metadata=True
            )

            print(f"   📊 Pinecone returned {len(search_result.matches)} matches")
            
            if search_result and len(search_result.matches) > 0:
                # Log all matches for debugging
                for j, match in enumerate(search_result.matches):
                    student_id = match.metadata.get("student_id", "Unknown")
                    score = match.score
                    print(f"      Match {j+1}: Student ID {student_id}, Score: {score:.4f}")
                
                best_match = search_result.matches[0]
                if best_match.score >= threshold:
                    student_id = best_match.metadata["student_id"]
                    print(f"   ✅ Match found above threshold: Student ID {student_id}, Score: {best_match.score:.4f}")
                    
                    student = db.query(Student).filter(Student.student_id == student_id).first()
                    if student:
                        recognized_students.add((student.id, student.name, student.student_id))
                        print(f"   ✅ Student added: {student.name} ({student.student_id})")
                    else:
                        print(f"   ❌ Student ID {student_id} not found in database")
                else:
                    print(f"   ❌ Best score {best_match.score:.4f} below threshold {threshold}")
            else:
                print("   ❌ No matches returned from Pinecone")
                
        except Exception as e:
            print(f"   ❌ Pinecone query failed: {e}")
    else:
        print("❌ No face embeddings collected, cannot perform recognition")

    print(f"🎯 Recognition completed:")
    print(f"   Frames processed: {len(frames)}")
    print(f"   Faces detected: {total_faces_detected}")
    print(f"   Average embedding used: {'Yes' if len(all_embeddings) > 0 else 'No'}")
    print(f"   Students recognized: {len(recognized_students)}")

    # Convert set → list of dicts
    return [
        {"id": sid, "name": name, "student_id": stud_id}
        for sid, name, stud_id in recognized_students
    ]
