import cv2
import base64
import os
import uuid
import asyncio
import numpy as np
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed

# import the Pinecone index from your vector_db.py
from vector_db import index
from models.models import Student
from database import SessionLocal


# ------------------------------
# Pinecone query helper
# ------------------------------
def pinecone_query(embedding: list, top_k: int = 1, threshold: float = 0.5):
    if index is None:
        return None

    try:
        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        if not response or not response.matches:
            return None

        filtered = [m for m in response.matches if m.score >= threshold]
        if not filtered:
            return None

        response.matches = filtered
        return response
    except Exception as e:
        print("Pinecone query error:", e)
        return None


# ------------------------------
# Frame encoder
# ------------------------------
def encode_frame_to_base64_jpg(frame: np.ndarray) -> str:
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ------------------------------
# Helper: get face embeddings
# ------------------------------
def get_embeddings_from_frame(frame):
    """Extract normalized embeddings from a single frame (BGR image)."""
    temp_name = f"temp_frame_{uuid.uuid4().hex[:8]}.jpg"
    try:
        cv2.imwrite(temp_name, frame)
        faces = DeepFace.represent(
            img_path=temp_name,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False
        )
        embeddings = []
        if isinstance(faces, list):
            for f in faces:
                emb = f.get("embedding")
                if emb is not None:
                    emb = np.array(emb)
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append({
                        "embedding": emb.tolist(),
                        "facial_area": f.get("facial_area", {})
                    })
        return embeddings
    except Exception as e:
        print("Embedding error:", e)
        return []
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


# ------------------------------
# Main function (parallelized)
# ------------------------------
async def process_and_stream_video(ws, video_path: str, frame_skip: int = 2, top_k: int = 1, batch_size: int = 8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await ws.send_json({"error": f"Cannot open video at path: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    delay_seconds = frame_skip / fps
    frame_idx = 0

    frames_batch = []
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frames_batch.append((frame_idx, frame.copy()))

                # Process batch in parallel
                if len(frames_batch) >= batch_size:
                    futures = {executor.submit(get_embeddings_from_frame, f): idx for idx, f in frames_batch}
                    for future in as_completed(futures):
                        idx = futures[future]
                        frame_for_send = [f for i, f in frames_batch if i == idx][0]
                        detections_out = []

                        embeddings_info = future.result() or []
                        for emb_info in embeddings_info:
                            emb = emb_info["embedding"]
                            box = emb_info.get("facial_area", {})
                            name, student_id, score = "Unknown", None, None

                            response = pinecone_query(emb, top_k=top_k, threshold=0.5)
                            if response and len(response.matches) > 0:
                                best_match = response.matches[0]
                                score = best_match.score
                                metadata = best_match.metadata or {}
                                student_id = metadata.get("student_id")

                                if student_id:
                                    db = SessionLocal()
                                    try:
                                        student = db.query(Student).filter(Student.student_id == student_id).first()
                                        name = student.name if student else "Unknown"
                                    finally:
                                        db.close()

                            # Draw rectangle & label on frame
                            if isinstance(box, dict) and {"x", "y", "w", "h"}.issubset(box.keys()):
                                x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                                cv2.rectangle(frame_for_send, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame_for_send, name, (x, max(0, y - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                box_out = {"x": x, "y": y, "w": w, "h": h}
                            else:
                                box_out = {}

                            detections_out.append({
                                "student_id": student_id,
                                "name": name,
                                "box": box_out,
                                "score": score
                            })

                        frame_b64 = encode_frame_to_base64_jpg(frame_for_send)
                        await ws.send_json({
                            "frame_b64": frame_b64,
                            "frame_index": idx,
                            "detections": detections_out
                        })
                        await asyncio.sleep(delay_seconds)

                    frames_batch.clear()

            frame_idx += 1

    finally:
        cap.release()
        executor.shutdown(wait=True)
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print("Error deleting temp video:", e)
