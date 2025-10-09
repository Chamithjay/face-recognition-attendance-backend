import cv2
import base64
import os
import asyncio
import numpy as np
from deepface import DeepFace
import tensorflow as tf

from vector_db import index
from models.models import Student
from database import SessionLocal

# ------------------------------
# GPU check
# ------------------------------
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("GPU not detected. This service only supports GPU execution.")
print(f"✅ GPU detected: {gpus}")

# ------------------------------
# Preload ArcFace Model
# ------------------------------
try:
    ARCFACE_MODEL = DeepFace.build_model("ArcFace")
    print("✅ ArcFace model loaded successfully on GPU")
except Exception as e:
    raise RuntimeError(f"Failed to load ArcFace model: {e}")

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
# GPU batch embedding function - OPTIMIZED (in-memory processing)
# ------------------------------
def get_embeddings_from_frame(frames):
    """
    Process frames directly in memory using numpy arrays.
    No disk I/O - massive performance improvement!
    """
    try:
        # DeepFace.represent accepts numpy arrays directly
        # Process frames in batch for GPU efficiency
        all_embeddings = []
        
        for frame in frames:
            try:
                # Pass numpy array directly - no temp files!
                faces = DeepFace.represent(
                    img_path=frame,  # numpy array supported!
                    model_name="ArcFace",
                    detector_backend="retinaface",
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
                all_embeddings.append(embeddings)
                
            except Exception as e:
                print(f"Frame embedding error: {e}")
                all_embeddings.append([])
        
        return all_embeddings

    except Exception as e:
        print("Batch embedding error:", e)
        return [[] for _ in frames]

# ------------------------------
# Async GPU queue video processor with dynamic batch sizing
# ------------------------------
async def process_and_stream_video(ws, video_path: str, frame_skip: int = 2, max_batch_size: int = 16, top_k: int = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await ws.send_json({"error": f"Cannot open video at path: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    delay_seconds = frame_skip / fps
    frame_idx = 0

    frame_queue = asyncio.Queue(maxsize=2*max_batch_size)
    stop_flag = False

    # ------------------------------
    # Reader coroutine
    # ------------------------------
    async def reader():
        nonlocal frame_idx, stop_flag
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    await frame_queue.put((frame_idx, frame.copy()))
                frame_idx += 1
        finally:
            stop_flag = True

    # ------------------------------
    # Processor coroutine with dynamic batch sizing
    # ------------------------------
    async def processor():
        while not stop_flag or not frame_queue.empty():
            # Estimate batch size based on GPU memory
            gpu = tf.config.experimental.get_memory_info("GPU:0")
            available_mem = gpu['current'] if 'current' in gpu else 1024*1024*1024
            batch_size = max(1, min(max_batch_size, int(available_mem // (50*1024*1024))))
            
            batch = []
            while len(batch) < batch_size and not frame_queue.empty():
                batch.append(await frame_queue.get())

            if not batch:
                await asyncio.sleep(0.01)
                continue

            idx_list, batch_frames = zip(*batch)
            embeddings_batch = get_embeddings_from_frame(list(batch_frames))

            for idx, frame_embeddings, frame_for_send in zip(idx_list, embeddings_batch, batch_frames):
                detections_out = []
                for emb_info in frame_embeddings:
                    emb = np.array(emb_info["embedding"])
                    box = emb_info.get("facial_area", {})
                    name, student_id, score = "Unknown", None, None

                    # Pinecone lookup
                    response = pinecone_query(emb.tolist(), top_k=top_k, threshold=0.5)
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

                    # Draw recognized faces
                    if student_id and isinstance(box, dict) and {"x","y","w","h"}.issubset(box.keys()):
                        x,y,w,h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                        cv2.rectangle(frame_for_send, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.putText(frame_for_send, name, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
                        detections_out.append({
                            "student_id": student_id,
                            "name": name,
                            "box": {"x":x,"y":y,"w":w,"h":h},
                            "score": score
                        })

                frame_b64 = encode_frame_to_base64_jpg(frame_for_send)
                await ws.send_json({
                    "frame_b64": frame_b64,
                    "frame_index": idx,
                    "detections": detections_out
                })
                await asyncio.sleep(delay_seconds)

    # ------------------------------
    # Run concurrently
    # ------------------------------
    await asyncio.gather(reader(), processor())

    cap.release()
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception as e:
            print("Error deleting video:", e)
    print("✅ Video stream fully processed with dynamic batching.")
