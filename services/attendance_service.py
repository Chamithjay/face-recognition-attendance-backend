# services/attendance_service.py
import cv2
import base64
import os
import uuid
import asyncio
import numpy as np
from deepface import DeepFace


# import the Pinecone index from your vector_db.py
from vector_db import index  # index may be None if init failed
# Import Student model if you want to fetch names from DB (optional)
from models.models import Student
from database import SessionLocal  # fallback to create sessions if needed

# Helper to query Pinecone safely


def pinecone_query(embedding: list, top_k: int = 1, threshold: float = 0.5):
    if index is None:
        return None

    try:
        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        # ✅ Apply threshold manually
        if not response or not response.matches:
            print("No matches returned from Pinecone")
            return None

        # Filter by score threshold
        filtered = [m for m in response.matches if m.score >= threshold]

        if len(filtered) == 0:
            print(f"No match above threshold ({response.matches[0].score:.2f} < {threshold})")
            return None

        # ✅ Return full Pinecone response object (not dict)
        response.matches = filtered
        return response

    except Exception as e:
        print("Pinecone query error:", e)
        return None


def encode_frame_to_base64_jpg(frame: np.ndarray) -> str:
    """
    Encode BGR OpenCV frame to JPEG base64 string (no data URI prefix).
    """
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        raise RuntimeError("Failed to encode frame")
    jpg_bytes = buffer.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8")

async def process_and_stream_video(ws, video_path: str, frame_skip: int = 2, top_k: int = 1):
    """
    Read video, process frames and stream results to a WebSocket connection (ws).
    - frame_skip: process every Nth frame (lower = more frequent)
    - top_k: Pinecone top_k
    Each send contains:
    {
      "frame_b64": "...",
      "frame_index": 10,
      "detections": [
         {"student_id": "...", "name": "...", "box": {"x":..,"y":..,"w":..,"h":..}, "score": 0.9}
      ]
    }
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await ws.send_json({"error": f"Cannot open video at path: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    # We'll wait roughly frame_skip / fps seconds between processed frames
    delay_seconds = frame_skip / fps

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every frame_skip-th frame (but still send the encoded frame)
            if frame_idx % frame_skip == 0:
                # Make a copy for drawing so original frame remains unmodified
                frame_for_send = frame.copy()

                # Use a temporary image file for DeepFace (consistent with your registration approach)
                tmp_name = f"temp_frame_{uuid.uuid4().hex[:8]}.jpg"
                try:
                    cv2.imwrite(tmp_name, frame)
                except Exception as e:
                    print("Failed to write temp frame:", e)
                    tmp_name = None

                detections_out = []
                try:
                    if tmp_name:
                        # Attempt detection with retinaface (fallback handled inside DeepFace if enforce_detection=False)
                        faces = DeepFace.represent(
                            img_path=tmp_name,
                            model_name="ArcFace",
                            detector_backend="mtcnn",
                            enforce_detection=False
                        )
                    else:
                        faces = []
                except Exception as e:
                    print("DeepFace error:", e)
                    faces = []

                # Clean temp file
                if tmp_name and os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass

                # faces is usually a list of dicts with 'embedding' and maybe 'facial_area'
                if isinstance(faces, list):
                    for f in faces:
                        try:
                            emb = f.get("embedding")
                            # facial_area may be dict like {"x":..,"y":..,"w":..,"h":..}
                            box = f.get("facial_area", {})
                            name = "Unknown"
                            student_id = None
                            score = None

                            if emb is not None:
                                emb = np.array(emb)
                                emb = emb / np.linalg.norm(emb)  # L2 normalization
                                emb = emb.tolist()  # ✅ Pinecone needs a list, not ndarray

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
                                        except Exception as e:
                                            print("DB fetch student error:", e)
                                            name = "Unknown"
                                        finally:
                                            db.close()
                                    else:
                                        name = "Unknown"

                                else:
                                    name = "Unknown"
                                    score = None
                                    student_id = None

                            else:
                                name = "Unknown"
                                score = None
                                student_id = None

                            # Draw rectangle & label on frame_for_send if box present
                            if isinstance(box, dict) and {"x","y","w","h"}.issubset(set(box.keys())):
                                x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                                # ensure coordinates are inside image
                                h_img, w_img = frame_for_send.shape[:2]
                                x = max(0, min(x, w_img-1))
                                y = max(0, min(y, h_img-1))
                                w = max(1, min(w, w_img-x))
                                h = max(1, min(h, h_img-y))
                                # annotate for preview
                                cv2.rectangle(frame_for_send, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame_for_send, name, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                box_out = {"x": x, "y": y, "w": w, "h": h}
                            else:
                                box_out = {}

                            detections_out.append({
                                "student_id": student_id,
                                "name": name,
                                "box": box_out,
                                "score": score
                            })
                        except Exception as inner_e:
                            print("Error handling face:", inner_e)
                            continue

                # encode annotated frame
                try:
                    frame_b64 = encode_frame_to_base64_jpg(frame_for_send)
                except Exception as e:
                    print("Frame encode error:", e)
                    frame_b64 = None

                payload = {
                    "frame_b64": frame_b64,
                    "frame_index": frame_idx,
                    "detections": detections_out
                }

                # send via websocket
                try:
                    await ws.send_json(payload)
                except Exception as send_e:
                    print("WebSocket send error:", send_e)
                    # client likely disconnected — stop processing
                    break

                # sleep to simulate real-time pacing
                await asyncio.sleep(delay_seconds)

            frame_idx += 1

    finally:
        cap.release()
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print("Error deleting temp video:", e)
