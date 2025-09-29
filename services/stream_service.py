"""
Streaming service: read frames from cv2.VideoCapture and send to WebSocket.

Provides functions for:
1. Live camera/webcam streaming with real-time recognition
2. Uploaded video file streaming with progress tracking and recognition

Behavior:
 - Opens the requested source (camera index if source.isdigit(), else file path).
 - Reads frames, skipping frames according to frame_skip.
 - Sends each encoded JPEG frame as a binary websocket message.
 - Maintains a rolling buffer of frames; every `buffer_size` frames it runs
   recognition (using your existing `recognize_students_from_frames`) in a thread
   and sends a JSON text message with recognized students.
 - Avoids duplicate recognition notifications by only sending new students seen
   since session start (keeps an in-memory set).
"""
import cv2
import asyncio
import json
import numpy as np
from typing import List
from fastapi import WebSocket
from services.face_service import recognize_students_from_frames, detect_faces_with_bboxes
from services.session_service import session_manager
from sqlalchemy.orm import Session
import time

async def stream_video_to_websocket(
    websocket: WebSocket,
    source: str,
    db: Session,
    buffer_size: int = 8,
    frame_skip: int = 1,
    fps: int = 10,
):
    """
    Core streaming loop for live camera/webcam.
    Uses separate tasks for streaming and recognition to prevent blocking.

    Args:
        websocket: FastAPI WebSocket to send binary frames & text messages
        source: camera index (str of int) or path to video file
        db: SQLAlchemy Session for recognition lookups
        buffer_size: call recognition every buffer_size frames
        frame_skip: send only every `frame_skip`-th frame
        fps: approximate frames-per-second to send (controls loop sleep)
    """
    # choose capture source
    capture_index = None
    cap = None
    if isinstance(source, str) and source.isdigit():
        capture_index = int(source)
        cap = cv2.VideoCapture(capture_index)
    else:
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # Shared state between streaming and recognition tasks
    frames_buffer: List[np.ndarray] = []
    seen_student_ids = set()
    buffer_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    
    async def recognition_task():
        """Background task for face recognition - runs independently"""
        last_recognition_time = 0
        recognition_interval = 2.0  # Run recognition every 2 seconds
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_recognition_time < recognition_interval:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get frames for recognition
                async with buffer_lock:
                    if len(frames_buffer) < buffer_size:
                        await asyncio.sleep(0.1)
                        continue
                    frames_to_process = frames_buffer.copy()
                    frames_buffer.clear()
                
                # Process frames for recognition
                recognitions = await asyncio.to_thread(
                    recognize_students_from_frames, frames_to_process, db
                )
                
                # Get face detections with bounding boxes from latest frame
                detections = await asyncio.to_thread(
                    detect_faces_with_bboxes, frames_to_process[-1]
                )
                
                # Prepare face data for frontend (simple detection without tracking)
                detected_faces = []
                for i, detection in enumerate(detections):
                    bbox = detection["bbox"]  # [x, y, w, h]
                    confidence = detection["confidence"]
                    
                    # Find matching recognition for this detection
                    recognition_data = None
                    if i < len(recognitions):
                        rec = recognitions[i]
                        recognition_data = {
                            "id": rec.get("id"),
                            "name": rec.get("name"),
                            "student_id": rec.get("student_id"),
                            "confidence": rec.get("confidence")
                        }
                    
                    face_data = {
                        "bbox": bbox,
                        "confidence": confidence,
                        "recognized": recognition_data is not None,
                        "recognition": recognition_data
                    }
                    detected_faces.append(face_data)
                
                # Send detection data to frontend
                if detected_faces:
                    payload = {
                        "type": "detection",
                        "faces": detected_faces,
                        "timestamp": time.time()
                    }
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception:
                        break
                
                # Send recognition events for new students
                new_students = []
                for rec in recognitions:
                    sid = str(rec.get("student_id", ""))
                    if sid and sid not in seen_student_ids:
                        seen_student_ids.add(sid)
                        new_students.append(rec)

                if new_students:
                    payload = {"type": "recognition", "students": new_students, "timestamp": time.time()}
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception:
                        break
                
                last_recognition_time = current_time
                
            except Exception as e:
                print(f"Recognition task error: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying

    # Start recognition task in background
    recognition_task_handle = asyncio.create_task(recognition_task())
    
    # Main streaming loop
    frame_count = 0
    last_sent_time = 0.0
    delay = 1.0 / max(1, fps)

    try:
        while True:
            # read frame (run blocking read in thread to avoid blocking event loop)
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                # If file ended, loop the video file. For camera, just continue trying.
                if capture_index is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # camera read failed; wait briefly and retry
                    await asyncio.sleep(0.1)
                    continue

            frame_count += 1
            # frame skipping to reduce load
            if frame_count % frame_skip != 0:
                continue

            # resize to reduce bandwidth/processing
            h, w = frame.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # encode to JPEG in thread (fast)
            success, jpeg = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not success:
                continue
            jpeg_bytes = jpeg.tobytes()

            # Send binary frame (this should be fast and not block)
            try:
                await websocket.send_bytes(jpeg_bytes)
            except Exception:
                # client likely disconnected
                break

            # Add frame to recognition buffer (non-blocking)
            async with buffer_lock:
                frames_buffer.append(frame.copy())
                if len(frames_buffer) > buffer_size * 2:  # Keep more frames for recognition
                    frames_buffer.pop(0)

            # throttle to target fps
            elapsed = time.time() - last_sent_time
            sleep_time = max(0.0, delay - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            last_sent_time = time.time()

    finally:
        # Stop recognition task
        stop_event.set()
        try:
            await asyncio.wait_for(recognition_task_handle, timeout=2.0)
        except asyncio.TimeoutError:
            recognition_task_handle.cancel()
        
        try:
            cap.release()
        except Exception:
            pass


async def stream_uploaded_video_to_websocket(
    websocket: WebSocket,
    session_id: str,
    video_path: str,
    db: Session,
    buffer_size: int = 8,
    frame_skip: int = 1,
    fps: int = 15,
):
    """
    Stream an uploaded video file with progress tracking and recognition.
    Uses separate tasks for streaming and recognition to prevent blocking.

    Args:
        websocket: FastAPI WebSocket to send frames & messages
        session_id: Session ID for tracking progress
        video_path: Path to uploaded video file
        db: SQLAlchemy Session for recognition lookups
        buffer_size: call recognition every buffer_size frames
        frame_skip: send only every `frame_skip`-th frame
        fps: approximate frames-per-second to send
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    session_manager.update_session(session_id, total_frames=total_frames)

    # Shared state between streaming and recognition tasks
    frames_buffer: List[np.ndarray] = []
    recognized_students_in_session = []
    seen_student_ids = set()
    buffer_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    
    async def recognition_task():
        """Background task for face recognition - runs independently"""
        last_recognition_time = 0
        recognition_interval = 3.0  # Run recognition every 3 seconds for uploaded videos
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_recognition_time < recognition_interval:
                    await asyncio.sleep(0.2)
                    continue
                
                # Get frames for recognition
                async with buffer_lock:
                    if len(frames_buffer) < buffer_size:
                        await asyncio.sleep(0.2)
                        continue
                    frames_to_process = frames_buffer.copy()
                    frames_buffer.clear()
                
                # Process frames for recognition
                recognitions = await asyncio.to_thread(
                    recognize_students_from_frames, frames_to_process, db
                )
                
                # Get face detections with bounding boxes from latest frame
                detections = await asyncio.to_thread(
                    detect_faces_with_bboxes, frames_to_process[-1]
                )
                
                # Prepare face data for frontend (simple detection without tracking)
                detected_faces = []
                for i, detection in enumerate(detections):
                    bbox = detection["bbox"]  # [x, y, w, h]
                    confidence = detection["confidence"]
                    
                    # Find matching recognition for this detection
                    recognition_data = None
                    if i < len(recognitions):
                        rec = recognitions[i]
                        recognition_data = {
                            "id": rec.get("id"),
                            "name": rec.get("name"),
                            "student_id": rec.get("student_id"),
                            "confidence": rec.get("confidence")
                        }
                    
                    face_data = {
                        "bbox": bbox,
                        "confidence": confidence,
                        "recognized": recognition_data is not None,
                        "recognition": recognition_data
                    }
                    detected_faces.append(face_data)
                
                # Send detection data to frontend
                if detected_faces:
                    payload = {
                        "type": "detection",
                        "faces": detected_faces,
                        "timestamp": time.time()
                    }
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception:
                        break
                
                # Handle new student recognitions
                new_students = []
                for r in recognitions:
                    sid = str(r.get("student_id", "") or r.get("id", ""))
                    if sid and sid not in seen_student_ids:
                        seen_student_ids.add(sid)
                        new_students.append(r)
                        recognized_students_in_session.append(r)

                if new_students:
                    # Update session
                    session_manager.update_session(
                        session_id, 
                        recognized_students=recognized_students_in_session
                    )
                    
                    # Send recognition event
                    payload = {
                        "type": "recognition", 
                        "students": new_students, 
                        "timestamp": time.time(),
                        "total_recognized": len(recognized_students_in_session)
                    }
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception:
                        break
                
                last_recognition_time = current_time
                
            except Exception as e:
                print(f"Recognition task error: {e}")
                await asyncio.sleep(1.0)

    # Start recognition task in background
    recognition_task_handle = asyncio.create_task(recognition_task())
    
    # Main streaming loop
    frame_count = 0
    processed_frames = 0
    last_sent_time = 0.0
    delay = 1.0 / max(1, fps)

    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "processing",
            "total_frames": total_frames,
            "message": "Starting video processing..."
        }))

        while True:
            # read frame
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                # End of video
                break

            frame_count += 1
            processed_frames += 1

            # Send progress update every 30 frames
            if processed_frames % 30 == 0:
                progress_percentage = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                session_manager.update_session(session_id, progress=progress_percentage)
                
                await websocket.send_text(json.dumps({
                    "type": "progress",
                    "current_frame": processed_frames,
                    "total_frames": total_frames,
                    "percentage": round(progress_percentage, 1)
                }))

            # frame skipping to reduce load
            if frame_count % frame_skip != 0:
                continue

            # resize to reduce bandwidth
            h, w = frame.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # encode to JPEG with quality control
            success, jpeg = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not success:
                continue
            jpeg_bytes = jpeg.tobytes()

            # Send binary frame (this should be fast and not block)
            try:
                await websocket.send_bytes(jpeg_bytes)
            except Exception:
                # client disconnected
                break

            # Add frame to recognition buffer (non-blocking)
            async with buffer_lock:
                frames_buffer.append(frame.copy())
                if len(frames_buffer) > buffer_size * 2:  # Keep more frames for recognition
                    frames_buffer.pop(0)

            # throttle to target fps
            elapsed = time.time() - last_sent_time
            sleep_time = max(0.0, delay - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            last_sent_time = time.time()

        # Send completion message
        final_progress = 100.0
        session_manager.update_session(
            session_id, 
            progress=final_progress,
            recognized_students=recognized_students_in_session
        )
        
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "completed",
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_recognized": len(recognized_students_in_session),
            "recognized_students": recognized_students_in_session,
            "message": f"Video processing completed. {len(recognized_students_in_session)} students recognized."
        }))

    finally:
        # Stop recognition task
        stop_event.set()
        try:
            await asyncio.wait_for(recognition_task_handle, timeout=3.0)
        except asyncio.TimeoutError:
            recognition_task_handle.cancel()
            
        try:
            cap.release()
        except Exception:
            pass