"""
Attendance tracking endpoints with video upload and WebSocket streaming.
"""

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
import os
import traceback
from services.video_service import save_uploaded_video, get_video_full_path
from services.attendance_service import process_and_stream_video

router = APIRouter(prefix="/attendance", tags=["Attendance"])


@router.post("/upload-video")
async def upload_video_endpoint(video: UploadFile = File(...)):
    """
    Upload video file for attendance processing.
    
    Args:
        video: Video file containing student faces
        
    Returns:
        Dictionary with filename and full path
    """
    try:
        saved_path = save_uploaded_video(video)
        filename = os.path.basename(saved_path)
        return {"filename": filename, "path": saved_path}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

@router.websocket("/ws")
async def attendance_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time attendance video processing.
    
    Connect to: ws://<host>/attendance/ws?video=<filename>
    
    Streams JSON objects with frame_b64, frame_index, and detections array.
    """
    await websocket.accept()
    params = websocket.query_params
    video_filename = params.get("video")
    if not video_filename:
        await websocket.send_json({"error": "Missing 'video' query parameter"})
        await websocket.close()
        return

    video_path = get_video_full_path(video_filename)
    if not os.path.exists(video_path):
        await websocket.send_json({"error": f"Video not found: {video_filename}"})
        await websocket.close()
        return

    try:
        await process_and_stream_video(websocket, video_path, frame_skip=2, top_k=1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"error": f"Server error: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
