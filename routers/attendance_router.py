# routers/attendance_router.py
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
import os
import traceback
from services.video_service import save_uploaded_video, get_video_full_path
from services.attendance_service import process_and_stream_video

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/upload-video")
async def upload_video_endpoint(video: UploadFile = File(...)):
    """
    Upload a test video. Returns file path (basename) to be used by the websocket stream.
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
    WebSocket streaming endpoint.
    Client should connect to: ws://<host>/attendance/ws?video=<filename>
    where <filename> is the basename returned from /attendance/upload.

    Sends JSON objects with:
    {
      frame_b64: "...", frame_index: int, detections: [...]
    }
    """
    await websocket.accept()
    params = websocket.query_params
    video_filename = params.get("video")
    if not video_filename:
        await websocket.send_json({"error": "Missing 'video' query parameter. Example: /attendance/ws?video=test.mp4"})
        await websocket.close()
        return

    # compute path
    video_path = get_video_full_path(video_filename)
    if not os.path.exists(video_path):
        await websocket.send_json({"error": f"Video not found on server: {video_filename}"})
        await websocket.close()
        return

    try:
        await process_and_stream_video(websocket, video_path, frame_skip=2, top_k=1)
    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
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
