"""
WebSocket router for streaming video frames and sending recognition results.

Provides websocket endpoints for:
1. /ws/stream - Live camera/webcam streaming with real-time recognition
2. /ws/stream-video/{session_id} - Stream uploaded video files with recognition

Live streaming endpoint:
  /ws/stream?source=<path_or_index>&buffer_size=8&frame_skip=1&fps=10

- source: If numeric string (e.g. "0") it will open that camera index.
         Otherwise treated as a filesystem path to a video file.
- buffer_size: number of frames to collect before running recognition (default 8).
- frame_skip: skip frames to reduce CPU (default 1 -> no skip).
- fps: approximate frames-per-second send rate (default 10).

Uploaded video streaming endpoint:
  /ws/stream-video/{session_id}?buffer_size=8&frame_skip=1&fps=15

- session_id: Session ID returned from /attendance/upload-video
- Other parameters same as live streaming
"""
import os
from fastapi import APIRouter, WebSocket, Query, Depends, WebSocketDisconnect
from database import get_db
from sqlalchemy.orm import Session
from services.stream_service import stream_video_to_websocket, stream_uploaded_video_to_websocket
from services.session_service import session_manager

router = APIRouter(prefix="/ws", tags=["Streaming"])


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query("0"),
    buffer_size: int = Query(8),
    frame_skip: int = Query(1),
    fps: int = Query(10),
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for live camera/webcam streaming with real-time recognition.

    Sends:
      - Binary messages: raw JPEG bytes of each frame.
      - Text messages: JSON strings for recognition events of form:
          {"type":"recognition","students":[{"id":..,"name":..,"student_id":..}, ...]}
    """
    await websocket.accept()
    try:
        await stream_video_to_websocket(
            websocket=websocket,
            source=source,
            db=db,
            buffer_size=buffer_size,
            frame_skip=frame_skip,
            fps=fps,
        )
    except WebSocketDisconnect:
        # client disconnected
        pass
    except Exception as e:
        # try to inform client of error
        try:
            await websocket.send_text(f'{{"type":"error","message":"{str(e)}"}}')
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/stream-video/{session_id}")
async def websocket_stream_uploaded_video(
    websocket: WebSocket,
    session_id: str,
    buffer_size: int = Query(8),
    frame_skip: int = Query(1),
    fps: int = Query(15),  # Slightly higher FPS for uploaded videos
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for streaming uploaded video files with recognition.

    Sends:
      - Binary messages: raw JPEG bytes of each frame
      - Text messages: JSON strings for:
          - Recognition events: {"type":"recognition","students":[...]}
          - Progress updates: {"type":"progress","current_frame":N,"total_frames":T,"percentage":P}
          - Status updates: {"type":"status","status":"processing|completed|error"}
    """
    await websocket.accept()
    
    # Validate session exists
    session_data = session_manager.get_session(session_id)
    if not session_data:
        try:
            await websocket.send_text('{"type":"error","message":"Session not found"}')
            await websocket.close()
        except Exception:
            pass
        return
    
    # Check if video file exists
    video_path = session_data["video_path"]
    if not video_path or not os.path.exists(video_path):
        try:
            await websocket.send_text('{"type":"error","message":"Video file not found"}')
            await websocket.close()
        except Exception:
            pass
        return
    
    try:
        # Update session status
        session_manager.update_session(session_id, status="streaming")
        
        # Stream the uploaded video
        await stream_uploaded_video_to_websocket(
            websocket=websocket,
            session_id=session_id,
            video_path=video_path,
            db=db,
            buffer_size=buffer_size,
            frame_skip=frame_skip,
            fps=fps,
        )
        
        # Mark session as completed
        session_manager.update_session(session_id, status="completed")
        
    except WebSocketDisconnect:
        # client disconnected
        session_manager.update_session(session_id, status="disconnected")
    except Exception as e:
        # Update session with error
        session_manager.update_session(session_id, status="error")
        
        # try to inform client of error
        try:
            await websocket.send_text(f'{{"type":"error","message":"{str(e)}"}}')
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass