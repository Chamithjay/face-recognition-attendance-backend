"""
Session management service for uploaded video streaming.

Manages video upload sessions, storing file paths and metadata for WebSocket streaming.
"""
import os
import uuid
import tempfile
from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio
from threading import Lock

class VideoSessionManager:
    """Manages uploaded video sessions for WebSocket streaming."""
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.lock = Lock()
        self.temp_dir = tempfile.mkdtemp(prefix="video_sessions_")
        
    def create_session(self, filename: str, file_content: bytes) -> str:
        """
        Create a new video session.
        
        Args:
            filename: Original filename of uploaded video
            file_content: Video file bytes
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        # Save video file
        file_extension = os.path.splitext(filename)[1] or '.mp4'
        video_path = os.path.join(self.temp_dir, f"{session_id}{file_extension}")
        
        with open(video_path, "wb") as f:
            f.write(file_content)
        
        # Create session metadata
        session_data = {
            "session_id": session_id,
            "filename": filename,
            "video_path": video_path,
            "created_at": datetime.now(),
            "status": "ready",  # ready, streaming, completed, error
            "progress": 0,
            "total_frames": None,
            "recognized_students": [],
        }
        
        with self.lock:
            self.sessions[session_id] = session_data
            
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data by ID."""
        with self.lock:
            return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, **updates):
        """Update session data."""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(updates)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions and their files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.lock:
            expired_sessions = [
                sid for sid, data in self.sessions.items()
                if data["created_at"] < cutoff_time
            ]
            
            for session_id in expired_sessions:
                session_data = self.sessions.pop(session_id)
                # Delete video file
                if os.path.exists(session_data["video_path"]):
                    try:
                        os.remove(session_data["video_path"])
                    except Exception:
                        pass  # File might already be deleted
    
    def delete_session(self, session_id: str):
        """Delete a specific session and its files."""
        with self.lock:
            if session_id in self.sessions:
                session_data = self.sessions.pop(session_id)
                # Delete video file
                if os.path.exists(session_data["video_path"]):
                    try:
                        os.remove(session_data["video_path"])
                    except Exception:
                        pass

# Global session manager instance
session_manager = VideoSessionManager()

# Periodic cleanup task
async def cleanup_sessions_periodically():
    """Background task to clean up old sessions."""
    while True:
        try:
            session_manager.cleanup_old_sessions()
            await asyncio.sleep(3600)  # Run every hour
        except Exception:
            await asyncio.sleep(3600)  # Continue on error