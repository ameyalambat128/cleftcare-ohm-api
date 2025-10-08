import os
import re
import uuid
import time
import math
from typing import Any, Dict


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())


def validate_upload_filename(filename: str) -> str:
    """Validate and sanitize upload filename to prevent path traversal"""
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Remove any path separators and relative path indicators
    filename = os.path.basename(filename)
    filename = filename.replace('..', '')

    # Allow only alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        raise ValueError("Invalid filename format")

    # Check file extension (only allow expected audio formats)
    allowed_extensions = {'.m4a', '.wav', '.mp3', '.flac'}
    _, ext = os.path.splitext(filename.lower())
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}")

    return filename


def create_response(success: bool, data: dict, request_id: str, processing_time: float, error_message: str = None) -> dict:
    """Create standardized API response with safe JSON values (no NaN/Inf)."""
    safe_data = sanitize_json_floats(data) if success else {}
    return {
        "success": success,
        "data": safe_data,
        "metadata": {
            "requestId": request_id,
            "processingTime": round(processing_time, 3),
            "timestamp": int(time.time() * 1000),
            "error": error_message if not success else None
        }
    }


# In-memory status tracking
status_tracking = {}

def update_status(user_id: str, request_id: str, status: str, endpoint: str, data: dict = None):
    """Update processing status for a user"""
    if user_id not in status_tracking:
        status_tracking[user_id] = []

    status_tracking[user_id].append({
        "requestId": request_id,
        "status": status,  # "processing", "completed", "failed"
        "endpoint": endpoint,
        "timestamp": int(time.time() * 1000),
        "data": sanitize_json_floats(data) if data else {}
    })


def sanitize_json_floats(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None and convert tuples to lists.

    This ensures the payload is JSON-serializable under strict encoders.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json_floats(v) for v in obj]
    return obj