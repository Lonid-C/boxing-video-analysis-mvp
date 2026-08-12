from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.getenv("BOXING_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'outputs' / 'boxing_demo.db'}")
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", PROJECT_ROOT)).resolve()
UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", MEDIA_ROOT / 'uploads')).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
ALLOWED_VIDEO_EXTENSIONS = {x.strip().lower() for x in os.getenv("ALLOWED_VIDEO_EXTENSIONS", ".mp4,.mov,.avi,.mkv,.webm").split(",") if x.strip()}
PAGE_SIZE = min(max(int(os.getenv("PAGE_SIZE", "50")), 1), 200)
