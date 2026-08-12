from __future__ import annotations
import html
import json
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_BYTES, MEDIA_ROOT, PROJECT_ROOT, UPLOAD_ROOT
from .db import get_db, init_db
from .models import AnalysisRun, Event, Track, Video
from .schemas import EventOut, RunCreate, RunOut, TrackOut, VideoCreate, VideoOut
from ..services import start_run
app = FastAPI(title="Boxing Analysis Demo API", version="1.1.0")
@app.on_event("startup")
def startup() -> None:
    init_db(); UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
def _safe_media_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    try: candidate.relative_to(MEDIA_ROOT.resolve())
    except ValueError as exc: raise HTTPException(404, "media not found") from exc
    if not candidate.is_file(): raise HTTPException(404, "media not found")
    return candidate
@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}
@app.get("/api/v1/videos", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=200)):
    return list(db.scalars(select(Video).order_by(Video.created_at.desc()).limit(limit)))
@app.post("/api/v1/uploads", response_model=VideoOut, status_code=201)
def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS: raise HTTPException(415, "unsupported video extension")
    target = (UPLOAD_ROOT / (Path(file.filename or "video").stem[:80] + suffix)).resolve()
    size = 0
    with target.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True); raise HTTPException(413, "video too large")
            out.write(chunk)
    video = Video(path=str(target), name=file.filename or target.name)
    db.add(video); db.commit(); db.refresh(video); return video
@app.post("/api/v1/videos", response_model=VideoOut, status_code=201)
def create_video(payload: VideoCreate, db: Session = Depends(get_db)):
    path = Path(payload.path).expanduser().resolve()
    try: path.relative_to(MEDIA_ROOT.resolve())
    except ValueError as exc: raise HTTPException(422, "video path must be inside MEDIA_ROOT") from exc
    if not path.is_file(): raise HTTPException(404, "video file not found")
    existing = db.scalar(select(Video).where(Video.path == str(path)))
    if existing: return existing
    video = Video(path=str(path), name=payload.name or path.name, duration_sec=payload.duration_sec, fps=payload.fps, width=payload.width, height=payload.height)
    db.add(video); db.commit(); db.refresh(video); return video
@app.get("/api/v1/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video: raise HTTPException(404, "video not found")
    return video
@app.post("/api/v1/runs", response_model=RunOut, status_code=201)
def create_run(payload: RunCreate, db: Session = Depends(get_db)):
    video = db.get(Video, payload.video_id)
    if not video: raise HTTPException(404, "video not found")
    run = AnalysisRun(video_id=video.id, mode=payload.mode, status="queued", params=payload.params)
    db.add(run); db.commit(); db.refresh(run); start_run(run, video.path); return run
@app.get("/api/v1/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(AnalysisRun, run_id)
    if not run: raise HTTPException(404, "run not found")
    return run
@app.get("/api/v1/runs/{run_id}/events", response_model=list[EventOut])
def list_events(run_id: str, source: str | None = None, hit_or_miss: str | None = None, start: float | None = None, end: float | None = None, db: Session = Depends(get_db), limit: int = Query(200, ge=1, le=1000)):
    query = select(Event).where(Event.run_id == run_id)
    if source: query = query.where(Event.source == source)
    if hit_or_miss: query = query.where(Event.hit_or_miss == hit_or_miss)
    if start is not None: query = query.where(Event.end_time_sec >= start)
    if end is not None: query = query.where(Event.start_time_sec <= end)
    return list(db.scalars(query.order_by(Event.start_time_sec).limit(limit)))
@app.get("/api/v1/runs/{run_id}/tracks", response_model=list[TrackOut])
def list_tracks(run_id: str, db: Session = Depends(get_db), limit: int = Query(500, ge=1, le=5000)):
    return list(db.scalars(select(Track).where(Track.run_id == run_id).order_by(Track.frame_index).limit(limit)))
@app.get("/media/{file_path:path}")
def media(file_path: str): return FileResponse(_safe_media_path(str(MEDIA_ROOT / file_path)))
@app.get("/", response_class=HTMLResponse)
def home(db: Session = Depends(get_db)):
    videos = list(db.scalars(select(Video).order_by(Video.created_at.desc()).limit(50)))
    rows = "".join(f'<li><a href="/videos/{html.escape(v.id)}">{html.escape(v.name)}</a> <span>{v.duration_sec or 0:.1f}s</span></li>' for v in videos)
    return "<!doctype html><meta charset='utf-8'><title>Boxing Demo</title><style>body{font-family:system-ui;max-width:900px;margin:40px auto}li{margin:14px 0}a{font-size:18px}</style><h1>拳击视频分析 Demo</h1><p>上传视频后创建分析任务。</p><ul>" + (rows or "<li>暂无视频</li>") + "</ul>"
@app.get("/videos/{video_id}", response_class=HTMLResponse)
def video_page(video_id: str, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video: raise HTTPException(404, "video not found")
    run = db.scalar(select(AnalysisRun).where(AnalysisRun.video_id == video_id).order_by(AnalysisRun.created_at.desc()))
    events = list(db.scalars(select(Event).where(Event.run_id == run.id).order_by(Event.start_time_sec))) if run else []
    data = [{"start": e.start_time_sec or 0, "end": e.end_time_sec or (e.start_time_sec or 0)+.8, "label": e.hit_or_miss or e.source} for e in events]
    rel = Path(video.path).resolve().relative_to(PROJECT_ROOT.resolve())
    media_url = "/media/" + "/".join(rel.parts)
    payload = json.dumps(data, ensure_ascii=False)
    return "<!doctype html><meta charset='utf-8'><title>" + html.escape(video.name) + "</title><style>body{font-family:system-ui;max-width:1000px;margin:24px auto}video{width:100%;background:#111}button{margin:5px;padding:8px}</style><h1>" + html.escape(video.name) + "</h1><p>运行状态：" + html.escape(run.status if run else "未运行") + "</p><video id='v' controls src='" + html.escape(media_url) + "'></video><h2>事件</h2><div id='events'></div><script>const es=" + payload + ",v=document.getElementById('v');let stop=0;es.forEach(e=>{let b=document.createElement('button');b.textContent=e.label+' '+e.start.toFixed(2)+'s';b.onclick=()=>{v.currentTime=e.start;stop=e.end;v.play()};document.getElementById('events').append(b)});v.ontimeupdate=()=>{if(stop&&v.currentTime>=stop){v.pause();stop=0}};</script>"
