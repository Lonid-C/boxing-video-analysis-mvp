from __future__ import annotations
import html
import json
import os
import cv2
from contextlib import asynccontextmanager
from uuid import uuid4
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from .config import ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_BYTES, MEDIA_ROOT, PROJECT_ROOT, UPLOAD_ROOT
from .db import get_db, init_db
from .models import AnalysisRun, Artifact, Event, Track, Video
from .schemas import ArtifactOut, EventOut, RunCreate, RunOut, TrackOut, VideoCreate, VideoOut
from ..services import start_run


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(); UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    # In-process worker threads do not survive a service restart. Make the
    # interrupted state explicit instead of leaving jobs stuck forever.
    from .db import SessionLocal
    with SessionLocal() as db:
        db.execute(update(AnalysisRun).where(AnalysisRun.status.in_(["queued", "running"])).values(
            status="failed", error="analysis interrupted by service restart; create a new run to retry"))
        db.commit()
    yield


app = FastAPI(title="Boxing Analysis Demo API", version="1.1.0", lifespan=lifespan)


def _video_metadata(path: Path) -> dict[str, float | int | None]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise HTTPException(422, "uploaded file is not a readable video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, _ = capture.read()
        if not ok or width <= 0 or height <= 0 or fps <= 0:
            raise HTTPException(422, "uploaded file contains no decodable video frames")
        return {"width": width, "height": height, "fps": fps,
                "duration_sec": frame_count / fps if frame_count > 0 else None}
    finally:
        capture.release()


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
    safe_stem = "".join(c for c in Path(file.filename or "video").stem[:80]
                        if c.isalnum() or c in {"-", "_"}).strip("._") or "video"
    target = (UPLOAD_ROOT / f"{safe_stem}-{uuid4().hex[:12]}{suffix}").resolve()
    temporary = target.with_name(f".{target.name}.part")
    try: target.relative_to(UPLOAD_ROOT.resolve())
    except ValueError as exc: raise HTTPException(422, "invalid upload path") from exc
    size = 0
    published = False
    try:
        with temporary.open("xb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "video too large")
                out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        os.replace(temporary, target)
        published = True
        metadata = _video_metadata(target)
        video = Video(path=str(target), name=file.filename or target.name, **metadata)
        db.add(video); db.commit(); db.refresh(video); return video
    except Exception:
        db.rollback()
        temporary.unlink(missing_ok=True)
        if published:
            target.unlink(missing_ok=True)
        raise
@app.post("/api/v1/videos", response_model=VideoOut, status_code=201)
def create_video(payload: VideoCreate, db: Session = Depends(get_db)):
    path = Path(payload.path).expanduser().resolve()
    try: path.relative_to(MEDIA_ROOT.resolve())
    except ValueError as exc: raise HTTPException(422, "video path must be inside MEDIA_ROOT") from exc
    if not path.is_file(): raise HTTPException(404, "video file not found")
    metadata = _video_metadata(path)
    existing = db.scalar(select(Video).where(Video.path == str(path)))
    if existing: return existing
    video = Video(path=str(path), name=payload.name or path.name,
        duration_sec=payload.duration_sec if payload.duration_sec is not None else metadata["duration_sec"],
        fps=payload.fps if payload.fps is not None else metadata["fps"],
        width=payload.width if payload.width is not None else metadata["width"],
        height=payload.height if payload.height is not None else metadata["height"])
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
    params = dict(payload.params)
    for key in ("export_dir", "output", "stats"):
        if key in params:
            target = Path(str(params[key])).expanduser().resolve()
            try: target.relative_to(MEDIA_ROOT.resolve())
            except ValueError as exc: raise HTTPException(422, f"{key} must be inside MEDIA_ROOT") from exc
    run = AnalysisRun(video_id=video.id, mode=payload.mode, status="queued", params=params)
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
@app.get("/api/v1/runs/{run_id}/artifacts", response_model=list[ArtifactOut])
def list_artifacts(run_id: str, db: Session = Depends(get_db)):
    if not db.get(AnalysisRun, run_id): raise HTTPException(404, "run not found")
    output = []
    for artifact in db.scalars(select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.kind)):
        media_url = None
        try:
            rel = Path(artifact.path).resolve().relative_to(MEDIA_ROOT.resolve())
            media_url = "/media/" + "/".join(rel.parts)
        except ValueError:
            pass
        output.append(ArtifactOut.model_validate(artifact).model_copy(update={"media_url": media_url}))
    return output
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
    artifacts = list(db.scalars(select(Artifact).where(Artifact.run_id == run.id))) if run else []
    data = [{"start": e.start_time_sec or 0, "end": e.end_time_sec or (e.start_time_sec or 0)+.8, "label": e.hit_or_miss or e.source} for e in events]
    try:
        rel = Path(video.path).resolve().relative_to(MEDIA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(404, "media not found") from exc
    media_url = "/media/" + "/".join(rel.parts)
    links = []
    heatmaps = []
    for artifact in artifacts:
        try:
            artifact_rel = Path(artifact.path).resolve().relative_to(MEDIA_ROOT.resolve())
        except ValueError:
            continue
        url = "/media/" + "/".join(artifact_rel.parts)
        label = html.escape(artifact.kind)
        links.append(f'<a href="{html.escape(url)}" download>{label}</a>')
        if artifact.kind in {"heatmap_red", "heatmap_blue"}:
            heatmaps.append(f'<figure><img src="{html.escape(url)}"><figcaption>{label}</figcaption></figure>')
        if artifact.kind == "summary_video":
            media_url = url
    payload = json.dumps(data, ensure_ascii=False)
    return "<!doctype html><meta charset='utf-8'><title>" + html.escape(video.name) + "</title><style>body{font-family:system-ui;max-width:1000px;margin:24px auto}video{width:100%;background:#111}button,a{margin:5px;padding:8px}figure{display:inline-block;width:46%;margin:2%}img{width:100%}</style><h1>" + html.escape(video.name) + "</h1><p>运行状态：" + html.escape(run.status if run else "未运行") + "</p><video id='v' controls src='" + html.escape(media_url) + "'></video><h2>结果下载</h2><div>" + " ".join(links) + "</div><h2>移动热力图</h2><div>" + "".join(heatmaps) + "</div><h2>事件</h2><div id='events'></div><script>const es=" + payload + ",v=document.getElementById('v');let stop=0;es.forEach(e=>{let b=document.createElement('button');b.textContent=e.label+' '+e.start.toFixed(2)+'s';b.onclick=()=>{v.currentTime=e.start;stop=e.end;v.play()};document.getElementById('events').append(b)});v.ontimeupdate=()=>{if(stop&&v.currentTime>=stop){v.pause();stop=0}};</script>"
