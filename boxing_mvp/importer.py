from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .web.models import AnalysisRun, Artifact, Event, Round, Track, Video


def _event(run_id: str, item: dict[str, Any], index: int) -> Event:
    source = str(item.get("source", "cv"))
    external = str(item.get("event_id", item.get("id", index)))
    return Event(run_id=run_id, external_id=external, source=source,
        start_time_sec=_number(item.get("start_time_sec")), peak_time_sec=_number(item.get("peak_time_sec")),
        end_time_sec=_number(item.get("end_time_sec")), fighter=item.get("fighter", item.get("side")),
        hand=item.get("hand"), hit_or_miss=item.get("hit_or_miss"), blocked=item.get("blocked"),
        confidence=_number(item.get("confidence")), payload=item)


def _number(value: Any) -> float | None:
    try: return float(value) if value is not None else None
    except (TypeError, ValueError): return None


def import_analysis(db: Session, analysis_path: str | Path, tracks_path: str | Path | None = None,
                    video_path: str | Path | None = None, run_id: str | None = None) -> AnalysisRun:
    path = Path(analysis_path).resolve(); data = json.loads(path.read_text(encoding="utf-8"))
    video_file = Path(video_path or data.get("input", "")).resolve()
    video = db.scalar(select(Video).where(Video.path == str(video_file)))
    if video is None:
        video = Video(path=str(video_file), name=video_file.name, duration_sec=_number(data.get("duration_sec")), fps=_number(data.get("fps")))
        db.add(video); db.flush()
    run = db.get(AnalysisRun, run_id) if run_id else None
    if run is None: run = AnalysisRun(video_id=video.id, mode=str(data.get("vlm_status", "cv")), params=data); db.add(run); db.flush()
    else:
        for model in (Event, Track, Round, Artifact):
            db.query(model).filter(model.run_id == run.id).delete()
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    run.params = {**(run.params or {}), "summary": summary,
        "analysis_path": str(path), "duration_sec": data.get("duration_sec")}
    structures: list[tuple[str, dict[str, Any]]] = []
    for item in summary.get("rounds", []):
        if isinstance(item, dict): structures.append(("round", item))
    for item in summary.get("phases", []):
        if isinstance(item, dict): structures.append(("phase", item))
    # Older CV-only files keep aggregate round counters at the top level. Only
    # import entries that actually provide a time boundary.
    for item in data.get("rounds", []):
        if isinstance(item, dict) and (item.get("start_time_sec") is not None or item.get("end_time_sec") is not None):
            structures.append(("round", item))
    for no, (kind, item) in enumerate(structures, 1):
        payload = {"kind": kind, **item}
        db.add(Round(run_id=run.id, round_no=no,
            start_time_sec=_number(item.get("start_time_sec")), end_time_sec=_number(item.get("end_time_sec")),
            summary=item.get("summary", item.get("description", item.get("label"))), payload=payload))
    for index, item in enumerate(data.get("events", [])):
        db.add(_event(run.id, item, index))
    tracks_value = tracks_path or data.get("tracks") or path.parent / "tracks.jsonl"
    tracks = Path(tracks_value)
    invalid_track_rows = 0
    if tracks.exists():
        with tracks.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict) or "frame_index" not in item:
                        raise ValueError("track row missing frame_index")
                    people = item.get("people", [])
                    if not isinstance(people, list):
                        raise ValueError("track people must be a list")
                    db.add(Track(run_id=run.id, frame_index=int(item["frame_index"]),
                                 time_sec=_number(item.get("time_sec")), people=people))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    invalid_track_rows += 1
    if invalid_track_rows:
        run.params = {**(run.params or {}), "import_warnings": {"invalid_track_rows": invalid_track_rows}}
    artifacts = [("video", data.get("output")), ("analysis", str(path))]
    exports = data.get("exports") if isinstance(data.get("exports"), dict) else {}
    for kind, artifact in exports.get("files", {}).items():
        if kind == "manifest" and Path(str(artifact)).resolve() == path:
            continue
        artifacts.append((kind, artifact))
    if "report" not in exports.get("files", {}):
        artifacts.append(("report", str(path.parent / "report" / "index.html")))
    seen: set[str] = set()
    for kind, artifact in artifacts:
        if kind in seen:
            continue
        seen.add(kind)
        if artifact:
            artifact_path = Path(artifact).resolve()
            if artifact_path.exists():
                mime_type = {
                    ".csv": "text/csv", ".json": "application/json", ".png": "image/png",
                    ".mp4": "video/mp4", ".html": "text/html",
                }.get(artifact_path.suffix.lower())
                db.add(Artifact(run_id=run.id, kind=kind, path=str(artifact_path), mime_type=mime_type))
    db.commit(); db.refresh(run); return run
