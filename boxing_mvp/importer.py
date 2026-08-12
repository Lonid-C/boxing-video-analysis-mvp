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
        db.query(Event).filter(Event.run_id == run.id).delete(); db.query(Track).filter(Track.run_id == run.id).delete()
    rounds = data.get("rounds", [])
    for item in rounds:
        no = int(item.get("round", len(run.rounds) + 1))
        if not db.scalar(select(Round).where(Round.run_id == run.id, Round.round_no == no)):
            db.add(Round(run_id=run.id, round_no=no, start_time_sec=_number(item.get("start_time_sec")), end_time_sec=_number(item.get("end_time_sec")), summary=item.get("summary"), payload=item))
    for index, item in enumerate(data.get("events", [])):
        db.add(_event(run.id, item, index))
    tracks = Path(tracks_path) if tracks_path else path.parent / "tracks.jsonl"
    if tracks.exists():
        for line in tracks.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line); db.add(Track(run_id=run.id, frame_index=int(item["frame_index"]), time_sec=_number(item.get("time_sec")), people=item.get("people", [])))
    for kind, artifact in (("video", data.get("output")), ("analysis", str(path)), ("report", str(path.parent / "report" / "index.html"))):
        if artifact and not db.scalar(select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == kind)):
            db.add(Artifact(run_id=run.id, kind=kind, path=str(Path(artifact).resolve()), mime_type=None))
    db.commit(); db.refresh(run); return run
