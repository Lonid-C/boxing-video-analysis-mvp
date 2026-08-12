from __future__ import annotations
import subprocess
import threading
from pathlib import Path
from .web.db import SessionLocal
from .web.models import AnalysisRun
from .importer import import_analysis

def _set_status(run_id: str, status: str, error: str | None = None) -> None:
    db = SessionLocal()
    try:
        run = db.get(AnalysisRun, run_id)
        if run:
            run.status = status
            run.error = error
            db.commit()
    finally:
        db.close()

def execute_run(run_id: str, video_path: str, mode: str, params: dict) -> None:
    _set_status(run_id, "running")
    try:
        output = Path(params.get("output", Path(video_path).with_name(Path(video_path).stem + "_analysis.mp4")))
        stats = Path(params.get("stats", output.with_suffix(".json")))
        cmd = ["/home/files/anaconda3/envs/birdclef2026/bin/python", "-m", "boxing_mvp.main", "--input", video_path, "--output", str(output), "--stats", str(stats), "--no-display"]
        if mode in {"summary", "scan", "all"} or params.get("vlm"):
            cmd += ["--vlm", "--vlm-mode", mode]
        for key, flag in (("vlm_media_mode", "--vlm-media-mode"), ("vlm_model", "--vlm-model")):
            if key in params: cmd += [flag, str(params[key])]
        subprocess.run(cmd, cwd="/home/files/boxing_mvp", check=True, timeout=int(params.get("timeout", 7200)), capture_output=True, text=True)
        db = SessionLocal()
        try:
            import_analysis(db, stats, video_path=video_path, run_id=run_id)
            run = db.get(AnalysisRun, run_id)
            run.status = "completed"
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        _set_status(run_id, "failed", str(exc)[:2000])

def start_run(run: AnalysisRun, video_path: str) -> None:
    threading.Thread(target=execute_run, args=(run.id, video_path, run.mode, run.params or {}), daemon=True).start()
