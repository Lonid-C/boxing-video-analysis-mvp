from __future__ import annotations
import subprocess
import threading
import os
import sys
from pathlib import Path
from threading import Lock
from .web.db import SessionLocal
from .web.models import AnalysisRun
from .importer import import_analysis

_threads: dict[str, threading.Thread] = {}
_threads_lock = Lock()
_run_slots = threading.BoundedSemaphore(max(1, int(os.getenv("BOXING_MAX_CONCURRENT_RUNS", "1"))))

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
    acquired = False
    try:
        _run_slots.acquire()
        acquired = True
        _set_status(run_id, "running")
        python = os.getenv("BOXING_PYTHON", sys.executable)
        project_root = Path(os.getenv("BOXING_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
        run_root = Path(os.getenv("BOXING_RUN_ROOT", project_root / "outputs" / "runs")).resolve() / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        output = Path(params.get("output", run_root / "annotated.mp4")).resolve()
        stats = Path(params.get("stats", run_root / "analysis.json")).resolve()
        tracks = stats.with_name(f"{stats.stem}_tracks.jsonl")
        export_dir = Path(params.get("export_dir", run_root / "result_package")).resolve()
        log_path = run_root / "run.log"
        cmd = [python, "-m", "boxing_mvp.main", "--input", video_path, "--output", str(output),
               "--stats", str(stats), "--tracks", str(tracks), "--no-display"]
        if mode in {"summary", "scan", "all"} or params.get("vlm"):
            cmd += ["--vlm", "--vlm-mode", mode]
        for key, flag in (("vlm_media_mode", "--vlm-media-mode"), ("vlm_model", "--vlm-model")):
            if key in params: cmd += [flag, str(params[key])]
        if "vlm_scan_source" in params:
            cmd += ["--vlm-scan-source", str(params["vlm_scan_source"])]
        if params.get("export_package") is False:
            cmd += ["--no-export"]
        else:
            cmd += ["--export-dir", str(export_dir)]
        if "end_card_seconds" in params:
            cmd += ["--end-card-seconds", str(params["end_card_seconds"])]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(cmd, cwd=project_root, check=False,
                timeout=int(params.get("timeout", 7200)), stdout=log, stderr=subprocess.STDOUT, text=True)
        if completed.returncode:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-1800:] or "analysis process failed"
            raise RuntimeError(detail)
        db = SessionLocal()
        try:
            import_analysis(db, stats, video_path=video_path, run_id=run_id)
            run = db.get(AnalysisRun, run_id)
            if run:
                import json
                result = json.loads(stats.read_text(encoding="utf-8"))
                run.status = result.get("analysis_status", "completed")
                run.error = json.dumps(result.get("issues", []), ensure_ascii=False)[:2000] or None
                db.commit()
        finally:
            db.close()
    except Exception as exc:
        _set_status(run_id, "failed", str(exc)[:2000])
    finally:
        if acquired:
            _run_slots.release()
        with _threads_lock:
            _threads.pop(run_id, None)

def start_run(run: AnalysisRun, video_path: str) -> None:
    with _threads_lock:
        existing = _threads.get(run.id)
        if existing and existing.is_alive():
            return
        thread = threading.Thread(target=execute_run, args=(run.id, video_path, run.mode, run.params or {}),
                                  daemon=True, name=f"boxing-run-{run.id}")
        _threads[run.id] = thread
        thread.start()
