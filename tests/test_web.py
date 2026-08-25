from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

from fastapi.testclient import TestClient

from boxing_mvp.web.app import app
from boxing_mvp.web.db import Base, engine
from boxing_mvp.web.models import AnalysisRun, Artifact, Video
from boxing_mvp.web.db import SessionLocal


def test_uploads_with_same_name_do_not_overwrite(monkeypatch, tmp_path):
    import boxing_mvp.web.app as app_module

    monkeypatch.setattr(app_module, "UPLOAD_ROOT", tmp_path)
    Base.metadata.create_all(engine)
    source = tmp_path / "valid.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8)); writer.release()
    payload = source.read_bytes()
    with TestClient(app) as client:
        first = client.post("/api/v1/uploads", files={"file": ("fight.mp4", payload, "video/mp4")})
        second = client.post("/api/v1/uploads", files={"file": ("fight.mp4", payload, "video/mp4")})
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["path"] != second.json()["path"]
    assert Path(first.json()["path"]).read_bytes() == payload
    assert Path(second.json()["path"]).read_bytes() == payload
    assert first.json()["width"] == 64


def test_invalid_video_upload_is_removed(monkeypatch, tmp_path):
    import boxing_mvp.web.app as app_module
    monkeypatch.setattr(app_module, "UPLOAD_ROOT", tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/v1/uploads", files={"file": ("bad.mp4", b"not-video", "video/mp4")})
    assert response.status_code == 422
    assert not list(tmp_path.glob("bad-*"))
    assert not list(tmp_path.glob("*.part"))


def test_artifact_endpoint_returns_media_url(tmp_path, monkeypatch):
    import boxing_mvp.web.app as app_module

    monkeypatch.setattr(app_module, "MEDIA_ROOT", tmp_path)
    artifact_file = tmp_path / "events.csv"
    artifact_file.write_text("a,b\n", encoding="utf-8")
    with SessionLocal() as db:
        video = Video(path=str(tmp_path / "v.mp4"), name="v.mp4")
        db.add(video); db.flush()
        run = AnalysisRun(video_id=video.id, status="completed")
        db.add(run); db.flush()
        db.add(Artifact(run_id=run.id, kind="events_csv", path=str(artifact_file), mime_type="text/csv"))
        db.commit(); run_id = run.id
    with TestClient(app) as client:
        response = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert response.status_code == 200
    assert response.json()[0]["media_url"] == "/media/events.csv"


def test_lifespan_marks_interrupted_runs_failed():
    unique = uuid4().hex
    with SessionLocal() as db:
        video = Video(path=f"/tmp/restart-test-{unique}.mp4", name=f"restart-test-{unique}.mp4")
        db.add(video); db.flush()
        run = AnalysisRun(video_id=video.id, status="running")
        db.add(run); db.commit(); run_id = run.id
    with TestClient(app):
        pass
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        assert run.status == "failed"
        assert "restart" in run.error
