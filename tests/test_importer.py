from __future__ import annotations

import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from boxing_mvp.importer import import_analysis
from boxing_mvp.web.models import Artifact, Base, Round


def test_imports_summary_rounds_and_phases(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({
        "input": str(video), "duration_sec": 20, "fps": 25, "events": [],
        "summary": {
            "summary": "test",
            "rounds": [{"start_time_sec": 0, "end_time_sec": 10, "description": "第一回合"}],
            "phases": [{"start_time_sec": 10, "end_time_sec": 20, "label": "对攻期"}],
        },
    }), encoding="utf-8")

    with Session(engine) as db:
        run = import_analysis(db, analysis)
        rows = list(db.scalars(select(Round).where(Round.run_id == run.id).order_by(Round.round_no)))
        assert [row.payload["kind"] for row in rows] == ["round", "phase"]
        assert [row.summary for row in rows] == ["第一回合", "对攻期"]
        assert run.params["summary"]["summary"] == "test"
        assert "analysis" not in run.params


def test_imports_export_artifacts(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    video = tmp_path / "video.mp4"; video.write_bytes(b"")
    events = tmp_path / "events.csv"; events.write_text("event_index\n", encoding="utf-8")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({
        "input": str(video), "events": [],
        "exports": {"files": {"events_csv": str(events)}},
    }), encoding="utf-8")
    with Session(engine) as db:
        run = import_analysis(db, analysis)
        artifact = db.scalar(select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == "events_csv"))
        assert artifact is not None
        assert artifact.mime_type == "text/csv"


def test_import_skips_bad_track_rows_and_records_warning(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    video = tmp_path / "video.mp4"; video.write_bytes(b"")
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text('{bad}\n{"frame_index":1,"time_sec":0.1,"people":[]}\n', encoding="utf-8")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"input": str(video), "events": [], "tracks": str(tracks)}), encoding="utf-8")
    with Session(engine) as db:
        run = import_analysis(db, analysis)
        assert len(run.tracks) == 1
        assert run.params["import_warnings"]["invalid_track_rows"] == 1
