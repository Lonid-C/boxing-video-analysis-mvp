from __future__ import annotations

import csv
import json
import pytest

from boxing_mvp.exporter import (
    AUDIT_FIELDS, EVENT_FIELDS, SUMMARY_FIELDS, event_rows, fighter_summary_rows, movement_rows,
    review_audit_rows,
    write_heatmap, _file_metadata, _write_csv,
)
import cv2
import numpy as np


def test_event_csv_has_bom_fixed_columns_and_cv_fighter_fallback(tmp_path):
    rows = event_rows({"events": [{
        "source": "cv", "event_id": 7, "side": "red", "punch_type": "直拳",
        "reason": "包含,逗号\n以及换行",
    }]})
    path = tmp_path / "events.csv"
    _write_csv(path, EVENT_FIELDS, rows)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.DictReader(stream))
    assert list(parsed[0]) == EVENT_FIELDS
    assert parsed[0]["fighter"] == "red"
    assert parsed[0]["reason"] == "包含,逗号\n以及换行"


def test_movement_prefers_ankles_and_falls_back_to_bbox(tmp_path):
    points = [[0, 0] for _ in range(17)]
    points[15], points[16] = [20, 80], [40, 80]
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text("\n".join([
        json.dumps({"frame_index": 1, "time_sec": .04, "people": [{
            "track_id": 1, "side": "red", "bbox": [10, 10, 50, 90],
            "keypoints": points, "keypoint_conf": [1] * 17, "confidence": .9,
        }]}),
        json.dumps({"frame_index": 2, "time_sec": .08, "people": [{
            "track_id": 2, "side": "blue", "bbox": [100, 20, 140, 95],
            "keypoints": None, "confidence": .8,
        }]}),
    ]), encoding="utf-8")
    rows = movement_rows(tracks, 200, 100)
    assert rows[0]["foot_source"] == "ankles"
    assert rows[0]["foot_x"] == 30
    assert rows[1]["foot_source"] == "bbox_bottom"
    assert rows[1]["foot_y"] == 95


def test_summary_counts_only_vlm_outcomes_for_hit_rate():
    events = [
        {"source": "cv", "side": "red", "hit_or_miss": "hit"},
        {"source": "vlm_scan", "fighter": "red", "hit_or_miss": "hit", "confidence": .9},
        {"source": "vlm_scan", "fighter": "red", "hit_or_miss": "blocked", "confidence": .8},
        {"source": "vlm_scan", "fighter": "red", "hit_or_miss": "uncertain", "confidence": .5},
    ]
    row = fighter_summary_rows(events, [
        {"fighter": "red", "normalized_foot_x": .5, "normalized_foot_y": .5},
    ], [{"candidate": {"side": "red"}, "rejection_reason": "non_direct_evidence"}])[0]
    assert list(row) == SUMMARY_FIELDS
    assert row["cv_candidates"] == 1
    assert row["vlm_reviewed"] == 4
    assert row["vlm_accepted"] == 3
    assert row["vlm_rejected"] == 1
    assert row["vlm_acceptance_rate"] == .75
    assert row["hit"] == 1
    assert row["decidable_events"] == 2
    assert row["hit_rate"] == .5


def test_heatmap_and_empty_placeholder_have_video_dimensions(tmp_path):
    background = np.full((120, 200, 3), 180, dtype=np.uint8)
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    write_heatmap(red, "red", [{"fighter": "red", "foot_x": 30, "foot_y": 80}], background)
    write_heatmap(blue, "blue", [], background)
    assert cv2.imread(str(red)).shape == (120, 200, 3)
    assert cv2.imread(str(blue)).shape == (120, 200, 3)


def test_movement_skips_bad_jsonl_rows_and_reports_issues(tmp_path):
    path = tmp_path / "tracks.jsonl"
    path.write_text('{bad json}\n{"frame_index":1,"people":[]}\n[]\n', encoding="utf-8")
    issues = []
    assert movement_rows(path, 100, 100, issues) == []
    assert [issue["line"] for issue in issues] == [1, 3]


def test_atomic_export_keeps_previous_package_on_failure(monkeypatch, tmp_path):
    import boxing_mvp.exporter as exporter

    output = tmp_path / "package"
    output.mkdir(); (output / "sentinel.txt").write_text("stable", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "package_type": "boxing_analysis_export", "files": {}, "source_video": "/video.mp4",
    }), encoding="utf-8")
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(exporter, "_export_package_into", fail)
    with pytest.raises(RuntimeError, match="injected"):
        exporter.export_package({}, tmp_path / "a.json", tmp_path / "t.jsonl",
                                tmp_path / "a.mp4", tmp_path / "s.mp4", output)
    assert (output / "sentinel.txt").read_text() == "stable"
    assert not list(tmp_path.glob(".package.*.staging"))


def test_manifest_file_metadata_detects_content_changes(tmp_path):
    path = tmp_path / "events.csv"; path.write_bytes(b"first")
    first = _file_metadata({"events_csv": str(path)})["events_csv"]
    path.write_bytes(b"second")
    second = _file_metadata({"events_csv": str(path)})["events_csv"]
    assert first["size_bytes"] == 5
    assert second["size_bytes"] == 6
    assert first["sha256"] != second["sha256"]


def test_event_rows_ignore_non_object_values():
    assert event_rows({"events": [None, "bad", {"side": "red"}]})[0]["fighter"] == "red"


def test_review_audit_separates_formal_and_rejected_events():
    rows = review_audit_rows({
        "vlm_events": [{"candidate": {"event_id": 1, "side": "red"},
                        "review_status": "reviewed", "fighter": "red", "hit_or_miss": "hit"}],
        "vlm_rejected_events": [{"candidate": {"event_id": 2, "side": "blue"},
                                 "response": {"review_status": "reviewed", "fighter": "red"},
                                 "rejection_reason": "fighter_identity_mismatch"}],
    })
    assert list(rows[0]) == AUDIT_FIELDS
    assert rows[0]["accepted"] == "yes"
    assert rows[1]["accepted"] == "no"
    assert rows[1]["rejection_reason"] == "fighter_identity_mismatch"


def test_refuses_to_replace_unmanaged_directory(tmp_path):
    import boxing_mvp.exporter as exporter
    output = tmp_path / "important"; output.mkdir()
    (output / "user-data.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError, match="unmanaged"):
        exporter.export_package({}, tmp_path / "a.json", tmp_path / "t.jsonl",
                                tmp_path / "a.mp4", tmp_path / "s.mp4", output)
    assert (output / "user-data.txt").read_text() == "do not delete"
