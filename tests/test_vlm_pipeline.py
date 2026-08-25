from __future__ import annotations

import json
import pytest

from boxing_mvp.vlm_pipeline import run_scan, windows


class FakeReviewer:
    enabled = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.fine_calls = []

    def stats(self):
        return {"vlm_status": "enabled", "vlm_calls": len(self.fine_calls)}

    def upload_video(self, path):
        return {"status": "ready"}

    def fine_scan(self, start, end, candidate):
        self.fine_calls.append((start, end, candidate))
        return next(self.responses)


def test_windows_rejects_invalid_values():
    assert windows(10, 0, 1) == []
    assert windows(10, 4, 0) == []
    assert windows(0, 4, 1) == []


def test_summary_removes_unmeasurable_metric_claims_with_audit_trail():
    from boxing_mvp.vlm_pipeline import sanitize_summary_response
    value = sanitize_summary_response({
        "summary": "蓝方持续压迫。双方距离在0.88–2.13m间波动，蓝方占据内圈。",
        "key_moments": [{"description": "蓝方出拳；手套峰值4687px/s，红方为0"}],
    })
    assert value["summary"] == "蓝方持续压迫。蓝方占据内圈。"
    assert value["key_moments"][0]["description"] == "蓝方出拳"
    assert value["sanitization"]["removed_claims"] == [
        {"path": "summary", "text": "双方距离在0.88–2.13m间波动，"},
        {"path": "key_moments[0].description", "text": "手套峰值4687px/s，"},
        {"path": "key_moments[0].description", "text": "红方为0"},
    ]


def test_cv_candidates_are_fine_scanned_and_invalid_results_rejected(tmp_path):
    reviewer = FakeReviewer([
        {"review_status": "reviewed", "is_punch": "yes", "hit_or_miss": "hit",
         "fighter": "red", "confidence": 0.9, "evidence": "direct",
         "contact_evidence": "clear", "blocked": "no",
         "start_time_sec": 0.7, "peak_time_sec": 0.9, "end_time_sec": 1.1},
        {"review_status": "reviewed", "is_punch": "no", "hit_or_miss": "uncertain",
         "confidence": 0.2, "start_time_sec": 0.1, "peak_time_sec": 0.15, "end_time_sec": 0.2},
    ])
    result = run_scan(reviewer, "video.mp4", 10, str(tmp_path), mode="scan",
        scan_source="cv", seed_candidates=[
            {"start_time_sec": 2.0, "end_time_sec": 2.5, "side": "red"},
            {"start_time_sec": 5.0, "end_time_sec": 5.4, "side": "blue"},
        ])

    assert len(reviewer.fine_calls) == 2
    assert len(result["events"]) == 1
    assert result["events"][0]["start_time_sec"] == pytest.approx(2.1)
    assert len(result["rejected_events"]) == 1
    assert json.loads((tmp_path / "vlm_scan.json").read_text())["scan_source"] == "cv"


def test_detects_absolute_time_and_rejects_zero_length():
    from boxing_mvp.vlm_pipeline import _accepted_event
    candidate = {"start_time_sec": 5.0, "end_time_sec": 5.8}
    response = {"review_status": "reviewed", "is_punch": "yes", "hit_or_miss": "hit",
        "fighter": "red", "confidence": .9, "evidence": "direct",
        "contact_evidence": "clear", "blocked": "no",
        "start_time_sec": 5.1, "peak_time_sec": 5.3, "end_time_sec": 5.5}
    event = _accepted_event(response, 4.4, 6.4, candidate)
    assert event["time_basis_detected"] == "absolute"
    assert event["start_time_sec"] == 5.1
    response.update({"start_time_sec": 5.3, "peak_time_sec": 5.3, "end_time_sec": 5.3})
    assert _accepted_event(response, 4.4, 6.4, candidate) is None


def test_deduplicates_same_temporal_action():
    from boxing_mvp.vlm_pipeline import _deduplicate_events
    base = {"fighter": "blue", "start_time_sec": 1.0, "peak_time_sec": 1.2,
            "end_time_sec": 1.4, "confidence": .8}
    better = {**base, "hand": "rear", "confidence": .95, "candidate": {"event_id": 2}}
    events, rejected = _deduplicate_events([{**base, "hand": "front"}, better])
    assert events == [better]
    assert rejected[0]["rejection_reason"] == "duplicate_temporal_event"


def test_rejects_fighter_identity_mismatch(tmp_path):
    reviewer = FakeReviewer([{
        "review_status": "reviewed", "is_punch": "yes", "hit_or_miss": "hit",
        "fighter": "blue", "confidence": .95, "evidence": "direct",
        "contact_evidence": "clear", "blocked": "no", "start_time_sec": .7,
        "peak_time_sec": .9, "end_time_sec": 1.1,
    }])
    result = run_scan(reviewer, "video.mp4", 4, str(tmp_path), mode="scan",
        seed_candidates=[{"side": "red", "start_time_sec": 1.0, "end_time_sec": 1.5}])
    assert result["events"] == []
    assert result["rejected_events"][0]["rejection_reason"] == "fighter_identity_mismatch"


@pytest.mark.parametrize("changes", [
    {"contact_evidence": "possible"},
    {"evidence": "inferred"},
    {"confidence": .6},
    {"blocked": "yes"},
    {"is_punch": "uncertain"},
])
def test_hit_requires_clear_direct_consistent_evidence(changes):
    from boxing_mvp.vlm_pipeline import _accepted_event
    response = {"review_status": "reviewed", "is_punch": "yes", "fighter": "red",
        "hit_or_miss": "hit", "blocked": "no", "contact_evidence": "clear",
        "evidence": "direct", "confidence": .95, "start_time_sec": 1.1,
        "peak_time_sec": 1.3, "end_time_sec": 1.5}
    response.update(changes)
    candidate = {"side": "red", "start_time_sec": 1.0, "end_time_sec": 1.6}
    assert _accepted_event(response, .5, 2.2, candidate) is None
