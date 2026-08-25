from __future__ import annotations

import json
import os
import math
import re
from uuid import uuid4
from pathlib import Path
from typing import Any

from .vlm_reviewer import VLMReviewer


_UNSUPPORTED_METRIC = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[–—~至-]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:km/h|公里/小时|px/s|像素/秒|cm|厘米|km|公里|m|米)(?![A-Za-z])",
    re.IGNORECASE,
)


def sanitize_summary_response(response: dict[str, Any]) -> dict[str, Any]:
    """Remove monocular-video measurements while retaining an audit trail."""
    cleaned = json.loads(json.dumps(response, ensure_ascii=False))
    previous = cleaned.pop("sanitization", {})
    removed: list[dict[str, str]] = []
    if isinstance(previous, dict):
        for item in previous.get("removed_claims", []):
            if isinstance(item, dict):
                removed.append({"path": str(item.get("path", "summary")), "text": str(item.get("text", ""))})
            else:
                removed.append({"path": "summary", "text": str(item)})

    def clean(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {key: clean(item, f"{path}.{key}" if path else key) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if not isinstance(value, str) or not _UNSUPPORTED_METRIC.search(value):
            return value
        kept = []
        clauses = re.split(r"(?<=[。！？!?；;，,])", value)
        remove_next_comparison = False
        for clause in clauses:
            if _UNSUPPORTED_METRIC.search(clause):
                removed.append({"path": path, "text": clause.strip()})
                remove_next_comparison = True
            elif (remove_next_comparison and
                  re.match(r"^\s*(?:红方|蓝方|对方|另一方).*(?:为|是|约)\s*\d+(?:\.\d+)?\s*[。！？!?；;，,]?\s*$", clause)):
                removed.append({"path": path, "text": clause.strip()})
                remove_next_comparison = False
            else:
                kept.append(clause)
                remove_next_comparison = False
        return "".join(kept).strip().rstrip("；;,， ")

    cleaned = clean(cleaned, "")
    if removed:
        cleaned["sanitization"] = {
            "policy": "remove_unmeasurable_physical_metrics",
            "removed_claims": removed,
        }
    return cleaned


def windows(duration: float, size: float, step: float) -> list[tuple[float, float]]:
    if duration <= 0 or size <= 0 or step <= 0:
        return []
    result = []
    start = 0.0
    while start < duration:
        result.append((start, min(duration, start + size)))
        start += step
    return result


def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: x.get("start_time_sec", 0)):
        match = next((old for old in merged if old.get("fighter") == item.get("fighter") and old.get("hand") == item.get("hand") and item["start_time_sec"] <= old["end_time_sec"] + 0.4), None)
        if match:
            match["end_time_sec"] = max(match["end_time_sec"], item["end_time_sec"])
            match["confidence"] = max(match.get("confidence", 0), item.get("confidence", 0))
        else:
            merged.append(dict(item))
    return merged


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate(item: dict[str, Any], duration: float) -> dict[str, Any] | None:
    start = max(0.0, min(duration, _number(item.get("start_time_sec"))))
    end = max(start, min(duration, _number(item.get("end_time_sec"), start)))
    if end <= start:
        return None
    return {**item, "start_time_sec": start, "end_time_sec": end}


def _strict_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _interval_score(start: float, peak: float, end: float,
                    candidate_start: float, candidate_end: float) -> tuple[float, float]:
    overlap = max(0.0, min(end, candidate_end) - max(start, candidate_start))
    center_distance = abs(peak - (candidate_start + candidate_end) / 2)
    return overlap, -center_distance


def _validated_event(response: dict[str, Any], window_start: float,
                     window_end: float, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return a formal event or a stable machine-readable rejection reason."""
    if response.get("review_status") != "reviewed":
        return None, "review_not_completed"
    if response.get("is_punch") not in {"yes", "uncertain"}:
        return None, "not_a_punch"
    outcome = response.get("hit_or_miss")
    if outcome not in {"hit", "miss", "blocked", "uncertain"}:
        return None, "invalid_outcome"
    confidence = _number(response.get("confidence"), -1.0)
    if not 0.0 <= confidence <= 1.0:
        return None, "invalid_confidence"
    is_punch = response.get("is_punch")
    if is_punch == "uncertain" and outcome != "uncertain":
        return None, "uncertain_punch_with_decisive_outcome"
    if outcome in {"hit", "miss", "blocked"}:
        if confidence < 0.7:
            return None, "confidence_below_formal_threshold"
        if response.get("evidence") != "direct":
            return None, "non_direct_evidence"
    if outcome == "hit" and response.get("contact_evidence") != "clear":
        return None, "hit_contact_not_clear"
    if outcome == "hit" and response.get("blocked") not in {"no", None}:
        return None, "hit_marked_blocked"
    if outcome == "miss" and response.get("contact_evidence") not in {"none", None}:
        return None, "miss_contact_conflict"
    if outcome == "blocked" and response.get("blocked") != "yes":
        return None, "blocked_not_confirmed"
    if outcome != "uncertain" and response.get("fighter") not in {"red", "blue"}:
        return None, "unknown_fighter"
    raw = [_strict_number(response.get(key)) for key in ("start_time_sec", "peak_time_sec", "end_time_sec")]
    if any(value is None for value in raw):
        return None, "invalid_timestamps"
    raw_start, raw_peak, raw_end = raw
    duration = window_end - window_start
    interpretations = []
    if 0 <= raw_start <= raw_peak <= raw_end <= duration + 1e-6:
        interpretations.append((window_start + raw_start, window_start + raw_peak,
                                window_start + raw_end, "relative"))
    if window_start - 1e-6 <= raw_start <= raw_peak <= raw_end <= window_end + 1e-6:
        interpretations.append((raw_start, raw_peak, raw_end, "absolute"))
    if not interpretations:
        return None, "timestamps_outside_review_window"
    candidate_start = _number(candidate.get("start_time_sec"), window_start)
    candidate_end = _number(candidate.get("end_time_sec"), window_end)
    start, peak, end, basis = max(
        interpretations,
        key=lambda values: _interval_score(values[0], values[1], values[2], candidate_start, candidate_end),
    )
    if end - start < 0.10:
        return None, "event_too_short"
    if min(end, candidate_end) - max(start, candidate_start) <= 0:
        return None, "no_candidate_overlap"
    event = dict(response)
    event.update({"start_time_sec": start, "peak_time_sec": peak, "end_time_sec": end,
                  "time_basis_detected": basis})
    return event, ""


def _accepted_event(response: dict[str, Any], window_start: float,
                    window_end: float, candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Backward-compatible event-only view used by callers and tests."""
    return _validated_event(response, window_start, window_end, candidate)[0]


def _temporal_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    intersection = max(0.0, min(a["end_time_sec"], b["end_time_sec"]) -
                       max(a["start_time_sec"], b["start_time_sec"]))
    union = max(a["end_time_sec"], b["end_time_sec"]) - min(a["start_time_sec"], b["start_time_sec"])
    return intersection / union if union > 0 else 0.0


def _deduplicate_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["start_time_sec"], -_number(item.get("confidence")))):
        match = next((old for old in accepted
            if old.get("fighter") == event.get("fighter")
            and _temporal_iou(old, event) >= 0.60
            and abs(_number(old.get("peak_time_sec")) - _number(event.get("peak_time_sec"))) <= 0.12), None)
        if match is None:
            accepted.append(event)
            continue
        if _number(event.get("confidence")) > _number(match.get("confidence")):
            accepted.remove(match); accepted.append(event); rejected = match
        else:
            rejected = event
        duplicates.append({"candidate": rejected.get("candidate"), "response": rejected,
                           "rejection_reason": "duplicate_temporal_event"})
    accepted.sort(key=lambda item: item["start_time_sec"])
    return accepted, duplicates


def run_scan(reviewer: VLMReviewer, video_path: str, duration: float, output_dir: str,
             mode: str = "all", coarse_window: float = 4.0, coarse_step: float = 1.0,
             fine_padding: float = 0.6, max_candidates: int = 0,
             scan_source: str = "cv",
             seed_candidates: list[dict[str, Any]] | None = None,
             output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"mode": mode, "scan_source": scan_source, "duration_sec": duration, "summary": {}, "coarse_windows": [], "coarse_candidates": [], "events": [], "rejected_events": [], **reviewer.stats()}
    if mode in ("summary", "all") and reviewer.enabled:
        reviewer.upload_video(video_path)
        result["summary"] = sanitize_summary_response(reviewer.summarize_video(duration))
    elif mode in ("summary", "all"):
        result["summary"] = {"review_status": "not_reviewed", "summary": "未启用 VLM 或未提供 BAILIAN_API_KEY"}
    if mode in ("scan", "all") and reviewer.enabled:
        reviewer.upload_video(video_path)
        candidates: list[dict[str, Any]] = []
        if scan_source == "coarse":
            for start, end in windows(duration, coarse_window, coarse_step):
                response = reviewer.coarse_scan(start, end)
                result["coarse_windows"].append({"start_time_sec": start, "end_time_sec": end, **response})
                if response.get("review_status") != "reviewed" or not isinstance(response.get("events"), list):
                    continue
                for event in response["events"]:
                    if not isinstance(event, dict):
                        continue
                    item = _candidate({**event,
                        "start_time_sec": start + _number(event.get("relative_start_sec")),
                        "end_time_sec": start + _number(event.get("relative_end_sec"), _number(event.get("relative_start_sec"))),
                    }, duration)
                    if item:
                        candidates.append(item)
            result["coarse_candidates"] = merge_candidates(candidates)
        else:
            for item in seed_candidates or []:
                if isinstance(item, dict):
                    normalized = _candidate(item, duration)
                    if normalized:
                        candidates.append(normalized)
            result["coarse_candidates"] = candidates
        selected = result["coarse_candidates"][:max_candidates] if max_candidates else result["coarse_candidates"]
        for candidate in selected:
            start = max(0.0, candidate["start_time_sec"] - fine_padding)
            end = min(duration, candidate["end_time_sec"] + fine_padding)
            response = reviewer.fine_scan(start, end, candidate)
            candidate_fighter = candidate.get("fighter", candidate.get("side", "unknown"))
            response_fighter = response.get("fighter", "unknown")
            if (response.get("review_status") == "reviewed"
                    and candidate_fighter in {"red", "blue"}
                    and response_fighter in {"red", "blue"}
                    and candidate_fighter != response_fighter):
                result["rejected_events"].append({
                    "candidate": candidate, "response": response,
                    "rejection_reason": "fighter_identity_mismatch",
                })
                continue
            event, rejection_reason = _validated_event(response, start, end, candidate)
            if event is None:
                result["rejected_events"].append({
                    "candidate": candidate, "response": response,
                    "rejection_reason": rejection_reason,
                })
                continue
            event.update({"source": "vlm_scan", "candidate": candidate})
            result["events"].append(event)
        result["events"], duplicates = _deduplicate_events(result["events"])
        result["rejected_events"].extend(duplicates)
    result.update(reviewer.stats())
    scan_path = Path(output_path) if output_path else out / "vlm_scan.json"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = scan_path.with_name(f".{scan_path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, scan_path)
    return result
