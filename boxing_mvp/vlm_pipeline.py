from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .vlm_reviewer import VLMReviewer


def windows(duration: float, size: float, step: float) -> list[tuple[float, float]]:
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


def run_scan(reviewer: VLMReviewer, video_path: str, duration: float, output_dir: str,
             mode: str = "all", coarse_window: float = 4.0, coarse_step: float = 1.0,
             fine_padding: float = 0.6, max_candidates: int = 0) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"mode": mode, "duration_sec": duration, "summary": {}, "coarse_windows": [], "coarse_candidates": [], "events": [], **reviewer.stats()}
    if mode in ("summary", "all") and reviewer.enabled:
        reviewer.upload_video(video_path)
        result["summary"] = reviewer.summarize_video(duration)
    elif mode in ("summary", "all"):
        result["summary"] = {"review_status": "not_reviewed", "summary": "未提供 VLM_API_KEY"}
    if mode in ("scan", "all") and reviewer.enabled:
        reviewer.upload_video(video_path)
        candidates = []
        for start, end in windows(duration, coarse_window, coarse_step):
            response = reviewer.coarse_scan(start, end)
            result["coarse_windows"].append({"start_time_sec": start, "end_time_sec": end, **response})
            for event in response.get("events", []):
                candidates.append({**event, "start_time_sec": start + float(event.get("relative_start_sec", 0)), "end_time_sec": start + float(event.get("relative_end_sec", event.get("relative_start_sec", 0)))} )
        result["coarse_candidates"] = merge_candidates(candidates)
        selected = result["coarse_candidates"][:max_candidates] if max_candidates else result["coarse_candidates"]
        for candidate in selected:
            start = max(0.0, candidate["start_time_sec"] - fine_padding)
            end = min(duration, candidate["end_time_sec"] + fine_padding)
            response = reviewer.fine_scan(start, end, candidate)
            event = dict(response)
            for key in ("start_time_sec", "peak_time_sec", "end_time_sec"):
                if key in event:
                    event[key] = start + float(event[key])
            event.update({"source": "vlm_scan", "candidate": candidate})
            result["events"].append(event)
    result.update(reviewer.stats())
    (out / "vlm_scan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
