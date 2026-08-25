from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import shutil
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EVENT_FIELDS = [
    "event_index", "source", "candidate_id", "start_time_sec", "peak_time_sec",
    "end_time_sec", "fighter", "hand", "punch_type", "target_region",
    "impact_area", "hit_or_miss", "blocked", "block_type", "reaction", "power",
    "part_of_combo", "occluded", "confidence", "evidence", "reason",
]
SUMMARY_FIELDS = [
    "fighter", "cv_candidates", "vlm_reviewed", "vlm_accepted", "vlm_rejected",
    "vlm_acceptance_rate", "hit", "miss", "blocked",
    "uncertain", "decidable_events", "hit_rate", "average_confidence",
    "left_hand", "right_hand", "front_hand", "rear_hand", "straight", "hook", "swing", "uppercut",
    "head_targets", "torso_targets", "movement_samples", "primary_area",
]
AUDIT_FIELDS = [
    "candidate_id", "candidate_fighter", "candidate_hand", "candidate_start_time_sec",
    "candidate_end_time_sec", "review_status", "accepted", "rejection_reason",
    "response_fighter", "response_hand", "is_punch", "hit_or_miss",
    "start_time_sec", "peak_time_sec", "end_time_sec", "confidence", "evidence",
    "contact_evidence", "blocked", "occluded", "reason",
]
TRACK_FIELDS = [
    "frame_index", "time_sec", "track_id", "fighter", "bbox_x1", "bbox_y1",
    "bbox_x2", "bbox_y2", "center_x", "center_y", "foot_x", "foot_y",
    "normalized_foot_x", "normalized_foot_y", "confidence", "foot_source",
]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def event_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    events = analysis.get("events") if isinstance(analysis.get("events"), list) else []
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            continue
        candidate = event.get("candidate") if isinstance(event.get("candidate"), dict) else {}
        values = {key: event.get(key, "") for key in EVENT_FIELDS if key not in {"event_index", "source", "candidate_id"}}
        values["fighter"] = event.get("fighter", event.get("side", ""))
        rows.append({
            "event_index": index,
            "source": event.get("source", "cv"),
            "candidate_id": candidate.get("event_id", event.get("event_id", "")),
            **values,
        })
    return rows


def review_audit_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every fine-review decision without mixing rejections into formal events."""
    rows: list[dict[str, Any]] = []
    items: list[tuple[dict[str, Any], dict[str, Any], bool, str]] = []
    accepted = analysis.get("vlm_events") if isinstance(analysis.get("vlm_events"), list) else []
    for response in accepted:
        if isinstance(response, dict):
            candidate = response.get("candidate") if isinstance(response.get("candidate"), dict) else {}
            items.append((candidate, response, True, ""))
    rejected = analysis.get("vlm_rejected_events") if isinstance(analysis.get("vlm_rejected_events"), list) else []
    for item in rejected:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        response = item.get("response") if isinstance(item.get("response"), dict) else {}
        items.append((candidate, response, False, str(item.get("rejection_reason") or "unspecified_rejection")))
    for candidate, response, is_accepted, rejection_reason in items:
        rows.append({
            "candidate_id": candidate.get("event_id", ""),
            "candidate_fighter": candidate.get("fighter", candidate.get("side", "")),
            "candidate_hand": candidate.get("hand", ""),
            "candidate_start_time_sec": candidate.get("start_time_sec", ""),
            "candidate_end_time_sec": candidate.get("end_time_sec", ""),
            "review_status": response.get("review_status", ""),
            "accepted": "yes" if is_accepted else "no",
            "rejection_reason": rejection_reason,
            "response_fighter": response.get("fighter", ""),
            "response_hand": response.get("hand", ""),
            **{key: response.get(key, "") for key in AUDIT_FIELDS
               if key not in {"candidate_id", "candidate_fighter", "candidate_hand",
                              "candidate_start_time_sec", "candidate_end_time_sec",
                              "review_status", "accepted", "rejection_reason",
                              "response_fighter", "response_hand"}},
        })
    return rows


def _valid_keypoint(person: dict[str, Any], index: int) -> tuple[float, float] | None:
    points, confidence = person.get("keypoints"), person.get("keypoint_conf")
    if not isinstance(points, list) or index >= len(points) or not isinstance(points[index], list):
        return None
    if isinstance(confidence, list) and index < len(confidence) and _number(confidence[index]) < 0.25:
        return None
    point = points[index]
    if len(point) < 2:
        return None
    x, y = _number(point[0], -1), _number(point[1], -1)
    return (x, y) if x >= 0 and y >= 0 else None


def movement_rows(tracks_path: str | Path, width: int, height: int,
                  issues: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    path = Path(tracks_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
      for line_number, line in enumerate(stream, 1):
        try:
            if not line.strip():
                continue
            frame = json.loads(line)
            if not isinstance(frame, dict) or not isinstance(frame.get("people", []), list):
                raise ValueError("track row must be an object with a people list")
        except (json.JSONDecodeError, ValueError) as exc:
            if issues is not None:
                issues.append({"line": line_number, "error": str(exc)[:300]})
            continue
        for person in frame.get("people", []):
            if not isinstance(person, dict):
                continue
            fighter = person.get("side")
            if fighter not in {"red", "blue"}:
                continue
            bbox = person.get("bbox") or [0, 0, 0, 0]
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = map(_number, bbox[:4])
            feet = [point for point in (_valid_keypoint(person, 15), _valid_keypoint(person, 16)) if point]
            if feet:
                foot_x = sum(point[0] for point in feet) / len(feet)
                foot_y = sum(point[1] for point in feet) / len(feet)
                source = "ankles"
            else:
                foot_x, foot_y, source = (x1 + x2) / 2, y2, "bbox_bottom"
            foot_x = min(max(foot_x, 0.0), max(width - 1, 0))
            foot_y = min(max(foot_y, 0.0), max(height - 1, 0))
            rows.append({
                "frame_index": frame.get("frame_index", ""), "time_sec": frame.get("time_sec", ""),
                "track_id": person.get("track_id", ""), "fighter": fighter,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
                "foot_x": foot_x, "foot_y": foot_y,
                "normalized_foot_x": foot_x / max(width, 1),
                "normalized_foot_y": foot_y / max(height, 1),
                "confidence": person.get("confidence", ""), "foot_source": source,
            })
    return rows


def fighter_summary_rows(events: list[dict[str, Any]], tracks: list[dict[str, Any]],
                         rejected_events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    events = [event for event in events if isinstance(event, dict)]
    rejected_events = [event for event in (rejected_events or []) if isinstance(event, dict)]
    output = []
    for fighter in ("red", "blue"):
        own = [event for event in events if event.get("fighter", event.get("side")) == fighter]
        cv_events = [event for event in own if event.get("source", "cv") == "cv"]
        vlm_events = [event for event in own if event.get("source") == "vlm_scan"]
        rejected = [item for item in rejected_events
                    if isinstance(item.get("candidate"), dict)
                    and item["candidate"].get("fighter", item["candidate"].get("side")) == fighter]
        reviewed = len(vlm_events) + len(rejected)
        outcomes = Counter(event.get("hit_or_miss") for event in vlm_events)
        decidable = outcomes["hit"] + outcomes["miss"] + outcomes["blocked"]
        confidences = [_number(event.get("confidence"), -1) for event in vlm_events]
        confidences = [value for value in confidences if value >= 0]
        detail_events = vlm_events if vlm_events else cv_events
        hands = Counter(event.get("hand") for event in detail_events)
        types = Counter(event.get("punch_type") for event in detail_events)
        targets = Counter(event.get("target_region") for event in detail_events)
        own_tracks = [row for row in tracks if row["fighter"] == fighter]
        if own_tracks:
            areas = Counter()
            for track in own_tracks:
                x, y = track["normalized_foot_x"], track["normalized_foot_y"]
                horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
                vertical = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "middle"
                areas[f"{vertical}_{horizontal}"] += 1
            primary_area = areas.most_common(1)[0][0]
        else:
            primary_area = "unknown"
        output.append({
            "fighter": fighter, "cv_candidates": len(cv_events), "vlm_reviewed": reviewed,
            "vlm_accepted": len(vlm_events), "vlm_rejected": len(rejected),
            "vlm_acceptance_rate": round(len(vlm_events) / reviewed, 4) if reviewed else "",
            "hit": outcomes["hit"], "miss": outcomes["miss"], "blocked": outcomes["blocked"],
            "uncertain": outcomes["uncertain"], "decidable_events": decidable,
            "hit_rate": round(outcomes["hit"] / decidable, 4) if decidable else "",
            "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else "",
            "left_hand": hands["left"], "right_hand": hands["right"],
            "front_hand": hands["front"], "rear_hand": hands["rear"],
            "straight": types["直拳"], "hook": types["勾拳"], "swing": types["摆拳"],
            "uppercut": types["上勾拳"], "head_targets": targets["head"],
            "torso_targets": targets["torso"], "movement_samples": len(own_tracks),
            "primary_area": primary_area,
        })
    return output


def _middle_frame(video_path: str | Path, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(frame, (width, height))


def write_heatmap(path: Path, fighter: str, rows: list[dict[str, Any]], background: np.ndarray) -> None:
    height, width = background.shape[:2]
    points = [(int(row["foot_x"]), int(row["foot_y"])) for row in rows if row["fighter"] == fighter]
    dim = np.clip(background.astype(np.float32) * 0.38, 0, 255).astype(np.uint8)
    if not points:
        cv2.putText(dim, f"{fighter.upper()}: insufficient tracking data", (30, max(50, height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 1200), (255, 255, 255), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(path), dim):
            raise RuntimeError(f"failed to write heatmap: {path}")
        return
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1
    sigma = max(7.0, min(width, height) / 35.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
    nonzero = density[density > 0]
    scale = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    normalized = np.clip(density / max(scale, 1e-9) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    alpha = (normalized.astype(np.float32) / 255.0 * 0.82)[..., None]
    result = (dim * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.putText(result, f"{fighter.upper()} movement density | samples={len(points)}", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1400), (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), result):
        raise RuntimeError(f"failed to write heatmap: {path}")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    configured = os.getenv("BOXING_CJK_FONT")
    candidates = [configured] if configured else []
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrapped(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
             max_width: int, max_lines: int) -> list[str]:
    lines, current = [], ""
    for char in str(text).replace("\n", " "):
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = char
            if len(lines) >= max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines and sum(len(line) for line in lines) < len(str(text)):
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def _end_cards(out: Path, width: int, height: int, summary_rows: list[dict[str, Any]],
               analysis: dict[str, Any]) -> list[Path]:
    title_size, body_size = max(24, height // 18), max(17, height // 31)
    title_font, body_font, bold_font = _font(title_size, True), _font(body_size), _font(body_size, True)
    cards = []
    canvas = Image.new("RGB", (width, height), "#0b1020")
    draw = ImageDraw.Draw(canvas)
    draw.text((width * .06, height * .07), "比赛分析总结", font=title_font, fill="#ffffff")
    for index, row in enumerate(summary_rows):
        x = width * (.07 if index == 0 else .54)
        color = "#ff6b6b" if row["fighter"] == "red" else "#60a5fa"
        draw.text((x, height * .22), f"{row['fighter'].upper()} 方", font=bold_font, fill=color)
        values = [
            f"CV 候选：{row['cv_candidates']}    VLM 复核：{row['vlm_reviewed']}",
            f"正式事件：{row['vlm_accepted']}    未通过门禁：{row['vlm_rejected']}",
            f"命中 {row['hit']}  未中 {row['miss']}  格挡 {row['blocked']}  不确定 {row['uncertain']}",
            f"可判定命中率：{row['hit_rate'] if row['hit_rate'] != '' else '暂无'}",
            f"左/右手：{row['left_hand']} / {row['right_hand']}    前/后手：{row['front_hand']} / {row['rear_hand']}",
        ]
        for line_no, line in enumerate(values):
            draw.text((x, height * (.31 + line_no * .09)), line, font=body_font, fill="#dbeafe")
    draw.text((width * .06, height * .88), "CV 候选不等同于命中；命中统计仅使用 VLM 可判定事件。", font=body_font, fill="#94a3b8")
    card = out / "end_card_1.png"; canvas.save(card); cards.append(card)

    canvas = Image.new("RGB", (width, height), "#0b1020")
    draw = ImageDraw.Draw(canvas)
    draw.text((width * .05, height * .05), "移动热区与整场摘要", font=title_font, fill="#ffffff")
    image_y, image_h = int(height * .16), int(height * .43)
    image_w = int(width * .40)
    for index, fighter in enumerate(("red", "blue")):
        heat = Image.open(out / f"heatmap_{fighter}.png").convert("RGB")
        heat.thumbnail((image_w, image_h))
        x = int(width * (.05 if index == 0 else .51))
        canvas.paste(heat, (x, image_y))
    summary = analysis.get("summary") if isinstance(analysis.get("summary"), dict) else {}
    summary_text = summary.get("summary") or "本次分析未生成 VLM 整场摘要。"
    y = int(height * .64)
    for line in _wrapped(draw, summary_text, body_font, int(width * .88), 3):
        draw.text((width * .06, y), line, font=body_font, fill="#e2e8f0")
        y += int(body_size * 1.45)
    moments = summary.get("key_moments") if isinstance(summary.get("key_moments"), list) else []
    phases = summary.get("phases") if isinstance(summary.get("phases"), list) else []
    details = []
    if phases:
        details.append("阶段：" + " / ".join(str(item.get("label", "")) for item in phases[:3] if isinstance(item, dict)))
    if moments and isinstance(moments[0], dict):
        details.append(f"关键时刻 {moments[0].get('time_sec', '?')}s：{moments[0].get('description', '')}")
    for detail in details[:2]:
        for line in _wrapped(draw, detail, body_font, int(width * .88), 1):
            draw.text((width * .06, y), line, font=body_font, fill="#bfdbfe")
            y += int(body_size * 1.4)
    draw.text((width * .06, height * .91), "热力图使用视频画面坐标，不代表真实场地米制位置。", font=body_font, fill="#94a3b8")
    card = out / "end_card_2.png"; canvas.save(card); cards.append(card)
    return cards


def _ffmpeg_executable() -> str:
    configured = os.getenv("BOXING_FFMPEG")
    if configured:
        return configured
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("缺少 imageio-ffmpeg，无法生成保留音频的总结视频") from exc


def _run(command: list[str], timeout: int = 1800) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-1600:])


def _has_audio(ffmpeg: str, video_path: str | Path) -> bool:
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video_path), "-map", "0:a:0", "-t", "0.1", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    return completed.returncode == 0


def _validate_video(path: Path, expected_duration: float, ffmpeg: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("summary video was not created")
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _ = capture.read()
    capture.release()
    duration = frames / fps if fps > 0 else 0
    tolerance = max(0.5, expected_duration * 0.03)
    if not ok or frames <= 0 or abs(duration - expected_duration) > tolerance:
        raise RuntimeError(f"summary video validation failed: duration={duration:.3f}, expected={expected_duration:.3f}")
    if not _has_audio(ffmpeg, path):
        raise RuntimeError("summary video validation failed: audio stream missing")


def _file_metadata(files: dict[str, str], exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    metadata = {}
    for kind, raw_path in files.items():
        if exclude and kind in exclude:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata[kind] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return metadata


def _managed_package(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (value.get("package_type") == "boxing_analysis_export" or
            (isinstance(value.get("files"), dict) and value.get("source_video")))


def write_summary_video(out: Path, annotated_video: str | Path, source_video: str | Path,
                        cards: list[Path], seconds: float, fps: float) -> Path:
    ffmpeg = _ffmpeg_executable()
    segments, body, concat = [], out / "body_with_audio.mp4", out / "concat.txt"
    try:
        for index, card in enumerate(cards, 1):
            segment = out / f"end_card_{index}.mp4"
            _run([ffmpeg, "-y", "-loop", "1", "-i", str(card), "-f", "lavfi", "-i",
                  "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", str(seconds),
                  "-r", str(max(fps, 1)), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-shortest", str(segment)])
            segments.append(segment)
        if _has_audio(ffmpeg, source_video):
            body_command = [ffmpeg, "-y", "-i", str(annotated_video), "-i", str(source_video),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-af", "apad", "-ar", "48000", "-ac", "2", "-shortest", str(body)]
        else:
            capture = cv2.VideoCapture(str(annotated_video))
            body_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
            body_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            capture.release()
            duration = body_frames / max(body_fps, 1)
            if duration <= 0:
                raise RuntimeError("annotated video has no readable frames")
            body_command = [ffmpeg, "-y", "-i", str(annotated_video), "-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", str(duration),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(body)]
        _run(body_command)
        concat.write_text("\n".join(f"file '{path.name}'" for path in [body, *segments]) + "\n", encoding="utf-8")
        final = out / "summary_video.mp4"
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
              "-c", "copy", "-movflags", "+faststart", str(final)])
        capture = cv2.VideoCapture(str(annotated_video))
        body_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
        body_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        _validate_video(final, body_frames / max(body_fps, 1) + seconds * len(cards), ffmpeg)
        return final
    finally:
        for temporary in [body, concat, *segments]:
            temporary.unlink(missing_ok=True)


def _export_package_into(analysis: dict[str, Any], analysis_path: str | Path,
                         tracks_path: str | Path, annotated_video: str | Path,
                         source_video: str | Path, out: Path,
                         final_out: Path, end_card_seconds: float,
                         make_video: bool, make_report: bool) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "package_type": "boxing_analysis_export", "schema_version": "2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "files": {}, "errors": [],
    }

    def attempt(kind: str, function):
        try:
            path = function()
            manifest["files"][kind] = str(Path(path).resolve())
            return path
        except Exception as exc:
            manifest["status"] = "partial"
            manifest["errors"].append({"kind": kind, "error": str(exc)[:1200]})
            return None

    source = cv2.VideoCapture(str(source_video))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or analysis.get("width") or 1280)
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or analysis.get("height") or 720)
    fps = float(source.get(cv2.CAP_PROP_FPS) or analysis.get("fps") or 25)
    source.release()
    events = event_rows(analysis)
    audit = review_audit_rows(analysis)
    track_issues: list[dict[str, Any]] = []
    tracks = movement_rows(tracks_path, width, height, track_issues)
    if track_issues:
        manifest["warnings"] = [{"kind": "invalid_track_rows", "count": len(track_issues),
                                 "examples": track_issues[:10]}]
    events_value = analysis.get("events") if isinstance(analysis.get("events"), list) else []
    rejected_value = analysis.get("vlm_rejected_events") if isinstance(analysis.get("vlm_rejected_events"), list) else []
    summaries = fighter_summary_rows(events_value, tracks, rejected_value)
    attempt("events_csv", lambda: (_write_csv(out / "events.csv", EVENT_FIELDS, events) or out / "events.csv"))
    attempt("review_audit_csv", lambda: (_write_csv(out / "review_audit.csv", AUDIT_FIELDS, audit) or out / "review_audit.csv"))
    attempt("fighter_summary_csv", lambda: (_write_csv(out / "fighter_summary.csv", SUMMARY_FIELDS, summaries) or out / "fighter_summary.csv"))
    attempt("movement_tracks_csv", lambda: (_write_csv(out / "movement_tracks.csv", TRACK_FIELDS, tracks) or out / "movement_tracks.csv"))
    background = _middle_frame(source_video, width, height)
    for fighter in ("red", "blue"):
        attempt(f"heatmap_{fighter}", lambda fighter=fighter: (write_heatmap(out / f"heatmap_{fighter}.png", fighter, tracks, background) or out / f"heatmap_{fighter}.png"))
    try:
        cards = _end_cards(out, width, height, summaries, analysis)
        for index, card in enumerate(cards, 1):
            manifest["files"][f"end_card_{index}"] = str(card.resolve())
    except Exception as exc:
        cards = []
        manifest["status"] = "partial"
        manifest["errors"].append({"kind": "end_cards", "error": str(exc)[:1200]})
    if make_video and cards:
        attempt("summary_video", lambda: write_summary_video(out, annotated_video, source_video, cards, end_card_seconds, fps))
    analysis_copy = out / "analysis.json"
    manifest["files"]["analysis"] = str(analysis_copy)
    manifest.update({"source_video": str(Path(source_video).resolve()), "annotated_video": str(Path(annotated_video).resolve()),
                     "analysis_path": str(Path(analysis_path).resolve()), "end_card_seconds": end_card_seconds,
                     "vlm_model": analysis.get("vlm_model"), "vlm_calls": analysis.get("vlm_calls", 0),
                     "vlm_failures": analysis.get("vlm_failures", 0)})
    manifest["fighter_summary"] = summaries
    manifest_path = out / "manifest.json"
    manifest["files"]["manifest"] = str(manifest_path)
    if make_report:
        try:
            from .report import write_report
            report_dir = out / "report"
            report_video = Path(manifest["files"].get("summary_video", annotated_video)).resolve()
            report_analysis = {**analysis, "exports": manifest}
            report_path = write_report(str(report_dir), os.path.relpath(report_video, report_dir), report_analysis)
            manifest["files"]["report"] = str(Path(report_path).resolve())
        except Exception as exc:
            manifest["status"] = "partial"
            manifest["errors"].append({"kind": "report", "error": str(exc)[:1200]})
    package_issues = []
    if int(analysis.get("vlm_failures") or 0):
        package_issues.append({"kind": "vlm_failures", "count": int(analysis["vlm_failures"]),
                               "detail": analysis.get("vlm_last_error", "")})
    if manifest["status"] != "completed":
        package_issues.append({"kind": "export_partial", "detail": manifest["errors"]})
    package_issues.extend(manifest.get("warnings", []))
    package_analysis = {**analysis,
        "analysis_status": "completed_with_warnings" if package_issues else "completed",
        "issues": package_issues,
    }
    # Keep the package copy independent of the manifest so it has a stable
    # checksum without a self-referential hash cycle.
    analysis_copy.write_text(json.dumps(package_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["file_metadata"] = _file_metadata(manifest["files"], exclude={"manifest"})
    staging_prefix, final_prefix = str(out), str(final_out)
    manifest = json.loads(json.dumps(manifest).replace(staging_prefix, final_prefix))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def export_package(analysis: dict[str, Any], analysis_path: str | Path,
                   tracks_path: str | Path, annotated_video: str | Path,
                   source_video: str | Path, output_dir: str | Path,
                   end_card_seconds: float = 5.0, make_video: bool = True,
                   make_report: bool = True) -> dict[str, Any]:
    """Build a complete package in staging and publish it as one directory.

    Readers see either the previous complete package or the new complete
    package; they never observe half-written CSV/video files.
    """
    final_out = Path(output_dir).resolve()
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if final_out.exists() and (not final_out.is_dir() or not _managed_package(final_out)):
        raise ValueError(f"refusing to replace unmanaged export directory: {final_out}")
    staging = final_out.with_name(f".{final_out.name}.{uuid4().hex}.staging")
    backup = final_out.with_name(f".{final_out.name}.{uuid4().hex}.backup")
    try:
        manifest = _export_package_into(
            analysis, analysis_path, tracks_path, annotated_video, source_video,
            staging, final_out, end_card_seconds, make_video, make_report,
        )
        if final_out.exists():
            os.replace(final_out, backup)
        try:
            os.replace(staging, final_out)
        except Exception:
            if backup.exists():
                os.replace(backup, final_out)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
