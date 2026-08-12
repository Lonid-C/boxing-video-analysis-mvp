from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .action_detector import PunchDetector
from .pose_estimator import PoseEstimator
from .report import write_report
from .tracker import FighterTracker
from .video_loader import VideoLoader, make_writer
from .vlm_pipeline import run_scan
from .vlm_reviewer import VLMReviewer

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
COLORS = {"red": (50, 70, 230), "blue": (230, 100, 40), "unknown": (180, 180, 180)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="固定机位拳击视频自动分析 MVP")
    p.add_argument("--input", required=True); p.add_argument("--output", default="outputs/output.mp4")
    p.add_argument("--stats", default="outputs/stats.json"); p.add_argument("--model", default="models/yolo11n-pose.pt")
    p.add_argument("--device", default="auto"); p.add_argument("--conf", type=float, default=.35); p.add_argument("--iou", type=float, default=.5)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--stride", type=int, default=1); p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--round-seconds", type=float, default=180.0); p.add_argument("--no-display", action="store_true")
    p.add_argument("--vlm", action="store_true", help="启用 Gemini VLM"); p.add_argument("--vlm-model", default="gemini-2.5-flash")
    p.add_argument("--vlm-mode", choices=("off", "summary", "scan", "all"), default="all")
    p.add_argument("--vlm-coarse-window", type=float, default=4.0); p.add_argument("--vlm-coarse-step", type=float, default=1.0)
    p.add_argument("--vlm-fine-padding", type=float, default=.6); p.add_argument("--vlm-max-candidates", type=int, default=0)
    p.add_argument("--vlm-media-mode", choices=("auto", "video", "images"), default="auto")
    p.add_argument("--report-dir", default=""); p.add_argument("--no-report", action="store_true")
    return p.parse_args()


def draw_pose(frame: np.ndarray, points: np.ndarray | None, confidence: np.ndarray | None, color: tuple[int, int, int]) -> None:
    if points is None: return
    for a, b in SKELETON:
        if a < len(points) and b < len(points) and (confidence is None or (float(confidence[a]) >= .25 and float(confidence[b]) >= .25)):
            cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2, cv2.LINE_AA)
    for i, point in enumerate(points):
        if confidence is None or float(confidence[i]) >= .25:
            cv2.circle(frame, tuple(point.astype(int)), 3, color, -1, cv2.LINE_AA)


def put_text(frame: np.ndarray, text: str, xy: tuple[int, int], color=(240, 240, 240), scale=.55) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args(); input_path = Path(args.input)
    if not input_path.exists(): raise FileNotFoundError(f"输入视频不存在: {input_path}")
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception: device = "cpu"
    print(f"[boxing-mvp] device={device}, model={args.model}")
    tracker = FighterTracker(args.model, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    estimator = PoseEstimator()
    mode = "off" if not args.vlm and args.vlm_mode == "off" else args.vlm_mode
    reviewer = VLMReviewer(model=args.vlm_model, enabled=(mode != "off"), media_mode=args.vlm_media_mode)
    if mode != "off": print(f"[boxing-mvp] VLM={reviewer.model}, status={reviewer.status}, mode={mode}")
    tracks_path = Path(args.stats).parent / "tracks.jsonl"
    with VideoLoader(input_path, stride=args.stride) as video:
        writer = make_writer(args.output, video.info); detector = PunchDetector(video.info.fps); processed = 0
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        with tracks_path.open("w", encoding="utf-8") as tracks:
            try:
                for frame_index, frame in video:
                    if args.max_frames and processed >= args.max_frames: break
                    people = tracker.update(frame); poses = [estimator.estimate(person) for person in people]
                    frame_events = detector.update_frame(frame_index, poses)
                    tracks.write(json.dumps({"frame_index": frame_index, "time_sec": frame_index / video.info.fps, "people": [
                        {"track_id": p.track_id, "side": p.side, "bbox": list(p.bbox), "confidence": p.confidence,
                         "keypoints": p.keypoints.tolist() if p.keypoints is not None else None,
                         "keypoint_conf": p.keypoint_conf.tolist() if p.keypoint_conf is not None else None} for p in people]}, ensure_ascii=False) + "\n")
                    for person, pose in zip(people, poses):
                        color = COLORS.get(person.side, COLORS["unknown"]); x1, y1, x2, y2 = person.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2); draw_pose(frame, pose.points, pose.confidence, color)
                        put_text(frame, f"{person.side.upper()} ID:{person.track_id}", (x1, max(20, y1 - 8)), color)
                    put_text(frame, f"FRAME {frame_index}  PUNCHES {len(detector.events) + len(frame_events)}", (18, 28))
                    for i, event in enumerate(frame_events):
                        put_text(frame, f"PUNCH {event.side.upper()} {event.hand} {event.punch_type} -> {event.target_region}", (18, 56 + i * 24), COLORS.get(event.side, COLORS["unknown"]), .5)
                    writer.write(frame); processed += 1
                    if not args.no_display:
                        cv2.imshow("boxing-mvp", frame)
                        if cv2.waitKey(1) & 0xff == ord("q"): break
            finally:
                writer.release(); cv2.destroyAllWindows()
        cv_result = detector.stats(video.info.fps, args.round_seconds)
        duration = video.info.duration_seconds
    scan_dir = Path(args.stats).parent
    vlm_result = run_scan(reviewer, str(input_path), duration, str(scan_dir), mode=mode, coarse_window=args.vlm_coarse_window, coarse_step=args.vlm_coarse_step, fine_padding=args.vlm_fine_padding, max_candidates=args.vlm_max_candidates)
    analysis = {"input": str(input_path), "output": str(args.output), "duration_sec": duration, "fps": video.info.fps, "processed_frames": processed,
                **cv_result, "summary": vlm_result.get("summary", {}), "coarse_windows": vlm_result.get("coarse_windows", []),
                "coarse_candidates": vlm_result.get("coarse_candidates", []), "vlm_events": vlm_result.get("events", []),
                "events": cv_result.get("events", []) + vlm_result.get("events", []), **reviewer.stats(), "tracks": str(tracks_path)}
    stats_path = Path(args.stats); stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_report:
        report_dir = args.report_dir or str(stats_path.parent / "report")
        write_report(report_dir, Path(args.output).name, analysis)
    print(json.dumps({"output": args.output, "stats": args.stats, "processed_frames": processed, "total_punches": analysis["total_punches"], "vlm_status": analysis["vlm_status"]}, ensure_ascii=False))


if __name__ == "__main__": main()
