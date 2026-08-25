from __future__ import annotations

import argparse
import os
import json
import atexit
from uuid import uuid4
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .action_detector import PunchDetector
from .exporter import export_package
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
    p.add_argument("--vlm", action="store_true", help="启用阿里云百炼 Qwen-VL"); p.add_argument("--vlm-model", default=os.getenv("BAILIAN_MODEL", "qwen3-vl-plus"))
    p.add_argument("--vlm-mode", choices=("off", "summary", "scan", "all"), default="all")
    p.add_argument("--vlm-scan-source", choices=("cv", "coarse"), default="cv", help="默认复核 CV 候选；coarse 会主动滑窗扫描整段视频")
    p.add_argument("--vlm-coarse-window", type=float, default=4.0); p.add_argument("--vlm-coarse-step", type=float, default=4.0)
    p.add_argument("--vlm-fine-padding", type=float, default=.6); p.add_argument("--vlm-max-candidates", type=int, default=0)
    p.add_argument("--vlm-media-mode", choices=("auto", "video", "images"), default="auto")
    p.add_argument("--report-dir", default=""); p.add_argument("--no-report", action="store_true")
    p.add_argument("--tracks", default="", help="轨迹 JSONL 路径，默认由 stats 文件名派生")
    p.add_argument("--no-export", action="store_true", help="关闭 CSV、热力图和片尾总结视频导出")
    p.add_argument("--export-dir", default="", help="完整结果包目录，默认位于 stats 同级")
    p.add_argument("--end-card-seconds", type=float, default=5.0, help="每张片尾总结页的秒数")
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
    if args.vlm_coarse_window <= 0 or args.vlm_coarse_step <= 0:
        raise ValueError("VLM 滑窗大小和步长必须大于 0")
    if args.vlm_fine_padding < 0 or args.end_card_seconds < 1:
        raise ValueError("VLM padding 必须非负，片尾每页必须至少 1 秒")
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception: device = "cpu"
    print(f"[boxing-mvp] device={device}, model={args.model}")
    tracker = FighterTracker(args.model, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    estimator = PoseEstimator()
    # API key may be present in a login shell. Never spend VLM quota unless the
    # caller explicitly opts in with --vlm.
    mode = args.vlm_mode if args.vlm else "off"
    reviewer = VLMReviewer(model=args.vlm_model, enabled=(mode != "off"), media_mode=args.vlm_media_mode)
    if mode != "off": print(f"[boxing-mvp] VLM={reviewer.model}, status={reviewer.status}, mode={mode}")
    stats_path = Path(args.stats).resolve()
    tracks_path = Path(args.tracks).resolve() if args.tracks else stats_path.with_name(f"{stats_path.stem}_tracks.jsonl")
    output_path = Path(args.output).resolve()
    export_dir = (Path(args.export_dir).resolve() if args.export_dir else
                  stats_path.parent / f"{stats_path.stem}_package")
    file_paths = {"input": input_path.resolve(), "output": output_path,
                  "stats": stats_path, "tracks": tracks_path}
    if len(set(file_paths.values())) != len(file_paths):
        raise ValueError(f"输入和输出路径必须互不相同: {file_paths}")
    if output_path.suffix.lower() != ".mp4" or stats_path.suffix.lower() != ".json" or tracks_path.suffix.lower() != ".jsonl":
        raise ValueError("output 必须是 .mp4，stats 必须是 .json，tracks 必须是 .jsonl")
    if export_dir in file_paths.values():
        raise ValueError("export_dir 不能与输入或输出文件路径相同")
    temporary_output = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp{output_path.suffix or '.mp4'}")
    temporary_tracks = tracks_path.with_name(f".{tracks_path.name}.{uuid4().hex}.tmp")
    for temporary in (temporary_output, temporary_tracks):
        atexit.register(lambda path=temporary: path.unlink(missing_ok=True))
    with VideoLoader(input_path, stride=args.stride) as video:
        writer = make_writer(temporary_output, video.info); detector = PunchDetector(video.info.fps); processed = 0
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_tracks.open("w", encoding="utf-8") as tracks:
            try:
                for frame_index, frame in video:
                    if args.max_frames and processed >= args.max_frames: break
                    people = tracker.update(frame)
                    # Unknown people are commonly referees or background staff.
                    # Keep them in tracks/overlays, but never emit punch events.
                    fighter_people = [person for person in people if person.side in {"red", "blue"}]
                    poses = [estimator.estimate(person) for person in fighter_people]
                    frame_events = detector.update_frame(frame_index, poses)
                    tracks.write(json.dumps({"frame_index": frame_index, "time_sec": frame_index / video.info.fps, "people": [
                        {"track_id": p.track_id, "side": p.side, "bbox": list(p.bbox), "confidence": p.confidence,
                         "keypoints": p.keypoints.tolist() if p.keypoints is not None else None,
                         "keypoint_conf": p.keypoint_conf.tolist() if p.keypoint_conf is not None else None} for p in people]}, ensure_ascii=False) + "\n")
                    for person in people:
                        pose = estimator.estimate(person)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, output_path)
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_tracks, tracks_path)
    scan_dir = stats_path.parent
    vlm_scan_path = stats_path.with_name(f"{stats_path.stem}_vlm_scan.json")
    vlm_result = run_scan(
        reviewer, str(input_path), duration, str(scan_dir), mode=mode,
        coarse_window=args.vlm_coarse_window, coarse_step=args.vlm_coarse_step,
        fine_padding=args.vlm_fine_padding, max_candidates=args.vlm_max_candidates,
        scan_source=args.vlm_scan_source, seed_candidates=cv_result.get("events", []),
        output_path=vlm_scan_path,
    )
    analysis = {"input": str(input_path.resolve()), "output": str(output_path), "duration_sec": duration, "fps": video.info.fps,
                "width": video.info.width, "height": video.info.height, "processed_frames": processed,
                **cv_result, "summary": vlm_result.get("summary", {}), "coarse_windows": vlm_result.get("coarse_windows", []),
                "coarse_candidates": vlm_result.get("coarse_candidates", []), "vlm_events": vlm_result.get("events", []),
                "vlm_rejected_events": vlm_result.get("rejected_events", []),
                "events": cv_result.get("events", []) + vlm_result.get("events", []), **reviewer.stats(), "tracks": str(tracks_path)}
    reviewed_count = len(vlm_result.get("events", [])) + len(vlm_result.get("rejected_events", []))
    analysis["quality_metrics"] = {
        "cv_candidates": len(cv_result.get("events", [])),
        "vlm_reviewed": reviewed_count,
        "vlm_accepted": len(vlm_result.get("events", [])),
        "vlm_rejected": len(vlm_result.get("rejected_events", [])),
        "vlm_acceptance_rate": round(len(vlm_result.get("events", [])) / reviewed_count, 4) if reviewed_count else None,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    def write_stats() -> None:
        temporary = stats_path.with_name(f".{stats_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, stats_path)
    write_stats()
    manifest = None
    if not args.no_export:
        manifest = export_package(
            analysis, stats_path, tracks_path, output_path, input_path, export_dir,
            end_card_seconds=max(1.0, args.end_card_seconds),
            make_report=not args.no_report,
        )
        analysis["exports"] = manifest
    if not args.no_report and args.no_export:
        report_dir = Path(args.report_dir) if args.report_dir else (export_dir / "report" if not args.no_export else stats_path.parent / "report")
        report_video = output_path
        write_report(str(report_dir), os.path.relpath(report_video, report_dir.resolve()), analysis)
    issues = []
    if reviewer.failures:
        issues.append({"kind": "vlm_failures", "count": reviewer.failures, "detail": reviewer.last_error})
    if manifest and manifest.get("status") != "completed":
        issues.append({"kind": "export_partial", "detail": manifest.get("errors", [])})
    warnings = manifest.get("warnings", []) if manifest else []
    issues.extend(warnings)
    analysis["analysis_status"] = "completed_with_warnings" if issues else "completed"
    analysis["issues"] = issues
    write_stats()
    print(json.dumps({"output": args.output, "stats": args.stats, "processed_frames": processed, "total_punches": analysis["total_punches"], "vlm_status": analysis["vlm_status"]}, ensure_ascii=False))


if __name__ == "__main__": main()
