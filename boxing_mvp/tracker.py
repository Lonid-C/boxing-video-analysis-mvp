from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import cv2


@dataclass
class TrackPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    side: str = "unknown"
    keypoints: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None


@dataclass
class IdentityAssigner:
    """Keep two fighter labels stable and leave extra people (for example a referee) unknown."""

    side_by_track: dict[int, str] = field(default_factory=dict)
    side_by_x: dict[str, float] = field(default_factory=dict)
    missing_frames: dict[int, int] = field(default_factory=dict)
    reassign_after: int = 12
    color_by_track: dict[int, dict[str, float]] = field(default_factory=dict)
    color_threshold: float = 0.07
    color_margin: float = 1.5
    min_color_samples: int = 2
    conflict_release_after: int = 3
    conflict_frames: dict[int, int] = field(default_factory=dict)

    def assign_frame(self, detections: list[tuple[int, float, float, float, float]]) -> dict[int, str]:
        """Assign a frame at once so red/blue cannot be duplicated."""
        present_ids = {track_id for track_id, *_ in detections}
        for track_id in list(self.side_by_track):
            if track_id in present_ids:
                self.missing_frames[track_id] = 0
            else:
                self.missing_frames[track_id] = self.missing_frames.get(track_id, 0) + 1
        # Hold an identity briefly across tracker dropouts, then release the old
        # ID so a replacement ID can be associated with the last known x.
        for track_id in list(self.side_by_track):
            if self.missing_frames.get(track_id, 0) > self.reassign_after:
                self.side_by_track.pop(track_id, None)
                self.missing_frames.pop(track_id, None)
                self.color_by_track.pop(track_id, None)
                self.conflict_frames.pop(track_id, None)
        released: set[int] = set()
        for track_id, x, score, red_score, blue_score in detections:
            colors = self.color_by_track.setdefault(track_id, {"red": 0.0, "blue": 0.0, "samples": 0.0})
            alpha = 0.25 if colors["samples"] else 1.0
            colors["red"] = colors["red"] * (1 - alpha) + red_score * alpha
            colors["blue"] = colors["blue"] * (1 - alpha) + blue_score * alpha
            colors["samples"] += 1
            side = self.side_by_track.get(track_id)
            if side in {"red", "blue"}:
                other = "blue" if side == "red" else "red"
                own_score = red_score if side == "red" else blue_score
                other_score = blue_score if side == "red" else red_score
                conflict = (other_score >= self.color_threshold and
                            other_score >= max(own_score * self.color_margin, self.color_threshold))
                self.conflict_frames[track_id] = self.conflict_frames.get(track_id, 0) + 1 if conflict else 0
                if self.conflict_frames[track_id] >= self.conflict_release_after:
                    self.side_by_track.pop(track_id, None)
                    self.missing_frames.pop(track_id, None)
                    self.conflict_frames[track_id] = 0
                    released.add(track_id)
        result = {track_id: self.side_by_track.get(track_id, "unknown")
                  for track_id, *_ in detections}
        reserved = set(self.side_by_track.values()) & {"red", "blue"}
        newcomers = []
        for track_id, x, score, *_ in detections:
            colors = self.color_by_track[track_id]
            if (result[track_id] == "unknown" and track_id not in released
                    and colors["samples"] >= self.min_color_samples):
                newcomers.append((track_id, x, score, colors["red"], colors["blue"]))
        available = {"red", "blue"} - reserved

        for side in ("red", "blue"):
            if side not in available:
                continue
            other = "blue" if side == "red" else "red"
            eligible = [item for item in newcomers
                        if item[3 if side == "red" else 4] >= self.color_threshold
                        and item[3 if side == "red" else 4] >=
                            max(item[4 if side == "red" else 3] * self.color_margin, self.color_threshold)]
            if not eligible:
                continue
            anchor = self.side_by_x.get(side)
            # Stronger color evidence wins; recent x is a tiebreaker for an ID
            # switch while both old and new boxes coexist briefly.
            chosen = max(eligible, key=lambda item: (
                item[3 if side == "red" else 4],
                -abs(item[1] - anchor) if anchor is not None else item[2],
            ))
            newcomers.remove(chosen)
            track_id, x = chosen[0], chosen[1]
            self.side_by_track[track_id] = side
            self.missing_frames[track_id] = 0
            self.side_by_x[side] = x
            result[track_id] = side

        for track_id, x, *_ in detections:
            side = result[track_id]
            if side in {"red", "blue"}:
                old = self.side_by_x.get(side, x)
                self.side_by_x[side] = old * 0.9 + x * 0.1
        return result

    @staticmethod
    def color_evidence(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Return saturated red/blue ratios from the fighter's upper body."""
        x1, y1, x2, y2 = bbox
        width, height = max(x2 - x1, 1), max(y2 - y1, 1)
        xa, xb = max(0, int(x1 + .12 * width)), min(frame.shape[1], int(x2 - .12 * width))
        ya, yb = max(0, int(y1 + .05 * height)), min(frame.shape[0], int(y1 + .82 * height))
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            return 0.0, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        visible = (saturation > 90) & (value > 45)
        red = (((hue < 12) | (hue > 168)) & visible).mean()
        blue = ((hue > 92) & (hue < 135) & visible).mean()
        return float(red), float(blue)


class FighterTracker:
    """Ultralytics YOLO Pose tracking wrapper using ByteTrack.

    Pose models expose the person boxes and COCO keypoints in one inference;
    this avoids running a separate detector and pose estimator in the MVP.
    """

    def __init__(self, model_path: str, device: str = "auto", conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 640) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("缺少 ultralytics，请先运行 setup_server.sh") from exc
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.identities = IdentityAssigner()

    def update(self, frame: np.ndarray) -> list[TrackPerson]:
        kwargs: dict[str, Any] = {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "classes": [0],  # person
            "verbose": False,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        result = self.model.track(frame, **kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        keypoints = getattr(result, "keypoints", None)
        xy = keypoints.xy.detach().cpu().numpy() if keypoints is not None else None
        kc = keypoints.conf.detach().cpu().numpy() if keypoints is not None else None
        detections = []
        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            red_score, blue_score = self.identities.color_evidence(frame, (x1, y1, x2, y2))
            detections.append((int(track_id), (x1 + x2) / 2.0,
                               max(0, x2 - x1) * max(0, y2 - y1) * float(confs[i]),
                               red_score, blue_score))
        sides = self.identities.assign_frame(detections)
        people: list[TrackPerson] = []
        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            side = sides.get(int(track_id), "unknown")
            people.append(TrackPerson(
                track_id=int(track_id), bbox=(x1, y1, x2, y2),
                confidence=float(confs[i]), side=side,
                keypoints=xy[i] if xy is not None and i < len(xy) else None,
                keypoint_conf=kc[i] if kc is not None and i < len(kc) else None,
            ))
        return people
