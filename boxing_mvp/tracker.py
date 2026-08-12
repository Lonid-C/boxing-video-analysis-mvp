from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    """Keep red/blue labels stable using first reliable left/right positions."""

    side_by_track: dict[int, str] = field(default_factory=dict)
    side_by_x: dict[str, float] = field(default_factory=dict)

    def assign(self, track_id: int, center_x: float, frame_width: int) -> str:
        if track_id in self.side_by_track:
            return self.side_by_track[track_id]
        # Fixed-camera MVP convention: left fighter = red, right fighter = blue.
        # The labels are intentionally configurable in the output, not inferred
        # from clothing color, because jersey color is unreliable under lighting.
        if not self.side_by_x:
            side = "red" if center_x < frame_width / 2 else "blue"
        else:
            available = {"red", "blue"} - set(self.side_by_x)
            side = next(iter(available)) if available else (
                "red" if center_x < frame_width / 2 else "blue"
            )
        self.side_by_track[track_id] = side
        self.side_by_x.setdefault(side, center_x)
        return side


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
        people: list[TrackPerson] = []
        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            cx = (x1 + x2) / 2.0
            side = self.identities.assign(int(track_id), cx, frame.shape[1])
            people.append(TrackPerson(
                track_id=int(track_id), bbox=(x1, y1, x2, y2),
                confidence=float(confs[i]), side=side,
                keypoints=xy[i] if xy is not None and i < len(xy) else None,
                keypoint_conf=kc[i] if kc is not None and i < len(kc) else None,
            ))
        return people
