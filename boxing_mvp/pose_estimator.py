from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Ultralytics/COCO pose indices.
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class PosePerson:
    track_id: int
    side: str
    bbox: tuple[int, int, int, int]
    points: np.ndarray | None
    confidence: np.ndarray | None


class PoseEstimator:
    """Adapter kept separate so YOLO Pose can later be replaced by MMPose."""

    def estimate(self, person: object) -> PosePerson:
        return PosePerson(
            track_id=person.track_id,
            side=person.side,
            bbox=person.bbox,
            points=person.keypoints,
            confidence=person.keypoint_conf,
        )

    @staticmethod
    def point(pose: PosePerson, index: int, min_conf: float = 0.25) -> np.ndarray | None:
        if pose.points is None or index >= len(pose.points):
            return None
        if pose.confidence is not None and (
            index >= len(pose.confidence) or float(pose.confidence[index]) < min_conf
        ):
            return None
        return np.asarray(pose.points[index], dtype=float)
