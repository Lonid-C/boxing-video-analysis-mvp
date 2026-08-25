from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable

import numpy as np

from .pose_estimator import (
    L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE,
    R_ELBOW, R_HIP, R_SHOULDER, R_WRIST,
    PoseEstimator, PosePerson,
)


@dataclass
class PunchEvent:
    event_id: int
    fighter_id: int
    side: str
    hand: str
    start_frame: int
    end_frame: int
    punch_type: str
    target: str
    peak_speed: float
    confidence: float
    opponent_id: int | None = None
    target_region: str = "unknown"
    target_distance: float | None = None
    proximity_score: float = 0.0
    vlm_review: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, fps: float) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["start_time_sec"] = round(self.start_frame / fps, 3) if fps else 0.0
        d["end_time_sec"] = round(self.end_frame / fps, 3) if fps else 0.0
        d["peak_speed"] = round(self.peak_speed, 4)
        d["confidence"] = round(self.confidence, 3)
        if d["target_distance"] is not None:
            d["target_distance"] = round(float(d["target_distance"]), 4)
        d["proximity_score"] = round(float(self.proximity_score), 3)
        if self.vlm_review:
            d.update(self.vlm_review)
        return d


@dataclass
class _HandState:
    last_point: np.ndarray | None = None
    last_frame: int = -1
    active: bool = False
    start_frame: int = -1
    peak_speed: float = 0.0
    peak_point: np.ndarray | None = None
    cooldown_until: int = -1
    opponent_id: int | None = None
    target_region: str = "unknown"
    target_distance: float | None = None
    proximity_score: float = 0.0


@dataclass
class _FighterState:
    hands: dict[str, _HandState] = field(default_factory=lambda: {"left": _HandState(), "right": _HandState()})


class PunchDetector:
    """Explainable wrist-speed detector with opponent proximity context.

    Speed uses original video FPS and the original frame indices. This is
    intentional: when VideoLoader skips frames, the frame-index delta already
    accounts for the elapsed time, so stride must not be applied a second time.
    Target fields describe spatial proximity, not a confirmed legal hit.
    """

    def __init__(self, fps: float, start_speed: float = 0.09, end_speed: float = 0.045,
                 min_duration: int = 2, max_duration: int = 24, cooldown: int = 5) -> None:
        self.fps = max(float(fps), 1.0)
        self.start_speed = start_speed
        self.end_speed = end_speed
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.cooldown = cooldown
        self.states: dict[int, _FighterState] = {}
        self.events: list[PunchEvent] = []
        self.next_event_id = 1

    def update(self, frame_index: int, pose: PosePerson) -> list[PunchEvent]:
        """Backward-compatible single-person update without target context."""
        return self.update_frame(frame_index, [pose])

    def update_frame(self, frame_index: int, poses: Iterable[PosePerson]) -> list[PunchEvent]:
        poses = list(poses)
        outputs: list[PunchEvent] = []
        for pose in poses:
            opponent = self._select_opponent(pose, poses)
            outputs.extend(self._update_person(frame_index, pose, opponent))
        self.events.extend(outputs)
        return outputs

    def _update_person(self, frame_index: int, pose: PosePerson,
                       opponent: PosePerson | None) -> list[PunchEvent]:
        state = self.states.setdefault(pose.track_id, _FighterState())
        x1, y1, x2, y2 = pose.bbox
        scale = max(hypot(x2 - x1, y2 - y1), 1.0)
        shoulder_l = PoseEstimator.point(pose, L_SHOULDER)
        shoulder_r = PoseEstimator.point(pose, R_SHOULDER)
        hip_l = PoseEstimator.point(pose, L_HIP)
        hip_r = PoseEstimator.point(pose, R_HIP)
        shoulder_mid = self._mid(shoulder_l, shoulder_r)
        hip_mid = self._mid(hip_l, hip_r)
        outputs: list[PunchEvent] = []
        # COCO keypoints encode anatomical left/right. Front/rear depends on
        # stance and cannot be inferred reliably from keypoint index alone.
        for hand, wrist_index, elbow_index in (("left", L_WRIST, L_ELBOW), ("right", R_WRIST, R_ELBOW)):
            wrist = PoseEstimator.point(pose, wrist_index)
            elbow = PoseEstimator.point(pose, elbow_index)
            hs = state.hands[hand]
            speed = 0.0
            if wrist is not None and hs.last_point is not None and frame_index > hs.last_frame:
                speed = float(np.linalg.norm(wrist - hs.last_point) / scale * self.fps / (frame_index - hs.last_frame))
            if wrist is not None:
                hs.last_point, hs.last_frame = wrist, frame_index
            if not hs.active and frame_index >= hs.cooldown_until and speed >= self.start_speed:
                hs.active = True
                hs.start_frame = frame_index
                hs.peak_speed = speed
                hs.peak_point = wrist
                self._reset_target(hs)
            elif hs.active:
                hs.peak_speed = max(hs.peak_speed, speed)
                if wrist is not None and speed >= hs.peak_speed:
                    hs.peak_point = wrist
                self._update_target(hs, wrist, opponent)
                duration = frame_index - hs.start_frame
                if (speed <= self.end_speed and duration >= self.min_duration) or duration >= self.max_duration:
                    outputs.append(self._finish(frame_index, pose, hand, hs, shoulder_mid, hip_mid, elbow))
                    hs.active = False
                    hs.cooldown_until = frame_index + self.cooldown
        return outputs

    @staticmethod
    def _select_opponent(pose: PosePerson, poses: list[PosePerson]) -> PosePerson | None:
        candidates = [p for p in poses if p.track_id != pose.track_id]
        if not candidates:
            return None
        x1, y1, x2, y2 = pose.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        own_scale = max(hypot(x2 - x1, y2 - y1), 1.0)
        return min(candidates, key=lambda p: (
            ((p.bbox[0] + p.bbox[2]) / 2 - cx) ** 2 + ((p.bbox[1] + p.bbox[3]) / 2 - cy) ** 2
        ) / own_scale ** 2)

    @staticmethod
    def _reset_target(hs: _HandState) -> None:
        hs.opponent_id = None
        hs.target_region = "unknown"
        hs.target_distance = None
        hs.proximity_score = 0.0

    def _update_target(self, hs: _HandState, wrist: np.ndarray | None,
                       opponent: PosePerson | None) -> None:
        if wrist is None or opponent is None:
            return
        region_points = self._target_points(opponent)
        if not region_points:
            return
        ox1, oy1, ox2, oy2 = opponent.bbox
        opponent_scale = max(hypot(ox2 - ox1, oy2 - oy1), 1.0)
        best_region, best_distance = min(
            ((name, float(np.linalg.norm(wrist - point) / opponent_scale))
             for name, point in region_points.items()),
            key=lambda item: item[1],
        )
        radius = {"head": 0.24, "torso": 0.34}[best_region]
        proximity = max(0.0, min(1.0, 1.0 - best_distance / radius))
        if proximity > hs.proximity_score:
            hs.opponent_id = opponent.track_id
            hs.target_distance = best_distance
            hs.proximity_score = proximity
            hs.target_region = best_region if proximity > 0 else "miss"

    @classmethod
    def _target_points(cls, opponent: PosePerson) -> dict[str, np.ndarray]:
        nose = PoseEstimator.point(opponent, NOSE)
        shoulders = cls._mid(PoseEstimator.point(opponent, L_SHOULDER), PoseEstimator.point(opponent, R_SHOULDER))
        hips = cls._mid(PoseEstimator.point(opponent, L_HIP), PoseEstimator.point(opponent, R_HIP))
        points: dict[str, np.ndarray] = {}
        if nose is not None:
            points["head"] = nose
        elif shoulders is not None:
            ox1, oy1, ox2, oy2 = opponent.bbox
            points["head"] = shoulders + np.array([0.0, -0.20 * max(oy2 - oy1, 1.0)])
        if shoulders is not None and hips is not None:
            points["torso"] = (shoulders + hips) / 2.0
        elif shoulders is not None:
            points["torso"] = shoulders
        return points

    @staticmethod
    def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return (a + b) / 2.0

    def _finish(self, frame: int, pose: PosePerson, hand: str, hs: _HandState,
                shoulder: np.ndarray | None, hip: np.ndarray | None,
                elbow: np.ndarray | None) -> PunchEvent:
        wrist = hs.peak_point
        punch_type = "直拳"
        if wrist is not None and elbow is not None and shoulder is not None:
            arm = wrist - shoulder
            bend = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow) - np.linalg.norm(wrist - shoulder))
            if bend > 0.22 * max(np.linalg.norm(arm), 1.0):
                punch_type = "勾拳" if abs(float(arm[1])) < abs(float(arm[0])) else "摆拳"
        legacy_target = {"head": "击头", "torso": "击躯干"}.get(hs.target_region, "未知")
        confidence = min(0.99, 0.45 + hs.peak_speed / max(self.start_speed * 5.0, 1e-6))
        event = PunchEvent(self.next_event_id, pose.track_id, pose.side, hand,
                           hs.start_frame, frame, punch_type, legacy_target, hs.peak_speed, confidence,
                           hs.opponent_id, hs.target_region, hs.target_distance, hs.proximity_score)
        self.next_event_id += 1
        return event

    def stats(self, fps: float, round_seconds: float = 180.0) -> dict[str, Any]:
        rounds: dict[int, dict[str, Any]] = {}
        for event in self.events:
            round_no = int((event.start_frame / max(fps, 1.0)) // round_seconds) + 1
            bucket = rounds.setdefault(round_no, {"round": round_no, "punch_count": 0, "by_side": {}, "by_type": {}, "by_target": {}})
            bucket["punch_count"] += 1
            for key, value in (("by_side", event.side), ("by_type", event.punch_type), ("by_target", event.target)):
                bucket[key][value] = bucket[key].get(value, 0) + 1
        return {"total_punches": len(self.events), "rounds": list(rounds.values()),
                "events": [e.as_dict(fps) for e in self.events]}
