from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoLoader:
    """Sequential MP4 reader with optional frame skipping.

    Keeping decoding here makes the inference loop testable and prevents the
    model/tracker layer from knowing anything about OpenCV capture details.
    """

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        self.path = str(path)
        self.stride = max(1, int(stride))
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"无法打开视频: {self.path}")
        self.info = VideoInfo(
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0),
            frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            if frame_index % self.stride == 0:
                yield frame_index, frame
            frame_index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_writer(path: str | Path, info: VideoInfo) -> cv2.VideoWriter:
    """Create an MP4 writer; mp4v is widely available on headless servers."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频输出: {output}")
    return writer
