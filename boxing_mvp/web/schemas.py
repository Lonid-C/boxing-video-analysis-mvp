from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
class VideoCreate(BaseModel):
    path: str
    name: str | None = None
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
class VideoOut(VideoCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
class RunCreate(BaseModel):
    video_id: str
    mode: str = "cv"
    params: dict[str, Any] = Field(default_factory=dict)
class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    video_id: str
    status: str
    mode: str
    schema_version: str
    params: dict[str, Any]
    error: str | None = None
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    run_id: str
    external_id: str | None
    source: str
    start_time_sec: float | None
    peak_time_sec: float | None
    end_time_sec: float | None
    fighter: str | None
    hand: str | None
    hit_or_miss: str | None
    confidence: float | None
    payload: dict[str, Any]
class TrackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    frame_index: int
    time_sec: float | None
    people: list[dict[str, Any]]
