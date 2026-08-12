from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    mode: Mapped[str] = mapped_column(String(32), default="cv")
    schema_version: Mapped[str] = mapped_column(String(16), default="1")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    video: Mapped[Video] = relationship(back_populates="runs")
    rounds: Mapped[list["Round"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    tracks: Mapped[list["Track"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Round(Base):
    __tablename__ = "rounds"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    round_no: Mapped[int] = mapped_column(Integer)
    start_time_sec: Mapped[float | None] = mapped_column(Float)
    end_time_sec: Mapped[float | None] = mapped_column(Float)
    summary: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    run: Mapped[AnalysisRun] = relationship(back_populates="rounds")
    __table_args__ = (UniqueConstraint("run_id", "round_no", name="uq_round_run_no"),)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32), default="cv", index=True)
    start_time_sec: Mapped[float | None] = mapped_column(Float, index=True)
    peak_time_sec: Mapped[float | None] = mapped_column(Float)
    end_time_sec: Mapped[float | None] = mapped_column(Float)
    fighter: Mapped[str | None] = mapped_column(String(32), index=True)
    hand: Mapped[str | None] = mapped_column(String(32))
    hit_or_miss: Mapped[str | None] = mapped_column(String(32), index=True)
    blocked: Mapped[str | None] = mapped_column(String(16))
    confidence: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    schema_version: Mapped[str] = mapped_column(String(16), default="1")
    run: Mapped[AnalysisRun] = relationship(back_populates="events")
    __table_args__ = (UniqueConstraint("run_id", "source", "external_id", name="uq_event_run_source_external"),)


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer, index=True)
    time_sec: Mapped[float | None] = mapped_column(Float)
    people: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    run: Mapped[AnalysisRun] = relationship(back_populates="tracks")
    __table_args__ = (UniqueConstraint("run_id", "frame_index", name="uq_track_run_frame"),)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    run: Mapped[AnalysisRun] = relationship(back_populates="artifacts")
    __table_args__ = (UniqueConstraint("run_id", "kind", name="uq_artifact_run_kind"),)
