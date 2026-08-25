from __future__ import annotations

from types import SimpleNamespace
import threading

import boxing_mvp.services as services


class FakeThread:
    starts = 0

    def __init__(self, *args, **kwargs):
        self.alive = False

    def start(self):
        self.alive = True
        FakeThread.starts += 1

    def is_alive(self):
        return self.alive


def test_start_run_is_idempotent(monkeypatch):
    FakeThread.starts = 0
    services._threads.clear()


def test_default_run_paths_are_isolated_by_run_id(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(services, "_set_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("stop after command")))
    monkeypatch.setenv("BOXING_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("BOXING_RUN_ROOT", str(tmp_path / "runs"))
    class Completed: returncode = 0
    def fake_run(command, **kwargs):
        calls.append(command); return Completed()
    monkeypatch.setattr(services.subprocess, "run", fake_run)
    for run_id in ("run-a", "run-b"):
        services.execute_run(run_id, "/video.mp4", "cv", {"export_package": False})
    joined = [" ".join(command) for command in calls]
    assert "/runs/run-a/analysis.json" in joined[0]
    assert "/runs/run-b/analysis.json" in joined[1]
    assert joined[0] != joined[1]


def test_run_status_changes_only_after_execution_slot(monkeypatch):
    statuses = []
    class Slot:
        def acquire(self): statuses.append("acquired")
        def release(self): statuses.append("released")
    monkeypatch.setattr(services, "_run_slots", Slot())
    monkeypatch.setattr(services, "_set_status", lambda _id, status, error=None: statuses.append(status))
    monkeypatch.setattr(services.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))
    services.execute_run("run", "/video", "cv", {})
    assert statuses[:2] == ["acquired", "running"]
    assert statuses[-2:] == ["failed", "released"]
    monkeypatch.setattr(services.threading, "Thread", FakeThread)
    run = SimpleNamespace(id="same-run", mode="cv", params={})
    services.start_run(run, "/video.mp4")
    services.start_run(run, "/video.mp4")
    assert FakeThread.starts == 1
    assert list(services._threads) == ["same-run"]
    services._threads.clear()
