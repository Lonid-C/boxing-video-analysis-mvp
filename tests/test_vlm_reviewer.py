from __future__ import annotations

import io
import json
import urllib.error

import pytest

from boxing_mvp.vlm_reviewer import VLMReviewer


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps({"choices": []}).encode()


def test_retries_transient_http_error(monkeypatch):
    calls = []
    def urlopen(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("url", 429, "rate", {}, io.BytesIO(b"slow down"))
        return Response()
    monkeypatch.setattr("boxing_mvp.vlm_reviewer.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("boxing_mvp.vlm_reviewer.time.sleep", lambda _: None)
    reviewer = VLMReviewer(api_key="key", max_retries=2)
    assert reviewer._json_request({"x": 1}) == {"choices": []}
    assert len(calls) == 2


def test_does_not_retry_auth_error(monkeypatch):
    calls = []
    def urlopen(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("url", 401, "auth", {}, io.BytesIO(b"bad key"))
    monkeypatch.setattr("boxing_mvp.vlm_reviewer.urllib.request.urlopen", urlopen)
    reviewer = VLMReviewer(api_key="key", max_retries=2)
    with pytest.raises(RuntimeError, match="401"):
        reviewer._json_request({"x": 1})
    assert len(calls) == 1
