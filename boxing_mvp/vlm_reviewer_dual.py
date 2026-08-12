from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any

import cv2


class VLMReviewer:
    """OpenAI-compatible VLM client with video and sampled-image modes."""

    def __init__(self, api_key: str | None = None, model: str = "[按次]gemini-2.5-flash",
                 base_url: str | None = None, timeout: float = 180.0,
                 enabled: bool = True, media_mode: str = "auto") -> None:
        self.api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = (base_url or os.getenv("VLM_BASE_URL") or "https://new.bitexingai.com/v1").rstrip("/")
        self.model, self.timeout = model, timeout
        self.media_mode = media_mode if media_mode in ("auto", "video", "images") else "auto"
        self.active_media_mode = "video" if self.media_mode != "images" else "images"
        self.enabled = bool(enabled and self.api_key)
        self.calls = 0; self.failures = 0; self.video_path: str | None = None; self.duration = 0.0; self.last_error = ""

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else ("no_api_key" if not self.api_key else "disabled")

    def _json_request(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(), method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response: return json.loads(response.read().decode())

    def upload_video(self, path: str) -> dict[str, Any]:
        if not self.enabled: return {"status": "not_reviewed"}
        self.video_path = path; cap = cv2.VideoCapture(path); fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); self.duration = count / fps if count else 0.0; cap.release()
        return {"status": "ready", "name": path, "mimeType": "video/mp4"}

    def _video_part(self) -> dict[str, Any] | None:
        if not self.video_path: return None
        with open(self.video_path, "rb") as stream: encoded = base64.b64encode(stream.read()).decode("ascii")
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}}

    def _image_parts(self, start: float, end: float, count: int = 8) -> list[dict[str, Any]]:
        if not self.video_path: return []
        cap = cv2.VideoCapture(self.video_path); points = [start + (end - start) * i / max(count - 1, 1) for i in range(count)]; parts: list[dict[str, Any]] = []
        for timestamp in points:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000); ok, frame = cap.read()
            if not ok: continue
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok: parts.extend([{"type": "text", "text": f"帧时间 {timestamp:.3f} 秒"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}}])
        cap.release(); return parts

    def _request(self, prompt: str, start: float, end: float) -> dict[str, Any]:
        if self.active_media_mode == "video":
            part = self._video_part(); content = ([part] if part else []) + [{"type": "text", "text": prompt}]
        else: content = [{"type": "text", "text": prompt}] + self._image_parts(start, end)
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.1, "response_format": {"type": "json_object"}}
        self.calls += 1; payload = self._json_request(body); text = payload["choices"][0]["message"]["content"]
        if isinstance(text, list): text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
        return {"review_status": "reviewed", "review_model": self.model, "media_mode": self.active_media_mode, **self.parse_json(text)}

    def _generate(self, prompt: str, start: float = 0.0, end: float | None = None) -> dict[str, Any]:
        if not self.enabled or not self.video_path: return {"review_status": "not_reviewed"}
        end = self.duration if end is None else end
        try: return self._request(prompt, start, end)
        except Exception as exc:
            self.last_error = str(exc)[:300]
            if self.media_mode == "auto" and self.active_media_mode == "video":
                self.active_media_mode = "images"
                try: return self._request(prompt, start, end)
                except Exception as fallback_exc: self.last_error = str(fallback_exc)[:300]
            self.failures += 1; return {"review_status": "error", "review_error": self.last_error, "media_mode": self.active_media_mode}

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"): text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        if not isinstance(value, dict): raise ValueError("VLM JSON response is not an object")
        return value

    def summarize_video(self, duration: float) -> dict[str, Any]:
        return self._generate(f'分析整段拳击视频并生成回合摘要。总时长 {duration:.2f} 秒。严格 JSON：{{"summary":"中文摘要","rounds":[],"global_observations":[]}}。只描述可见内容；空间接近不等于命中。', 0, duration)

    def coarse_scan(self, start: float, end: float) -> dict[str, Any]:
        return self._generate(f'扫描视频 {start:.2f} 到 {end:.2f} 秒，找出疑似出拳。严格 JSON：{{"events":[{{"relative_start_sec":0.0,"relative_end_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","target_region":"head|torso|miss|unknown","confidence":0.0}}]}}。时间是相对本窗口起点；不判断命中。', start, end)

    def fine_scan(self, start: float, end: float, candidate: dict[str, Any]) -> dict[str, Any]:
        return self._generate(f'精细复核 {start:.2f} 到 {end:.2f} 秒的候选拳：{json.dumps(candidate, ensure_ascii=False)}。严格 JSON：{{"is_punch":"yes|no|uncertain","start_time_sec":0.0,"peak_time_sec":0.0,"end_time_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","punch_type":"直拳|勾拳|摆拳|不确定","target_region":"head|torso|miss|unknown","contact_evidence":"clear|possible|none|occluded","hit_or_miss":"hit|miss|blocked|uncertain","blocked":"yes|no|uncertain","occluded":"yes|no|uncertain","confidence":0.0,"reason":""}}。时间相对窗口起点；看不到接触必须 uncertain。', start, end)

    def stats(self) -> dict[str, Any]:
        return {"vlm_status": self.status, "vlm_model": self.model, "vlm_base_url": self.base_url, "vlm_media_mode": self.active_media_mode, "vlm_calls": self.calls, "vlm_failures": self.failures, "vlm_last_error": self.last_error}
