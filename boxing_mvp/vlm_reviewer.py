from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2


class VLMReviewer:
    """Alibaba Bailian multimodal reviewer using the OpenAI-compatible API.

    The default ``images`` mode samples frames from the local video and sends
    them as multimodal image content. This is more portable than embedding a
    whole MP4 in a request body. ``video`` mode is available when the selected
    Bailian model and endpoint explicitly support video input.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        enabled: bool = True,
        media_mode: str = "images",
    ) -> None:
        self.api_key = api_key or os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (base_url or os.getenv("BAILIAN_BASE_URL") or
                         "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.model = model or os.getenv("BAILIAN_MODEL") or "qwen-vl-max"
        self.timeout = timeout
        self.media_mode = media_mode if media_mode in ("auto", "video", "images") else "images"
        self.active_media_mode = "video" if self.media_mode == "video" else "images"
        self.enabled = bool(enabled and self.api_key)
        self.calls = 0
        self.failures = 0
        self.video_path: str | None = None
        self.video_url: str | None = None
        self.duration = 0.0
        self.last_error = ""

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else ("no_api_key" if not self.api_key else "disabled")

    def _json_request(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Bailian HTTP {exc.code}: {detail}") from exc

    def upload_video(self, path: str, public_url: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "not_reviewed"}
        self.video_path = str(Path(path).resolve())
        self.video_url = public_url or None
        capture = cv2.VideoCapture(self.video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = count / fps if count else 0.0
        capture.release()
        return {"status": "ready", "name": self.video_path, "mimeType": "video/mp4"}

    def _video_part(self) -> dict[str, Any] | None:
        if self.video_url:
            return {"type": "video_url", "video_url": {"url": self.video_url}}
        if not self.video_path:
            return None
        with open(self.video_path, "rb") as stream:
            encoded = base64.b64encode(stream.read()).decode("ascii")
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}}

    def _image_parts(self, start: float, end: float, count: int = 8) -> list[dict[str, Any]]:
        if not self.video_path:
            return []
        capture = cv2.VideoCapture(self.video_path)
        points = [start + (end - start) * i / max(count - 1, 1) for i in range(count)]
        parts: list[dict[str, Any]] = []
        for timestamp in points:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                parts.extend([
                    {"type": "text", "text": f"帧时间 {timestamp:.3f} 秒"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
                    }},
                ])
        capture.release()
        return parts

    @staticmethod
    def _content_text(payload: dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        raise ValueError("Bailian response content is not text")

    def _request(self, prompt: str, start: float, end: float) -> dict[str, Any]:
        if self.active_media_mode == "video":
            part = self._video_part()
            content = ([part] if part else []) + [{"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}] + self._image_parts(start, end)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        self.calls += 1
        text = self._content_text(self._json_request(body))
        return {
            "review_status": "reviewed",
            "review_model": self.model,
            "review_provider": "bailian",
            "media_mode": self.active_media_mode,
            **self.parse_json(text),
        }

    def _generate(self, prompt: str, start: float = 0.0, end: float | None = None) -> dict[str, Any]:
        if not self.enabled or not self.video_path:
            return {"review_status": "not_reviewed"}
        end = self.duration if end is None else end
        try:
            return self._request(prompt, start, end)
        except Exception as exc:
            self.last_error = str(exc)[:500]
            if self.media_mode == "auto" and self.active_media_mode == "video":
                self.active_media_mode = "images"
                try:
                    return self._request(prompt, start, end)
                except Exception as fallback_exc:
                    self.last_error = str(fallback_exc)[:500]
            self.failures += 1
            return {
                "review_status": "error",
                "review_error": self.last_error,
                "review_provider": "bailian",
                "media_mode": self.active_media_mode,
            }

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("VLM JSON response is not an object")
        return value

    def summarize_video(self, duration: float) -> dict[str, Any]:
        return self._generate(
            f"""你是拳击视频分析员。分析整段视频，视频总时长为 {duration:.2f} 秒。
目标是提供保守、可核验的比赛摘要，不是编故事，也不是统计所有动作。
只记录视频中直接可见的内容；无法确认的内容写“未知”，不要根据拳击常识补全。
不要把选手靠近、挥拳、身体反应或裁判动作单独当作命中。
只有能看到拳套与头部/躯干发生清晰接触，才可在描述中称为命中；否则使用“疑似”“未确认”或“未知”。
回合边界只有在视频中出现明确的回合开始/结束信号时才填写；不要因为服装变化、镜头切换或时间间隔擅自创造回合。
区分红方和蓝方；如果颜色、身份或时间无法确认，使用 unknown。
所有时间必须是整段视频的绝对秒数，范围为 0 到 {duration:.2f}。
严格只返回一个 JSON 对象，不要 Markdown、解释或额外文本：
{{"summary":"中文摘要","rounds":[{{"start_time_sec":0.0,"end_time_sec":0.0,"description":"只写可见事实"}}],"global_observations":["可见事实"]}}""",
            0,
            duration,
        )

    def coarse_scan(self, start: float, end: float) -> dict[str, Any]:
        return self._generate(
            f"""你是拳击视频候选事件检测器。只分析 {start:.2f} 到 {end:.2f} 秒窗口。
均匀查看输入帧，找出可能的出拳动作，宁可漏检也不要把移动、举手、格挡或搂抱误报为拳。
这一阶段只生成候选，不判断是否命中；空间接近不等于接触，接触也不等于有效命中。
相对时间从窗口起点 0 秒开始，必须在 0 到 {end - start:.2f} 秒内。
每个候选必须是一个独立动作；看不清出拳者、手、目标区域就写 unknown，不要猜。
严格只返回一个 JSON 对象，不要 Markdown 或额外文本：
{{"events":[{{"relative_start_sec":0.0,"relative_end_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","target_region":"head|torso|miss|unknown","confidence":0.0}}]}}。
confidence 是对“确实存在出拳动作”的信心，不是命中信心。没有候选时返回空数组。""",
            start,
            end,
        )

    def fine_scan(self, start: float, end: float, candidate: dict[str, Any]) -> dict[str, Any]:
        return self._generate(
            f"""你是拳击视频事件复核员。精细复核 {start:.2f} 到 {end:.2f} 秒窗口。
候选标注仅供参考，可能错误，必须以窗口内可见帧为准：{json.dumps(candidate, ensure_ascii=False)}
先确认是否真的有出拳，再独立判断出拳者、前后手、目标区域和结果。
“hit”必须有清晰可见的拳套接触头部或躯干证据；仅仅拳套靠近、身体反应、姿势相似或遮挡下的推断都不能算 hit。
“blocked”只在能看到防守动作拦截拳路时使用；完全看不清接触点使用 uncertain 或 occluded。
区分 miss、blocked 和 uncertain：miss 是能看清未接触，blocked 是被防守动作拦截，uncertain 是证据不足。
不要把搂抱、推搡、触碰手套、裁判介入或非拳击动作当作出拳。
时间是相对窗口起点的秒数，必须位于 0 到 {end - start:.2f} 秒；无法精确定位时给出保守范围。
严格只返回一个 JSON 对象，不要 Markdown 或额外文本：
{{"is_punch":"yes|no|uncertain","start_time_sec":0.0,"peak_time_sec":0.0,"end_time_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","punch_type":"直拳|勾拳|摆拳|不确定","target_region":"head|torso|miss|unknown","contact_evidence":"clear|possible|none|occluded","hit_or_miss":"hit|miss|blocked|uncertain","blocked":"yes|no|uncertain","occluded":"yes|no|uncertain","confidence":0.0,"reason":"用中文简述可见证据和不确定性"}}。
confidence 是对整条复核结论的信心；看不到接触必须降低信心并避免使用 hit。""",
            start,
            end,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "vlm_status": self.status,
            "vlm_provider": "bailian",
            "vlm_model": self.model,
            "vlm_base_url": self.base_url,
            "vlm_media_mode": self.active_media_mode,
            "vlm_calls": self.calls,
            "vlm_failures": self.failures,
            "vlm_last_error": self.last_error,
        }
