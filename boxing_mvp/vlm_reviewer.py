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
            f"""你是拳击视频分析员，负责给教练和运动员复盘整场比赛。
视频总时长为 {duration:.2f} 秒。你有权限做深入分析，但必须区分：直接可见的事实（evidence=direct）、合理的推断（evidence=inferred）、完全无法确认（evidence=unknown）。

分析要求：
1. 比赛结构：只有出现明确的回合开始/结束信号才创建回合；否则按动作节奏划分为"阶段"，不要编造回合。
2. 双方风格：根据整场出拳频率、移动、防守方式，用一段话分别描述红方和蓝方的风格（进攻型/防守型/游击型等），推断必须标注为推断。
3. 关键时间点：找出比赛节奏变化的关键时间戳（例如某一方突然提速、出现决定性连击、某方开始搂抱消耗时间）。
4. 出拳统计：对每一方估计全场出拳总量级（low<20拳 / medium 20-60拳 / high>60拳）和命中总量级，标注为估计，不要假装精确。
5. 胜负趋势：只描述画面可见的优劣态势（谁控制距离、谁压迫、谁被击中更多），不宣布胜负。

所有时间用整段视频绝对秒数，范围 0 到 {duration:.2f}。
严格只返回一个 JSON 对象，不要 Markdown、解释或额外文本：
{{"summary":"中文摘要，300字以内，结构清晰","rounds":[{{"start_time_sec":0.0,"end_time_sec":0.0,"description":"只写可见事实"}}],"phases":[{{"start_time_sec":0.0,"end_time_sec":0.0,"label":"试探期/对攻期/消耗期/尾声","evidence":"direct|inferred|unknown"}}],"key_moments":[{{"time_sec":0.0,"description":"关键事件描述","evidence":"direct|inferred|unknown"}}],"fighter_analysis":{{"red":{{"style":"...","punch_volume":"low|medium|high","observations":["可见事实"]}},"blue":{{"style":"...","punch_volume":"low|medium|high","observations":["可见事实"]}}}},"momentum":{{"description":"画面可见的优劣态势","evidence":"direct|inferred|unknown"}},"global_observations":["可见事实"]}}""",
            0,
            duration,
        )
    def coarse_scan(self, start: float, end: float) -> dict[str, Any]:
        return self._generate(
            f"""你是拳击视频候选事件检测器，只分析 {start:.2f} 到 {end:.2f} 秒窗口。
你有权限精细描述，但每个候选必须基于窗口内可见帧，宁可漏检也不把移动、举手、格挡或搂抱误报为拳。

输出要求：
1. 只找疑似出拳动作，不判断是否命中；空间接近不等于接触。
2. 动作窗口必须紧贴实际动作起止：起始=拳开始离开起始位置，结束=拳收回或到达终点，误差控制在 0.3 秒内。
3. 连续组合拳：同一方连续多拳（间隔<1秒）标记为组合，用 sequence_id 关联，每拳仍是独立候选。
4. 出手幅度：full=大幅度全伸，half=短促，unknown=看不清。
5. 看不清出拳者、手、目标就写 unknown，不要猜。

相对时间从窗口起点 0 秒开始，必须在 0 到 {end - start:.2f} 秒内。
严格只返回一个 JSON 对象，不要 Markdown 或额外文本：
{{"events":[{{"relative_start_sec":0.0,"relative_end_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","punch_type":"直拳|摆拳|勾拳|上勾拳|不确定","target_region":"head|torso|miss|unknown","amplitude":"full|half|unknown","sequence_id":1,"confidence":0.0}}]}}。
confidence 是对"确实存在出拳动作"的信心，不是命中信心。没有候选返回空数组。""",
            start,
            end,
        )
    def fine_scan(self, start: float, end: float, candidate: dict[str, Any]) -> dict[str, Any]:
        return self._generate(
            f"""你是拳击视频事件复核员，精细复核 {start:.2f} 到 {end:.2f} 秒窗口。
候选标注仅供参考，可能错误，必须以窗口内可见帧为准：{json.dumps(candidate, ensure_ascii=False)}

你有权限做深入分析，但每一条结论都必须有证据支撑，并在 evidence 字段标注来源：
- direct：画面中直接可见
- inferred：基于可见证据的合理推断
- unknown：完全无法确认

复核步骤（逐步执行，不要跳过）：
1. 先确认是否真的有出拳；搂抱、推搡、触碰手套、裁判介入、非拳击动作一律不算。
2. 判断出拳者、前后手、拳型、幅度（full/half）。
3. 判断目标区域和命中部位细节：head_left/head_right/head_top/torso/arm_block/unknown。
4. 判定结果：
   - hit：画面中清晰看到拳套接触对方头部或躯干，或接触后对方出现可见反应（头部偏移、身体后仰、晃动），结合接触前一刻的拳路连续性。
   - blocked：清晰看到防守动作（格挡、拍击、格架）在接触前拦截拳路。
   - miss：清晰看到出拳轨迹且未触碰到对方。
   - uncertain：证据不足、遮挡或镜头角度无法判断。
5. 若判定 hit，补充：对方防守动作类型（格挡/后撤/摇闪/搂抱/无/看不清）、对方受击反应（明显偏移/轻微反应/无反应/遮挡看不清）、力度感（low/medium/high，基于手臂速度和位移的视觉估计，标注 inferred）。
6. 判断是否属于组合拳的一部分（该候选前后 1 秒内是否有同方出拳），以及出拳后是否有立即反击。

时间是相对窗口起点的秒数，必须位于 0 到 {end - start:.2f} 秒；无法精确定位时给出保守范围。
严格只返回一个 JSON 对象，不要 Markdown 或额外文本：
{{"is_punch":"yes|no|uncertain","start_time_sec":0.0,"peak_time_sec":0.0,"end_time_sec":0.0,"fighter":"red|blue|unknown","hand":"front|rear|unknown","punch_type":"直拳|摆拳|勾拳|上勾拳|组合|不确定","amplitude":"full|half|unknown","target_region":"head|torso|miss|unknown","impact_area":"head_left|head_right|head_top|torso|arm_block|unknown","contact_evidence":"clear|possible|none|occluded","hit_or_miss":"hit|miss|blocked|uncertain","blocked":"yes|no|uncertain","block_type":"格挡|后撤|摇闪|搂抱|无|看不清","reaction":"明显偏移|轻微反应|无反应|遮挡看不清","power":"low|medium|high|unknown","part_of_combo":"yes|no|uncertain","countered":"yes|no|uncertain","occluded":"yes|no|uncertain","confidence":0.0,"evidence":"direct|inferred|unknown","reason":"用中文写出可见证据链和不确定性，2-4句，具体到动作细节"}}。
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
