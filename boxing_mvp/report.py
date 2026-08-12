from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(report_dir: str, video_path: str, analysis: dict[str, Any]) -> str:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(analysis, ensure_ascii=False).replace("</", "<\\/")
    events = analysis.get("events", [])
    rows = []
    for i, event in enumerate(events):
        start = float(event.get("start_time_sec", 0))
        end = float(event.get("end_time_sec", start + 0.8))
        label = event.get("hit_or_miss", event.get("target", "unknown"))
        rows.append(f'<tr><td>{i + 1}</td><td>{start:.2f}-{end:.2f}</td><td>{html.escape(str(event.get("fighter", event.get("side", "unknown"))))}</td><td><span class="tag {html.escape(str(label))}">{html.escape(str(label))}</span></td><td>{html.escape(str(event.get("reason", "")))}</td><td><button onclick="playEvent({start:.3f},{end:.3f})">回放</button></td></tr>')
    summary = analysis.get("summary", {})
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>拳击视频分析报告</title><style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;background:#f6f7fb;color:#172033}} video{{width:100%;max-height:620px;background:#111;border-radius:8px}} .card{{background:white;border-radius:10px;padding:16px;margin:16px 0;box-shadow:0 1px 5px #ccd}} table{{width:100%;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left}} .tag{{padding:3px 8px;border-radius:12px;background:#e5e7eb}} .hit{{background:#c8f7d2}} .miss{{background:#ffe3a3}} .blocked{{background:#ffc7c7}} .uncertain{{background:#d9d9ff}} button{{cursor:pointer;padding:4px 10px}} .muted{{color:#667085}}
</style></head><body><h1>拳击视频分析报告</h1><p class="muted">空间接近不等于命中；VLM 判断仅代表视频视觉证据，不是裁判判定。</p>
<video id="player" controls src="{html.escape(video_path)}"></video>
<section class="card"><h2>整场摘要</h2><pre id="summary">{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>
<section class="card"><h2>事件时间轴与回放</h2><table><thead><tr><th>#</th><th>时间</th><th>选手</th><th>VLM 判断</th><th>说明</th><th></th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="6">暂无事件</td></tr>'}</tbody></table></section>
<script>const analysis={data};const player=document.getElementById('player');let stopAt=0;function playEvent(start,end){{player.currentTime=start;stopAt=end;player.play();}}player.addEventListener('timeupdate',()=>{{if(stopAt&&player.currentTime>=stopAt){{player.pause();stopAt=0;}}}});</script></body></html>'''
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return str(path)
