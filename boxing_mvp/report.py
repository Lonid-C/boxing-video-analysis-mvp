from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(report_dir: str, video_path: str, analysis: dict[str, Any]) -> str:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(analysis, ensure_ascii=False).replace("</", "<\\/")
    all_events = analysis.get("events", []) if isinstance(analysis.get("events"), list) else []
    formal_events = analysis.get("vlm_events") if isinstance(analysis.get("vlm_events"), list) else []
    cv_events = [event for event in all_events if isinstance(event, dict) and event.get("source", "cv") == "cv"]

    def render_rows(events: list[dict[str, Any]], formal: bool) -> str:
      rows = []
      for i, event in enumerate(events):
        start = float(event.get("start_time_sec", 0))
        end = float(event.get("end_time_sec", start + 0.8))
        label = event.get("hit_or_miss", "正式事件" if formal else "CV候选")
        reason = event.get("reason", "") if formal else "仅表示疑似出拳动作，不表示命中"
        rows.append(f'<tr><td>{i + 1}</td><td>{start:.2f}-{end:.2f}</td><td>{html.escape(str(event.get("fighter", event.get("side", "unknown"))))}</td><td><span class="tag {html.escape(str(label))}">{html.escape(str(label))}</span></td><td>{html.escape(str(reason))}</td><td><button onclick="playEvent({start:.3f},{end:.3f})">回放</button></td></tr>')
      return "".join(rows)

    formal_rows = render_rows(formal_events, True)
    candidate_rows = render_rows(cv_events, False)
    summary = analysis.get("summary", {})
    exports = analysis.get("exports") if isinstance(analysis.get("exports"), dict) else {}
    fighter_summary = exports.get("fighter_summary") if isinstance(exports.get("fighter_summary"), list) else []
    summary_rows = []
    for row in fighter_summary:
        summary_rows.append(
            f'<tr><td>{html.escape(str(row.get("fighter", "")))}</td>'
            f'<td>{row.get("cv_candidates", 0)}</td><td>{row.get("vlm_reviewed", 0)}</td>'
            f'<td>{row.get("vlm_accepted", 0)}</td><td>{row.get("vlm_rejected", 0)}</td>'
            f'<td>{row.get("hit", 0)}</td><td>{row.get("miss", 0)}</td>'
            f'<td>{row.get("blocked", 0)}</td><td>{row.get("uncertain", 0)}</td>'
            f'<td>{row.get("hit_rate", "")}</td><td>{row.get("movement_samples", 0)}</td></tr>'
        )
    phases = summary.get("phases") if isinstance(summary, dict) and isinstance(summary.get("phases"), list) else []
    moments = summary.get("key_moments") if isinstance(summary, dict) and isinstance(summary.get("key_moments"), list) else []
    phase_items = "".join(f'<li>{html.escape(str(item.get("start_time_sec", "?")))}–{html.escape(str(item.get("end_time_sec", "?")))}s：{html.escape(str(item.get("label", "")))}</li>' for item in phases if isinstance(item, dict))
    moment_items = "".join(f'<li>{html.escape(str(item.get("time_sec", "?")))}s：{html.escape(str(item.get("description", "")))}</li>' for item in moments if isinstance(item, dict))
    file_links = []
    heatmaps = []
    for kind, raw_path in exports.get("files", {}).items():
        path = Path(raw_path)
        relative = Path("..") / path.name if path.parent == out.parent else Path(raw_path)
        href = html.escape(str(relative))
        file_links.append(f'<a class="download" href="{href}" download>{html.escape(kind)}</a>')
        if kind in {"heatmap_red", "heatmap_blue"}:
            heatmaps.append(f'<figure><img src="{href}"><figcaption>{html.escape(kind)}</figcaption></figure>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>拳击视频分析报告</title><style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;background:#f6f7fb;color:#172033}} video{{width:100%;max-height:620px;background:#111;border-radius:8px}} .card{{background:white;border-radius:10px;padding:16px;margin:16px 0;box-shadow:0 1px 5px #ccd}} table{{width:100%;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left}} .tag{{padding:3px 8px;border-radius:12px;background:#e5e7eb}} .hit{{background:#c8f7d2}} .miss{{background:#ffe3a3}} .blocked{{background:#ffc7c7}} .uncertain{{background:#d9d9ff}} button{{cursor:pointer;padding:4px 10px}} .muted{{color:#667085}} .download{{display:inline-block;margin:5px;padding:8px 12px;background:#e7eefc;border-radius:6px}} figure{{display:inline-block;width:46%;margin:2%}} figure img{{width:100%;border-radius:8px}}
</style></head><body><h1>拳击视频分析报告</h1><p class="muted">空间接近不等于命中；VLM 判断仅代表视频视觉证据，不是裁判判定。</p>
<video id="player" controls src="{html.escape(video_path)}"></video>
<section class="card"><h2>结果下载</h2>{''.join(file_links) or '暂无导出文件'}</section>
<section class="card"><h2>双方移动热力图</h2>{''.join(heatmaps) or '暂无热力图'}</section>
<section class="card"><h2>双方统计</h2><table><thead><tr><th>选手</th><th>CV候选</th><th>VLM复核</th><th>正式事件</th><th>未通过门禁</th><th>命中</th><th>未中</th><th>格挡</th><th>不确定</th><th>命中率</th><th>移动采样</th></tr></thead><tbody>{''.join(summary_rows) or '<tr><td colspan="11">暂无统计</td></tr>'}</tbody></table></section>
<section class="card"><h2>整场摘要</h2><pre id="summary">{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>
<section class="card"><h2>阶段与关键时刻</h2><h3>阶段</h3><ul>{phase_items or '<li>暂无</li>'}</ul><h3>关键时刻</h3><ul>{moment_items or '<li>暂无</li>'}</ul></section>
<section class="card"><h2>正式事件时间轴与回放</h2><p class="muted">仅展示通过身份、证据、置信度、时间与去重门禁的 VLM 事件。</p><table><thead><tr><th>#</th><th>时间</th><th>选手</th><th>VLM 判断</th><th>证据说明</th><th></th></tr></thead><tbody>{formal_rows or '<tr><td colspan="6">暂无通过门禁的正式事件</td></tr>'}</tbody></table></section>
<section class="card"><h2>CV 候选时间轴</h2><p class="muted">仅用于证据链和回放定位，不等同于命中。</p><table><thead><tr><th>#</th><th>时间</th><th>选手</th><th>类型</th><th>说明</th><th></th></tr></thead><tbody>{candidate_rows or '<tr><td colspan="6">暂无 CV 候选</td></tr>'}</tbody></table></section>
<script>const analysis={data};const player=document.getElementById('player');let stopAt=0;function playEvent(start,end){{player.currentTime=start;stopAt=end;player.play();}}player.addEventListener('timeupdate',()=>{{if(stopAt&&player.currentTime>=stopAt){{player.pause();stopAt=0;}}}});</script></body></html>'''
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return str(path)
