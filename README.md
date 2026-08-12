# Boxing Video Analysis MVP

一个面向拳击视频复盘的可运行 MVP：先用 CV 找候选拳，再用视觉语言模型复核，最后在网页时间轴中点击回放。

## 现在能做到什么

```text
上传/登记视频
  → YOLO Pose + ByteTrack 跟踪两位选手
  → 规则检测器生成疑似出拳候选
  → Qwen-VL（阿里云百炼）整场摘要、粗扫描、精扫描
  → 保存回合、事件、轨迹和分析任务
  → 网页播放器按事件时间点回放
```

当前 MVP 已经打通真实视频链路：

- 支持本地视频，也支持通过 OSS HTTPS URL 让 Qwen 直接读取视频；
- 支持 `hit`、`miss`、`blocked`、`uncertain` 四种复核结果；
- 每个事件保留出拳者、左右手、目标区域、时间区间、置信度和中文证据说明；
- 生成 `analysis.json`、`tracks.jsonl`、标注视频和可点击 HTML 报告；
- FastAPI + SQLite 页面可以查看视频、任务状态、事件列表和短片段回放；
- API 已有 `/api/v1` 版本前缀，后续可替换为队列、PostgreSQL 和独立前端。

## 效果示例

在真实拳击视频的一段精扫描中，Qwen-VL 返回过如下结构化结果：

```json
{
  "is_punch": "yes",
  "start_time_sec": 2.2,
  "peak_time_sec": 2.27,
  "end_time_sec": 2.35,
  "fighter": "blue",
  "hand": "rear",
  "punch_type": "直拳",
  "target_region": "head",
  "contact_evidence": "clear",
  "hit_or_miss": "hit",
  "blocked": "no",
  "confidence": 0.95,
  "reason": "蓝方右后手明显前伸，拳套与红方头部有清晰接触。"
}
```

这说明当前 MVP 已经不只是输出“某处可能有动作”，而是能够把 CV 候选缩小到具体时间段，再由 VLM 根据可见接触证据给出可解释的复核结果。

> 这是视频分析辅助工具，不是裁判系统。空间接近不等于命中；遮挡、镜头质量和选手身份变化仍会影响结果。`hit` 只应在画面中能看到清晰接触时使用。

## 快速运行

```bash
# 安装依赖
pip install -r requirements.txt

# 本地 CV 分析
PYTHONPATH=. python -m boxing_mvp.main \
  --input inputs/fight.mp4 \
  --output outputs/fight_annotated.mp4 \
  --stats outputs/fight.json \
  --no-display

# 启用阿里云百炼 Qwen-VL
export BAILIAN_API_KEY="你的 Key"
export BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export BAILIAN_MODEL="qwen3-vl-plus"
PYTHONPATH=. python -m boxing_mvp.main \
  --input inputs/fight.mp4 \
  --output outputs/fight_vlm.mp4 \
  --stats outputs/fight_vlm.json \
  --vlm --vlm-mode all --vlm-media-mode images --no-display
```

## 启动 Web Demo

```bash
PYTHONPATH=. python scripts/init_db.py
PYTHONPATH=. uvicorn boxing_mvp.web.app:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`。主要接口：

```text
POST /api/v1/uploads
POST /api/v1/videos
POST /api/v1/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/events
GET  /api/v1/runs/{run_id}/tracks
```

## 项目结构

```text
boxing_mvp/              CV、VLM 和分析管线
boxing_mvp/web/          FastAPI、数据库模型和网页入口
boxing_mvp/services.py   后台分析任务（MVP 版进程内线程）
boxing_mvp/importer.py   analysis.json / tracks.jsonl 导入 SQLite
scripts/init_db.py       初始化数据库
```

## 下一步

当前版本重点是验证“候选召回 → VLM 复核 → 可回放结果”是否可用。下一阶段会加入 Redis/Celery 任务队列、用户权限、私有 OSS 短期 URL、PostgreSQL、任务重试和更完整的前端可视化。

## 安全提醒

不要把 API Key、OSS AccessKey、预签名 URL、真实视频或数据库文件提交到 GitHub。生产环境请使用最小权限的 RAM 用户，并通过环境变量或密钥管理服务注入凭据。
