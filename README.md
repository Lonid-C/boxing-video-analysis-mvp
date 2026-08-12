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

以下结果均来自真实拳击比赛视频（约 160 秒室内比赛，红蓝双方 + 裁判），使用阿里云百炼 Qwen-VL（`qwen3-vl-plus`）生成。

### 整场视频摘要

Qwen 通过 OSS HTTPS URL 直接读取完整 160 秒视频，返回结构化摘要：

```json
{
  “summary”: “视频为一场室内拳击比赛，红方与蓝方选手在标准拳击台上对抗，裁判在场。比赛分为两个阶段：第一阶段约0—101秒，双方进行了多轮试探和交手，移动较多，有数次清晰的出拳与格挡；第二阶段约101—159秒，节奏有所加快，出现较密集的对攻和搂抱。蓝方在出拳距离控制上略占优势。”,
  “rounds”: [
    { “start_time_sec”: 0, “end_time_sec”: 101, “description”: “比赛前期，红蓝双方多以试探和单拳为主” },
    { “start_time_sec”: 101, “end_time_sec”: 159.46, “description”: “比赛中后期，对攻频率增加，出现搂抱” }
  ],
  “global_observations”: [
    “固定机位拍摄，画面稳定，无镜头切换”,
    “红方选手身穿红色背心，蓝方选手身穿蓝色背心”,
    “裁判身穿白色衬衫在台角附近移动”,
    “双方均佩戴头盔和拳击手套”
  ]
}
```

VLM 在没有时间戳和回合提示的情况下，仅靠画面内容就自动识别了场上角色、比赛节奏变化和阶段划分。

### 粗扫描：滑窗候选检测

对前 8 秒的 4 秒滑窗进行粗扫描（图片抽帧模式，均匀抽取 8 帧），Qwen 返回：

```json
{
  “events”: [
    {
      “relative_start_sec”: 2.286,
      “relative_end_sec”: 3.429,
      “fighter”: “blue”,
      “hand”: “rear”,
      “target_region”: “head”,
      “confidence”: 0.7
    }
  ]
}
```

粗扫描的职责是宁可漏检也不误报：这个阶段只判断”这里可能有一拳”，不判断命中与否。`confidence` 是对”存在出拳动作”的信心，不是命中信心。注意粗扫窗口偏宽（3.429 秒），这是合理的——粗扫只需要圈出候选区间，精确边界留给精扫描。

### 精扫描：证据级复核

对上一步候选区间 `[1.6s, 3.0s]`（加了 ±0.6s 填充）进行精扫描复核：

```json
{
  “is_punch”: “yes”,
  “start_time_sec”: 2.2,
  “peak_time_sec”: 2.27,
  “end_time_sec”: 2.35,
  “fighter”: “blue”,
  “hand”: “rear”,
  “punch_type”: “直拳”,
  “target_region”: “head”,
  “contact_evidence”: “clear”,
  “hit_or_miss”: “hit”,
  “blocked”: “no”,
  “occluded”: “no”,
  “confidence”: 0.95,
  “reason”: “在2.27秒帧中，蓝方选手右后手明显前伸击中红方头部左侧，拳套与红方头盔有清晰接触，红方头部可见受击后向左偏移；接触瞬间无遮挡、证据充分。”
}
```

精扫描不仅修正了粗扫描偏宽的结束时间（从 3.429s 收紧到 2.35s），还独立确认了出拳方、手部、拳型和命中结果。`reason` 字段用中文解释了判定依据，方便人工复核。

### 三种复核结果对比

同一视频的不同片段，精扫描会输出不同的判断：

| 结果 | 含义 | 典型理由 |
|------|------|---------|
| `hit` | 清晰接触 | “拳套与头部有可见接触，受击方出现身体偏移” |
| `miss` | 未接触 | “出拳方向明确但未触碰到对手，距离约一个拳套宽度” |
| `blocked` | 被格挡 | “蓝方直拳被红方前手格挡，拳路在接触前被拦截” |
| `uncertain` | 证据不足 | “画面中出现遮挡，无法判断是否接触，置信度较低” |

每条精扫描结果都包含 `contact_evidence`（`clear` / `possible` / `none` / `occluded`）和 `occluded` 字段，把”看不清”和”没打中”区分开。

### 效果总结

当前 MVP 的核心价值不是自动计分，而是把”手动拖进度条找动作”变成”CV 圈候选 → VLM 给证据 → 网页点一下回放”。一个 160 秒的视频，粗扫描大约 160 个窗口（4s 窗 / 1s 步长），候选合并后产生少量精扫描任务，每个精扫描返回结构化的、带中文理由的复核结论，可直接在网页时间轴上点击按钮跳转播放。

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

