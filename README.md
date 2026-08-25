# Boxing Video Analysis MVP

**本项目供竞体中心分析demo**

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

以下结果全部来自真实奥运拳击比赛视频（Olympic Boxing Punch Classification 数据集，固定机位、红蓝双方 + 裁判），使用阿里云百炼 Qwen-VL（`qwen3-vl-plus`）、本地抽帧图片模式生成。

### 批量验证：5 个真实视频

2026-08-12 在服务器上跑了 5 个真实比赛视频（时长 121~258 秒不等），每个视频执行”整场摘要 → 前 8 秒粗扫描 → 精扫描复核”：

| 视频 | 时长 | 精扫描判定 | 说明 |
|------|------|-----------|------|
| GH088416 | 250.5s | 2 × `hit`（直拳，命中头右侧，conf 0.95） | 红方主动压迫，蓝方移动反击 |
| GH098416 | 121.4s | 2 × `hit`（直拳，命中头左侧，conf 0.95/0.98） | 蓝方进攻主导，红方防守反击 |
| GH108416 | 146.0s | 1 × `hit` + 1 × `uncertain` | 有 1 拳被遮挡，模型正确拒判 |
| GH118416 | 258.4s | 1 × `uncertain` + 1 × `hit`（勾拳，命中躯干） | 模型识别出候选错误并纠正 |
| GH128416 | 131.0s | 2 × `hit`（直拳，命中头右侧，conf 0.95） | 红方连击命中蓝方头部 |

合计 10 次精扫描：**8 次 `hit`、2 次 `uncertain`、0 次误报为 hit**。其中 GH118416 的候选区间实际没有出拳，模型明确给出 `uncertain` 而不是硬判；GH108416 因蓝方头部被护具遮挡，模型判定证据不足。这说明”证据优先”提示词在真实数据上有效：该判就判、看不清就不判。

### 整场摘要（升级版）

升级后的摘要不再只有一句概述，而是输出**阶段划分、关键时刻、双方风格、出拳量级估计**。以 GH098416（121.4s）为例：

```json
{
  “summary”: “开场双方试探（0-17s），随后蓝方主动前压、高频出拳，红方以防守反击为主；34-69秒进入对攻期，蓝方多次突进并命中头部，红方偶有还击但多被格挡；86s后节奏放缓，双方移动减少，出现搂抱与裁判介入。”,
  “phases”: [
    { “start_time_sec”: 0.0, “end_time_sec”: 17.346, “label”: “试探期”, “evidence”: “direct” },
    { “start_time_sec”: 17.346, “end_time_sec”: 69.383, “label”: “对攻期”, “evidence”: “direct” },
    { “start_time_sec”: 69.383, “end_time_sec”: 86.729, “label”: “消耗期”, “evidence”: “direct” }
  ],
  “key_moments”: [
    { “time_sec”: 52.037, “description”: “蓝方连续三拳（刺拳-摆拳-勾拳）组合进攻，红方低头躲闪后退至围绳”, “evidence”: “direct” }
  ],
  “fighter_analysis”: {
    “red”:  { “style”: “防守反击型（推断）”, “punch_volume”: “medium”, “observations”: [“可见多次格挡与低头躲闪动作”, “多数时间保持中远距离”] },
    “blue”: { “style”: “进攻主导型（推断）”, “punch_volume”: “high”, “observations”: [“17.346s起持续前压，步频明显高于红方”, “多次击中红方头部/躯干”] }
  }
}
```

模型在没有时间戳、没有回合提示的情况下，仅凭画面就划出了阶段边界（17.3s、69.4s）、定位了关键时刻（52.0s 的连击）、并对双方风格和出拳量级做了标注为”推断”的分析。

### 精扫描（升级版字段）

升级后的精扫描增加 `impact_area`（命中部位）、`reaction`（受击反应）、`power`（力度感）、`evidence`（证据来源）等维度。以 GH098416 的命中为例：

```json
{
  “is_punch”: “yes”,
  “start_time_sec”: 2.196,
  “peak_time_sec”: 2.449,
  “end_time_sec”: 2.701,
  “fighter”: “red”,
  “hand”: “rear”,
  “punch_type”: “直拳”,
  “amplitude”: “full”,
  “target_region”: “head”,
  “impact_area”: “head_left”,
  “contact_evidence”: “clear”,
  “hit_or_miss”: “hit”,
  “blocked”: “no”,
  “block_type”: “无”,
  “reaction”: “明显偏移”,
  “power”: “medium”,
  “part_of_combo”: “no”,
  “countered”: “no”,
  “occluded”: “no”,
  “confidence”: 0.98,
  “evidence”: “direct”,
  “reason”: “在2.196秒红方后手启动出拳，2.449秒拳套清晰接触蓝方头部左侧（可见拳套与头盔接触瞬间及蓝方头部向右明显偏移），2.701秒红方手臂已回撤；全程无格挡动作，裁判未介入；蓝方未做摇闪或后撤，属直接命中。”
}
```

注意精扫描给出了**逐帧的证据链**（启动→接触→回撤三阶段时间戳），比”发生了命中”更有复盘价值。

### 边界情况：模型正确拒判

不是所有候选都是真拳。GH118416 的候选区间实际没有任何出拳动作，模型返回：

```json
{
  “is_punch”: “no”,
  “hit_or_miss”: “uncertain”,
  “confidence”: 0.1,
  “reason”: “蓝方选手始终处于防守姿态，红方仅做小幅步法调整与手部微动，无任何手臂伸展加速动作；候选标注的0.57–1.14秒区间内画面无出拳行为。”
}
```

GH108416 中一拳因蓝方头部被护具和手臂遮挡，模型给出 `uncertain` + `occluded`，而不是硬猜。**看不清时不硬判**，是这套管线和裸 CV 规则检测的关键差异。

### 效果总结

当前 MVP 的核心价值不是自动计分，而是把”手动拖进度条找动作”变成”CV 圈候选 → VLM 给证据 → 网页点一下回放”。10 次精扫描在 5 个不同比赛视频上做到 8 次证据充分的 `hit` 判定、2 次诚实的 `uncertain`，每次判定都附带可核对的中文证据链。

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

`--vlm` 是显式费用开关：即使环境中已经存在 API Key，不传该参数也不会调用百炼。
扫描默认使用 `CV 候选 → VLM 精扫`，调用量约等于 CV 候选数。只有需要脱离 CV
主动扫描整段视频时才增加 `--vlm-scan-source coarse`；该模式会按滑窗产生较多调用。

分析默认还会在 stats 文件同级生成 `<名称>_package/` 完整结果包，包括三张 CSV、
红蓝双方移动热力图、HTML 报告和带 10 秒总结片尾的视频。使用 `--no-export`
关闭，或用 `--export-dir` 指定目录、`--end-card-seconds` 调整每页片尾时长。

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
GET  /api/v1/runs/{run_id}/artifacts
```

`artifacts` 会返回 CSV、热力图、HTML 和总结视频的类型、MIME 类型及媒体 URL。

Web 后台默认一次只执行一个分析任务，避免同时占满 GPU、内存和 VLM 配额；需要提高并发时
设置 `BOXING_MAX_CONCURRENT_RUNS`。每个任务使用 `outputs/runs/<run_id>/` 独立目录。

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
