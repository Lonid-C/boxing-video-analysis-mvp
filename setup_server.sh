#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-birdclef2026}"
CONDA_BIN="${CONDA_BIN:-/home/files/anaconda3/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "找不到 conda: $CONDA_BIN" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "复用已有 Conda 环境: $ENV_NAME"
else
  echo "创建 Conda 环境: $ENV_NAME"
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
fi

# 服务器已有 torch/opencv 时只补装轻量依赖，避免重装 CUDA Torch。
if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import ultralytics' >/dev/null 2>&1; then
  echo "安装 Ultralytics 及项目依赖"
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r "$PROJECT_DIR/requirements.txt"
else
  echo "Ultralytics 已存在，跳过重复安装"
fi

mkdir -p "$PROJECT_DIR"/{inputs,models,outputs}
"$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY'
import importlib.util
mods = ["torch", "cv2", "ultralytics"]
for name in mods:
    print(f"{name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
try:
    import torch
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch probe failed: {exc}")
PY

echo "安装完成。请把 MP4 放入 $PROJECT_DIR/inputs/ 后运行 README.md 中的命令。"
