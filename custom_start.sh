#!/usr/bin/env bash
set -euo pipefail

TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d' | head -n 1 || true)"
if [ -n "${TCMALLOC}" ]; then
  export LD_PRELOAD="${TCMALLOC}"
fi

echo "qwen-v19-worker: checking GPU availability"
python - <<'PY'
import torch

try:
    torch.cuda.init()
    print(f"qwen-v19-worker: GPU ready - {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"qwen-v19-worker: GPU init failed - {exc}")
    raise
PY

echo "qwen-v19-worker: bootstrapping models and patches"
python /bootstrap_models.py

if command -v comfy-manager-set-mode >/dev/null 2>&1; then
  comfy-manager-set-mode offline || true
fi

COMFY_PID_FILE="/tmp/comfyui.pid"
cd /comfyui

if [ "${SERVE_API_LOCALLY:-false}" = "true" ]; then
  python -u main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
  echo $! > "${COMFY_PID_FILE}"
  echo "qwen-v19-worker: starting RunPod handler with local API"
  python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  python -u main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
  echo $! > "${COMFY_PID_FILE}"
  echo "qwen-v19-worker: starting RunPod handler"
  python -u /handler.py
fi

