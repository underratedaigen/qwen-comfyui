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
: "${COMFY_LOG_LEVEL:=INFO}"
: "${COMFY_HOST:=127.0.0.1:8188}"
: "${COMFY_STARTUP_TIMEOUT_S:=300}"
: "${COMFY_STARTUP_POLL_INTERVAL_S:=2}"

cd /comfyui

echo "qwen-v19-worker: starting ComfyUI"
if [ "${SERVE_API_LOCALLY:-false}" = "true" ]; then
  python -u main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
  echo $! > "${COMFY_PID_FILE}"
  echo "qwen-v19-worker: waiting for ComfyUI readiness"
  python3 - <<'PY'
import os
import time
import urllib.error
import urllib.request

comfy_host = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
timeout_s = float(os.environ.get("COMFY_STARTUP_TIMEOUT_S", "300"))
poll_interval_s = float(os.environ.get("COMFY_STARTUP_POLL_INTERVAL_S", "2"))
url = f"http://{comfy_host}/"
start = time.time()

while True:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 500:
                print(f"qwen-v19-worker: ComfyUI ready after {time.time() - start:.1f}s")
                break
    except urllib.error.URLError:
        if time.time() - start >= timeout_s:
            raise SystemExit(f"qwen-v19-worker: timed out waiting for ComfyUI after {timeout_s:.1f}s")
        time.sleep(poll_interval_s)
PY
  echo "qwen-v19-worker: starting RunPod handler with local API"
  python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  python -u main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
  echo $! > "${COMFY_PID_FILE}"
  echo "qwen-v19-worker: waiting for ComfyUI readiness"
  python3 - <<'PY'
import os
import time
import urllib.error
import urllib.request

comfy_host = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
timeout_s = float(os.environ.get("COMFY_STARTUP_TIMEOUT_S", "300"))
poll_interval_s = float(os.environ.get("COMFY_STARTUP_POLL_INTERVAL_S", "2"))
url = f"http://{comfy_host}/"
start = time.time()

while True:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 500:
                print(f"qwen-v19-worker: ComfyUI ready after {time.time() - start:.1f}s")
                break
    except urllib.error.URLError:
        if time.time() - start >= timeout_s:
            raise SystemExit(f"qwen-v19-worker: timed out waiting for ComfyUI after {timeout_s:.1f}s")
        time.sleep(poll_interval_s)
PY
  echo "qwen-v19-worker: starting RunPod handler"
  python -u /handler.py
fi
