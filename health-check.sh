#!/usr/bin/env bash
# Robust health check: HTTP + GPU
set -Eeuo pipefail

HTTP_OK=0
GPU_OK=0

# 1) HTTP health (tight timeouts)
if curl -fsS --max-time 4 --connect-timeout 10 http://127.0.0.1:8189/health >/dev/null; then
  HTTP_OK=1
fi

# 2) GPU health via NVML; fallback to nvidia-smi
PYTHON_NVML="$(python3 - <<'PY' 2>/dev/null || true
try:
    import sys
    import pynvml as n
    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(0)
    _ = n.nvmlDeviceGetTemperature(h, 0)
    print("OK")
except Exception as e:
    print("FAIL:", e)
    sys.exit(1)
PY
)"
if [[ "$PYTHON_NVML" == "OK" ]]; then
  GPU_OK=1
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_OK=1
fi

if [[ $HTTP_OK -eq 1 && $GPU_OK -eq 1 ]]; then
  echo "HEALTHY: http+gpu"
  exit 0
fi

echo "UNHEALTHY: http=$HTTP_OK gpu=$GPU_OK"
exit 1
