#!/usr/bin/env bash
# Run the whole device suite for one device: IGL_BENCH_DEVICE=<cpu|mps|cuda> benchmarks/device/run_all.sh [python]
set -euo pipefail
PY=${1:-python}
DEV=${IGL_BENCH_DEVICE:-cpu}
echo "== device suite on $DEV =="
$PY -m benchmarks.device.regions --no-gate
$PY -m benchmarks.device.syncs --no-gate
$PY -m benchmarks.device.components --no-gate
$PY -m benchmarks.device.profile --no-gate
$PY -m benchmarks.device.e2e --no-gate
if [ "$DEV" = cuda ]; then $PY -m benchmarks.device.epoch_modes --no-gate; $PY -m benchmarks.device.launch_bound --no-gate; fi
