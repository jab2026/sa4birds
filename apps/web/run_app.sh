#!/usr/bin/env bash
# Launch the Gradio demo (app.py). Torch-free: onnxruntime + librosa/numpy.
# Everything it needs except an ONNX export lives under this directory; point
# --ckpt at an ../../export_onnx.py output to serve a different checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# onnxruntime-gpu does not put its pip-installed CUDA/cuDNN (nvidia-*-cu12) on the
# dynamic linker's search path the way torch does for itself. Silent no-op on the
# CPU-only install.
_nvidia_libs() {
  local dir
  for dir in "$1"/lib/python3.*/site-packages/nvidia; do
    [ -d "$dir" ] && find "$dir" -maxdepth 2 -type d -name lib 2>/dev/null
  done | paste -sd: -
  return 0
}

if [ -x ./venv/bin/python ]; then
  nvlibs="$(_nvidia_libs ./venv)"
  export LD_LIBRARY_PATH="${nvlibs}${nvlibs:+:}${LD_LIBRARY_PATH:-}"
  exec ./venv/bin/python app.py "$@"
else
  # Whatever interpreter is already active, assumed to have requirements.txt.
  exec python3 app.py "$@"
fi
