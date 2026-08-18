#!/usr/bin/env bash
# Prepare a pinned, Modal-free BrowseComp+ environment on an RTX 5090 node.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_ENV="${PYTHON_ENV:-.venv}"

if [ ! -x "$UV_BIN" ]; then
    echo "ERROR: uv is not executable: $UV_BIN" >&2
    exit 2
fi

"$UV_BIN" venv --python 3.11 "$PYTHON_ENV"
"$UV_BIN" pip install \
    --python "$PYTHON_ENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.7.1'
"$UV_BIN" pip install \
    --python "$PYTHON_ENV/bin/python" \
    -r reproduction/requirements-browsecomp-5090.txt

"$PYTHON_ENV/bin/python" - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
architectures = torch.cuda.get_arch_list()
if "RTX 5090" not in name:
    raise SystemExit(f"expected RTX 5090, found {name}")
if capability != (12, 0):
    raise SystemExit(f"expected compute capability 12.0, found {capability}")
if "sm_120" not in architectures:
    raise SystemExit(f"PyTorch wheel has no sm_120 support: {architectures}")
print(json.dumps({
    "cuda": True,
    "device": name,
    "capability": capability,
    "torch": torch.__version__,
    "sm_120": True,
}))
PY
