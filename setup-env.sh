#!/usr/bin/env bash
#
# Set up a Python venv with CUDA-enabled llama-cpp-python (built from source)
# plus all runtime dependencies for this project.
#
# Tested target: Ubuntu 24.04, Python 3.12, CUDA toolkit 12.0, RTX 4080 SUPER.
#
# Usage:
#   ./setup-env.sh                 # full setup into ./venv
#   VENV_DIR=.venv ./setup-env.sh  # custom venv location
#   SKIP_APT=1 ./setup-env.sh      # skip the system (apt) packages step
#   CUDA_ARCH=86 ./setup-env.sh    # override GPU compute capability (e.g. 3090=86)
#
set -euo pipefail

# ── Configuration (override via environment) ──────────────────────────────────
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
# GPU compute capability without the dot. RTX 4080 SUPER (Ada) = 8.9 -> 89.
CUDA_ARCH="${CUDA_ARCH:-89}"
# PyTorch CUDA wheel channel. cu124 suits a recent driver; drop to cu121 if needed.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
SKIP_APT="${SKIP_APT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. Prerequisite checks ────────────────────────────────────────────────────
say "Checking prerequisites"
command -v "$PYTHON" >/dev/null || die "$PYTHON not found"
command -v nvcc >/dev/null || die "nvcc (CUDA toolkit) not found on PATH — needed to build llama-cpp-python with CUDA"
nvcc --version | grep -i release || true
"$PYTHON" --version

# nvcc 12.0 cannot use gcc > 12. Pick a compatible host compiler for the CUDA build.
CUDA_HOST_CXX=""
for cand in g++-12 g++-11 g++; do
  if command -v "$cand" >/dev/null; then
    ver="$(${cand} -dumpversion | cut -d. -f1)"
    if [ "$ver" -le 12 ]; then CUDA_HOST_CXX="$(command -v "$cand")"; break; fi
  fi
done
[ -n "$CUDA_HOST_CXX" ] || die "No gcc/g++ <= 12 found (need it for nvcc). Install with: sudo apt install g++-12"
say "Using CUDA host compiler: $CUDA_HOST_CXX"

# ── 2. System packages (audio / build libs) ───────────────────────────────────
if [ "$SKIP_APT" != "1" ] && command -v apt-get >/dev/null; then
  say "Installing system packages (sudo apt)"
  sudo apt-get update -qq
  sudo apt-get install -y \
    build-essential g++-12 \
    libportaudio2 portaudio19-dev \
    libsndfile1 \
    espeak-ng \
    ffmpeg
else
  say "Skipping apt step (SKIP_APT=1 or apt-get unavailable)"
fi

# ── 3. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  say "Creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
say "Upgrading pip toolchain"
python -m pip install --upgrade pip setuptools wheel
# Build backend tools for the from-source llama-cpp-python compile.
python -m pip install --upgrade cmake ninja scikit-build-core

# ── 4. PyTorch (CUDA build) ───────────────────────────────────────────────────
say "Installing PyTorch (CUDA wheels from $TORCH_INDEX)"
python -m pip install --upgrade torch --index-url "$TORCH_INDEX"

# ── 5. Other Python dependencies ──────────────────────────────────────────────
# requirements.txt is the source of truth. Install everything EXCEPT
# llama-cpp-python (built from source below, with CUDA) and torch (installed
# from the CUDA wheel index in step 4).
say "Installing project Python dependencies (from requirements.txt)"
[ -f requirements.txt ] || die "requirements.txt not found next to this script"
grep -ivE '^\s*(llama-cpp-python|torch)\b' requirements.txt > /tmp/req-no-llama.txt
python -m pip install -r /tmp/req-no-llama.txt

# ── 6. llama-cpp-python with CUDA (compiled from source, LAST) ─────────────────
say "Building llama-cpp-python with CUDA (arch=$CUDA_ARCH) — this takes a few minutes"
CMAKE_ARGS="-DGGML_CUDA=on \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -DCMAKE_CUDA_HOST_COMPILER=${CUDA_HOST_CXX}" \
FORCE_CMAKE=1 \
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  --no-binary llama-cpp-python llama-cpp-python

# ── 7. Verify ─────────────────────────────────────────────────────────────────
say "Verifying installation"
python - <<'PYEOF'
import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")

import llama_cpp
print(f"llama_cpp {llama_cpp.__version__}")
try:
    gpu = llama_cpp.llama_supports_gpu_offload()
    print(f"  GPU offload supported (CUDA compiled in): {gpu}")
    assert gpu, "llama-cpp-python was NOT built with GPU support!"
except AttributeError:
    print("  (could not query GPU offload flag on this version)")
PYEOF

say "Done. Activate with:  source $VENV_DIR/bin/activate"
