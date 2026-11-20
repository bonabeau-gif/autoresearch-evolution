#!/usr/bin/env bash
set -eu
printf 'Python:\n'; python --version
printf '\nuv:\n'; uv --version || true
printf '\nGPU:\n'; nvidia-smi --query-gpu=name,memory.total --format=csv || true
printf '\nTorch + CUDA:\n'; python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available())' || true
printf '\nData cache:\n'; ls -lah "$HOME/.cache/autoresearch" 2>/dev/null || true
