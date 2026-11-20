#!/usr/bin/env bash
set -euo pipefail
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL first}"
uv run python -m autoresearch_evo run --provider openai --model "$OPENAI_MODEL" --hours 8
