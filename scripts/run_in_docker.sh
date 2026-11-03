#!/usr/bin/env bash
# Run one EvaLoop evaluation of an OpenRouter model inside the sandbox image.
#
# Usage: scripts/run_in_docker.sh <openrouter model id> <provider slug> <loop> [extra evaloop args]
#   e.g. scripts/run_in_docker.sh google/gemma-3-12b-it deepinfra/bf16 code_generation_pseudocode
#
# The API key is read from $KEY_FILE (default: openrouter.key) and passed to the container
# by variable name only, so it never appears on a command line. Results are written to
# ./results/<model>_<loop>_results.json and logs to ./logs/<model>_<loop>/; the container
# can see no other host directory.
set -euo pipefail

MODEL=$1 PROVIDER=$2 LOOP=$3
shift 3
KEY_FILE=${KEY_FILE:-openrouter.key}
IMAGE=${IMAGE:-evaloop-b1}
NAME=$(echo "${MODEL##*/}" | tr '.:' '__')

# One log directory per run, so parallel runs do not interleave in logs/evaloop.log
LOG_DIR="logs/${NAME}_${LOOP}"
mkdir -p results "$LOG_DIR"
# Git Bash on Windows needs a Windows-style path for the bind mounts
HOST_DIR="$(pwd -W 2>/dev/null || pwd)"

OPENROUTER_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
export OPENROUTER_API_KEY

MSYS_NO_PATHCONV=1 docker run --rm \
  --memory 6g --cpus 4 --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges \
  -e OPENROUTER_API_KEY \
  -v "$HOST_DIR/results:/app/results" \
  -v "$HOST_DIR/$LOG_DIR:/app/logs" \
  "$IMAGE" evaluate \
  --model "$MODEL" \
  --loop "$LOOP" \
  --api_base_url https://openrouter.ai/api/v1 \
  --provider_order "$PROVIDER" \
  --max_cycles 10 \
  --temperature 0.0 \
  --output_dir results/ \
  --experiment_name "$NAME" \
  "$@"
