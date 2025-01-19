#!/usr/bin/env bash
#
# Usage:
#   ./train_models.sh [TARGET_VARIABLE] [NUM_MODELS]
#
# Example:
#   ./train_models.sh energy_U0 5
#

set -euo pipefail  # safer script execution

# Maximum number of concurrent processes
MAX_JOBS=2

# Default arguments
TARGET_VARIABLE=${1:-"energy_U0"}
NUM_MODELS=${2:-5}

echo "Training models with a maximum of $MAX_JOBS concurrent processes..."
echo "Target variable: $TARGET_VARIABLE"
echo "Number of models: $NUM_MODELS"

# Array to hold the PIDs of running jobs
pids=()

# Function to start a job in the background and track its PID
run_job() {
  echo "Starting job: $*"
  "$@" &
  local job_pid=$!
  pids+=("$job_pid")
}

# Function to wait for *one* job to finish, then remove its PID from `pids`
wait_for_any_job() {
  # Wait for one (any) background job to finish
  wait -n

  # Clean out PIDs of jobs that have finished
  local new_pids=()
  for pid in "${pids[@]}"; do
    # kill -0 checks if the process is still running
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    fi
  done
  pids=("${new_pids[@]}")
}

# Main loop
for MODEL_NUMBER in $(seq 1 "$NUM_MODELS"); do
  # 1) Wait until we have fewer than MAX_JOBS active processes
  while [ "${#pids[@]}" -ge "$MAX_JOBS" ]; do
    wait_for_any_job
  done
  run_job python3 -m torch_main.py "$TARGET_VARIABLE" "$MODEL_NUMBER" false

  # 2) Again wait if we're at the concurrency limit before launching next
  while [ "${#pids[@]}" -ge "$MAX_JOBS" ]; do
    wait_for_any_job
  done
  run_job python3 -m torch_main.py "$TARGET_VARIABLE" "$MODEL_NUMBER" true
done

# Finally, wait for all remaining jobs
while [ "${#pids[@]}" -gt 0 ]; do
  wait_for_any_job
done

echo "All training processes completed."
