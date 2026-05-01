#!/usr/bin/env bash
# Single-GPU W&B sweep agent with crash-restart loop.
#
# Usage:
#   scripts/sweeps/launch_agent.sh <gpu_id> <sweep_id> [trials_per_round]
#
# Example:
#   scripts/sweeps/launch_agent.sh 0 kailintong-wb/TAR_workspace/abc123 8
#
# The agent loops: run up to `trials_per_round` trials, then exit and
# restart. This means a crashed/OOM agent self-recovers, and the loop
# stops cleanly only when the sweep itself is exhausted (the agent gets
# `RUN_CAP_COMPLETED` from the W&B server and exits 0 with no work left).

set -u

GPU_ID="${1:?usage: launch_agent.sh <gpu_id> <sweep_id> [trials_per_round]}"
SWEEP_ID="${2:?usage: launch_agent.sh <gpu_id> <sweep_id> [trials_per_round]}"
TRIALS_PER_ROUND="${3:-8}"

# --- env activation ---------------------------------------------------
# Conda needs `conda activate` to be a function, which only the init
# script provides. `source`ing it is the safe way inside non-login shells.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate tarloco

# --- repo root --------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- logging ----------------------------------------------------------
LOG_DIR="${REPO_ROOT}/logs/sweeps"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/agent_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# Pin Isaac Sim / Omniverse caches per-GPU so concurrent agents don't
# race on the same compile cache directory.
export OV_CACHE_DIR="${HOME}/.cache/ov_gpu${GPU_ID}"
export NVIDIA_KERNEL_CACHE_PATH="${HOME}/.cache/nvidia_gpu${GPU_ID}"
mkdir -p "${OV_CACHE_DIR}" "${NVIDIA_KERNEL_CACHE_PATH}"

echo "[$(date -Is)] starting agent on GPU ${GPU_ID} for sweep ${SWEEP_ID}" \
    | tee -a "${LOG_FILE}"

trap 'echo "[$(date -Is)] received SIGTERM, exiting"; exit 0' SIGTERM SIGINT

# --- restart loop -----------------------------------------------------
# `wandb agent --count N` exits when N trials have run OR the sweep is
# finished. We re-enter the loop on the former and break on the latter.
# A non-zero exit (e.g. CUDA OOM, network blip) sleeps and retries.
while true; do
    if wandb agent --count "${TRIALS_PER_ROUND}" "${SWEEP_ID}" 2>&1 \
           | tee -a "${LOG_FILE}"; then
        # Probe sweep status: if marked finished, stop looping.
        STATE="$(wandb sweep --status "${SWEEP_ID}" 2>/dev/null \
                  | awk -F': *' '/state/{print $2; exit}')"
        if [[ "${STATE}" == "FINISHED" || "${STATE}" == "CANCELLED" ]]; then
            echo "[$(date -Is)] sweep ${SWEEP_ID} state=${STATE}, stopping" \
                | tee -a "${LOG_FILE}"
            break
        fi
        echo "[$(date -Is)] round done, starting next round" \
            | tee -a "${LOG_FILE}"
    else
        echo "[$(date -Is)] agent crashed (rc=$?), restarting in 30s" \
            | tee -a "${LOG_FILE}"
        sleep 30
    fi
done
