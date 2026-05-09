#!/usr/bin/env bash
# Launch W&B sweep agents across GPUs inside a detached tmux session.
# The session survives SSH disconnect; reattach with `tmux attach -t sweep`.
#
# Usage:
#   scripts/sweeps/launch_all.sh <sweep_id> [num_gpus] [trials_per_round] [agents_per_gpu]
#
# Examples:
#   scripts/sweeps/launch_all.sh kailintong-wb/TAR_workspace/abc123 4 8    # 1 agent/GPU
#   scripts/sweeps/launch_all.sh kailintong-wb/TAR_workspace/abc123 4 4 2  # 2 agents/GPU
#
# Common follow-ups:
#   tmux attach -t sweep            # watch live (Ctrl-b d to detach)
#   tmux ls                         # confirm session is up
#   tmux kill-session -t sweep      # stop everything

set -euo pipefail

SWEEP_ID="${1:?usage: launch_all.sh <sweep_id> [num_gpus] [trials_per_round] [agents_per_gpu]}"
NUM_GPUS="${2:-4}"
TRIALS_PER_ROUND="${3:-8}"
AGENTS_PER_GPU="${4:-1}"
SESSION="sweep"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AGENT_SCRIPT="${REPO_ROOT}/scripts/sweeps/launch_agent.sh"
chmod +x "${AGENT_SCRIPT}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; install it or use the slurm path." >&2
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists." >&2
    echo "Attach with:  tmux attach -t ${SESSION}" >&2
    echo "Or kill with: tmux kill-session -t ${SESSION}" >&2
    exit 1
fi

TOTAL_AGENTS=$(( NUM_GPUS * AGENTS_PER_GPU ))
WINDOW=0

for ((g = 0; g < NUM_GPUS; g++)); do
    for ((a = 0; a < AGENTS_PER_GPU; a++)); do
        NAME="gpu${g}_${a}"
        SLOT="${g}_${a}"
        if (( WINDOW == 0 )); then
            tmux new-session -d -s "${SESSION}" -n "${NAME}" \
                "${AGENT_SCRIPT} ${g} ${SWEEP_ID} ${TRIALS_PER_ROUND} ${SLOT}"
        else
            tmux new-window -t "${SESSION}:${WINDOW}" -n "${NAME}" \
                "${AGENT_SCRIPT} ${g} ${SWEEP_ID} ${TRIALS_PER_ROUND} ${SLOT}"
        fi
        (( WINDOW++ ))
    done
done

echo "Started ${TOTAL_AGENTS} agents (${AGENTS_PER_GPU}/GPU × ${NUM_GPUS} GPUs) for ${SWEEP_ID}."
echo "Attach: tmux attach -t ${SESSION}    (Ctrl-b d to detach, Ctrl-b n/p to switch windows)"
echo "Logs:   ${REPO_ROOT}/logs/sweeps/"
