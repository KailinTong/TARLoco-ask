#!/usr/bin/env bash
# Launch one W&B sweep agent per GPU inside a detached tmux session.
# The session survives SSH disconnect; reattach with `tmux attach -t sweep`.
#
# Usage:
#   scripts/sweeps/launch_all.sh <sweep_id> [num_gpus] [trials_per_round]
#
# Example:
#   scripts/sweeps/launch_all.sh kailintong-wb/TAR_workspace/abc123 4 8
#
# Common follow-ups:
#   tmux attach -t sweep            # watch live (Ctrl-b d to detach)
#   tmux ls                         # confirm session is up
#   tmux kill-session -t sweep      # stop everything

set -euo pipefail

SWEEP_ID="${1:?usage: launch_all.sh <sweep_id> [num_gpus] [trials_per_round]}"
NUM_GPUS="${2:-4}"
TRIALS_PER_ROUND="${3:-8}"
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

# Window 0: GPU 0
tmux new-session -d -s "${SESSION}" -n "gpu0" \
    "${AGENT_SCRIPT} 0 ${SWEEP_ID} ${TRIALS_PER_ROUND}"

# Windows 1..N-1
for ((i = 1; i < NUM_GPUS; i++)); do
    tmux new-window -t "${SESSION}:${i}" -n "gpu${i}" \
        "${AGENT_SCRIPT} ${i} ${SWEEP_ID} ${TRIALS_PER_ROUND}"
done

echo "Started ${NUM_GPUS} agents in tmux session '${SESSION}' for ${SWEEP_ID}."
echo "Attach: tmux attach -t ${SESSION}    (Ctrl-b d to detach, Ctrl-b n/p to switch windows)"
echo "Logs:   ${REPO_ROOT}/logs/sweeps/"
