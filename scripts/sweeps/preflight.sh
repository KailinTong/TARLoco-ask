#!/usr/bin/env bash
# Pre-flight sanity checks for the W&B sweep on the cluster.
#
# Usage:
#   scripts/sweeps/preflight.sh [sweep_id]
#
# With no args: checks env, imports, GPUs, wandb auth, tmux, repo, disk,
# network, and that train.py --help runs (argparse smoke test).
# With a sweep_id: also probes that the sweep is accessible via the W&B API.
#
# Run this BEFORE launch_all.sh.

set -u

PASS=0
FAIL=0
WARN=0

if [[ -t 1 ]]; then
    GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; RST=$'\033[0m'
else
    GREEN=""; RED=""; YEL=""; RST=""
fi

ok()   { echo "  ${GREEN}✓${RST} $1"; PASS=$((PASS+1)); }
bad()  { echo "  ${RED}✗${RST} $1"; FAIL=$((FAIL+1)); }
warn() { echo "  ${YEL}!${RST} $1"; WARN=$((WARN+1)); }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --------------------------------------------------------------------
# 1. conda env
# --------------------------------------------------------------------
echo "[env]"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    CONDA_ENV="${TARLOCO_CONDA_ENV:-tar}"
    if conda activate "${CONDA_ENV}" 2>/dev/null; then
        ok "conda activate ${CONDA_ENV}  (python: $(command -v python))"
    else
        bad "conda activate ${CONDA_ENV} failed"
        echo
        echo "Aborting: cannot activate env."
        exit 1
    fi
else
    bad "conda init script not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

# --------------------------------------------------------------------
# 2. python imports (lightweight only — no Isaac Sim app launch)
# --------------------------------------------------------------------
echo "[imports]"
ERR_FILE="$(mktemp)"
if python - >/dev/null 2>"${ERR_FILE}" <<'PY'
import importlib, sys
mods = ["torch", "wandb", "gymnasium", "toml", "isaaclab", "isaaclab.app"]
fail = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        fail.append((m, repr(e)))
if fail:
    for m, e in fail:
        print(f"{m}: {e}", file=sys.stderr)
    sys.exit(1)
PY
then
    ok "imports: torch, wandb, gymnasium, toml, isaaclab"
else
    bad "import failure(s):"
    sed 's/^/      /' "${ERR_FILE}"
fi
rm -f "${ERR_FILE}"

# --------------------------------------------------------------------
# 3. GPUs (nvidia-smi + torch agree)
# --------------------------------------------------------------------
echo "[gpu]"
if command -v nvidia-smi >/dev/null; then
    NVSMI_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    GPU_NAMES="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd '; ' -)"
    ok "nvidia-smi sees ${NVSMI_COUNT} GPU(s): ${GPU_NAMES}"
    if [[ "${NVSMI_COUNT}" -lt 4 ]]; then
        warn "expected 4 GPUs, found ${NVSMI_COUNT} (launch_all.sh defaults to 4)"
    fi
else
    bad "nvidia-smi not on PATH"
    NVSMI_COUNT=0
fi

CUDA_PROBE="$(python -c 'import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())' 2>/dev/null)"
read -r CUDA_AVAIL TORCH_COUNT <<<"${CUDA_PROBE:-0 0}"
if [[ "${CUDA_AVAIL}" == "1" ]]; then
    ok "torch.cuda available, device_count=${TORCH_COUNT}"
    if [[ "${NVSMI_COUNT}" != "0" && "${TORCH_COUNT}" != "${NVSMI_COUNT}" ]]; then
        warn "torch sees ${TORCH_COUNT} but nvidia-smi sees ${NVSMI_COUNT} — CUDA_VISIBLE_DEVICES may already be set"
    fi
else
    bad "torch.cuda.is_available() == False"
fi

# --------------------------------------------------------------------
# 4. wandb auth
# --------------------------------------------------------------------
echo "[wandb]"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    ok "WANDB_API_KEY is set in env"
elif [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    ok "~/.netrc has api.wandb.ai entry"
else
    bad "no wandb credentials found (run: wandb login)"
fi

if python -c "import wandb; wandb.login(anonymous='never', timeout=10)" >/dev/null 2>&1; then
    WB_USER="$(python -c "import wandb,sys; api=wandb.Api(); print(api.viewer.username)" 2>/dev/null)"
    ok "wandb.login() ok (user: ${WB_USER:-unknown})"
else
    bad "wandb.login() failed — credentials present but server rejected them"
fi

# --------------------------------------------------------------------
# 5. tmux
# --------------------------------------------------------------------
echo "[tmux]"
if command -v tmux >/dev/null; then
    ok "tmux $(tmux -V | awk '{print $2}')"
    if tmux has-session -t sweep 2>/dev/null; then
        warn "a tmux session named 'sweep' already exists — launch_all.sh will refuse to start"
    fi
else
    bad "tmux not installed (apt install tmux, or use slurm)"
fi

# --------------------------------------------------------------------
# 6. repo files
# --------------------------------------------------------------------
echo "[repo]"
for f in standalone/tarloco/train.py \
         scripts/sweeps/sweep_him_rough.yaml \
         scripts/sweeps/launch_agent.sh \
         scripts/sweeps/launch_all.sh; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
        ok "${f}"
    else
        bad "missing: ${f}"
    fi
done
for f in scripts/sweeps/launch_agent.sh scripts/sweeps/launch_all.sh; do
    if [[ -x "${REPO_ROOT}/${f}" ]]; then
        ok "${f} is executable"
    else
        warn "${f} is not executable (chmod +x ${f})"
    fi
done

# --------------------------------------------------------------------
# 7. train.py argparse smoke test (does NOT launch Isaac Sim)
# --------------------------------------------------------------------
echo "[train.py --help]"
if timeout 30 python standalone/tarloco/train.py --help >/dev/null 2>"${ERR_FILE:=/tmp/preflight_train.err}"; then
    ok "standalone/tarloco/train.py --help exits cleanly"
else
    bad "standalone/tarloco/train.py --help failed (see ${ERR_FILE})"
fi

# --------------------------------------------------------------------
# 8. sweep yaml parses + agent forwarding format
# --------------------------------------------------------------------
echo "[sweep yaml]"
if python -c "import yaml,sys; yaml.safe_load(open('scripts/sweeps/sweep_him_rough.yaml'))" 2>/dev/null; then
    ok "sweep_him_rough.yaml parses as YAML"
else
    bad "sweep_him_rough.yaml does not parse"
fi
if grep -q "args_no_hyphens" scripts/sweeps/sweep_him_rough.yaml; then
    ok "sweep YAML uses \${args_no_hyphens} (hydra-compatible forwarding)"
else
    warn "sweep YAML may not forward args in hydra format — check command: block"
fi

# --------------------------------------------------------------------
# 9. disk (~200 GB recommended for full sweep with checkpoints)
# --------------------------------------------------------------------
echo "[disk]"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
AVAIL_GB="$(df -BG --output=avail "${LOG_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
AVAIL_GB="${AVAIL_GB:-0}"
if [[ "${AVAIL_GB}" -ge 200 ]]; then
    ok "${AVAIL_GB} GB free under ${LOG_DIR}"
elif [[ "${AVAIL_GB}" -ge 50 ]]; then
    warn "${AVAIL_GB} GB free under ${LOG_DIR} — full sweep may need ~200 GB"
else
    bad "${AVAIL_GB} GB free under ${LOG_DIR} — symlink logs/ to scratch first"
fi

# --------------------------------------------------------------------
# 10. network (api.wandb.ai reachable)
# --------------------------------------------------------------------
echo "[network]"
if curl -sSf --max-time 5 -o /dev/null https://api.wandb.ai 2>/dev/null; then
    ok "https://api.wandb.ai reachable"
else
    warn "api.wandb.ai not reachable — set WANDB_MODE=offline and sync later"
fi

# --------------------------------------------------------------------
# 11. (optional) sweep id probe
# --------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    echo "[sweep $1]"
    SWEEP_ID="$1"
    SWEEP_ERR="$(mktemp)"
    if python - >/dev/null 2>"${SWEEP_ERR}" <<PY
import wandb, sys
api = wandb.Api()
s = api.sweep("${SWEEP_ID}")
print(f"name={s.name} state={s.state} runs={len(list(s.runs))}")
PY
    then
        ok "sweep ${SWEEP_ID} accessible"
    else
        bad "cannot access sweep ${SWEEP_ID}:"
        sed 's/^/      /' "${SWEEP_ERR}"
    fi
    rm -f "${SWEEP_ERR}"
fi

# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------
echo
echo "=================================================="
printf "  %spassed: %d%s    %swarned: %d%s    %sfailed: %d%s\n" \
    "${GREEN}" "${PASS}" "${RST}" \
    "${YEL}"   "${WARN}" "${RST}" \
    "${RED}"   "${FAIL}" "${RST}"
echo "=================================================="

if [[ ${FAIL} -gt 0 ]]; then
    echo "Fix the failures above before running launch_all.sh."
    exit 1
fi
if [[ ${WARN} -gt 0 ]]; then
    echo "Warnings above are non-blocking. Review them before launching."
fi
exit 0
