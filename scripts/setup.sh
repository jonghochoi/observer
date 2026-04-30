#!/usr/bin/env bash
#
# OBSERVER — first-run setup helper.
#
# Installs observer in editable mode and verifies the surrounding
# environment that observer cannot install for you (ffmpeg, Isaac Lab,
# W&B login). After this script passes, run `observer doctor` to
# verify your eval_config.yaml.
#
# Usage:
#     ./scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GREEN=$'\e[92m'; YELLOW=$'\e[93m'; RED=$'\e[91m'; DIM=$'\e[2m'; RESET=$'\e[0m'
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }
fail() { echo "  ${RED}✗${RESET} $*" >&2; exit 1; }

echo "── OBSERVER setup ─────────────────────────────────────────────────"

# Python version
py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$py_ver" in
    3.1[0-9]|3.2[0-9]) ok "Python $py_ver" ;;
    *) fail "Python $py_ver — observer requires >= 3.10" ;;
esac

# Editable install
if python3 -c 'import observer' 2>/dev/null && command -v observer >/dev/null; then
    ok "observer already installed (console script: $(command -v observer))"
else
    echo "  ${DIM}installing observer (pip install -e .)...${RESET}"
    if ! pip install -e . > /tmp/observer-pip.log 2>&1; then
        fail "pip install failed — see /tmp/observer-pip.log"
    fi
    ok "observer installed"
fi

# ffmpeg (required for video stage)
if command -v ffmpeg >/dev/null; then
    ok "ffmpeg in PATH"
else
    warn "ffmpeg not found — sudo apt install ffmpeg (needed for video stage)"
fi

# CUDA
if command -v nvidia-smi >/dev/null; then
    ok "nvidia-smi available"
else
    warn "nvidia-smi not found — CPU-only mode"
fi

# Isaac Lab launcher (only used by record stage)
if [ -n "${ISAACLAB_PATH:-}" ]; then
    if [ -x "$ISAACLAB_PATH/isaaclab.sh" ]; then
        ok "ISAACLAB_PATH=$ISAACLAB_PATH"
    else
        warn "ISAACLAB_PATH set but $ISAACLAB_PATH/isaaclab.sh is missing"
    fi
else
    warn "ISAACLAB_PATH not set — needed only when video recording is on"
fi

# Optional: W&B login
if python3 -c 'import wandb' 2>/dev/null; then
    if [ -f "${HOME}/.netrc" ] && grep -q 'machine api.wandb.ai' "${HOME}/.netrc"; then
        ok "wandb logged in"
    else
        warn "wandb installed but not logged in — run: wandb login"
    fi
else
    warn "wandb not installed — pip install wandb (optional)"
fi

echo
echo "── Next steps ─────────────────────────────────────────────────────"
echo "  1. Edit configs/eval_config.yaml — set runtime.task / eval_module / record_script"
echo "  2. Verify the config:    observer doctor"
echo "  3. Smoke test:           make dry DIR=runs/"
echo "  4. Real run:             make best DIR=runs/"
