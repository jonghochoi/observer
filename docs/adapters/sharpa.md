# 🤖 Adapter — sharpa-rl-lab

## Table of contents

- [TL;DR](#tldr)
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Config stanza](#config-stanza)
- [Running](#running)
- [Verifying the integration](#verifying-the-integration)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

- `rl_isaaclab.scripts.eval` — eval script (invoked with python -m)
- `rl_isaaclab/scripts/record.py` — record script
- Paste the config stanza into `eval_config.yaml` and run immediately

---

## Prerequisites

- sharpa-rl-lab installed: `pip install -e /path/to/sharpa-rl-lab`
- observer installed: `pip install -e /path/to/observer`
- Both packages must be in the **same Python environment**

---

## Overview

[`sharpa-rl-lab`](../../..) is a PPO / ProprioAdapt stack built on `DirectRLEnv` + `GymStyleEnvWrapper`.
The adapter code lives entirely inside the sharpa repo; observer has zero sharpa-specific code.

| Item | Location |
|:---|:---|
| Metrics script | `rl_isaaclab.scripts.eval` |
| Record script | `rl_isaaclab/scripts/record.py` |
| Env instrumentation | `sharpa_wave_env.py` |

### ── Env instrumentation (gym info keys)

`sharpa_wave_env.py` exposes the following keys in the gym `info` dict:

| Key | Description |
|:---|:---|
| `eval/success` | Episode success flag |
| `eval/slip_detected` | Tactile contact signal transition |
| `eval/fingertip_forces` | Fingertip forces (per-step) |
| `eval/joint_velocities` | Joint velocities (per-step) |
| `eval/joint_torques` | Joint torques (per-step) |
| `eval/pos_error` | Position error (m) |
| `eval/rot_error_deg` | Rotation error (deg) |
| `eval/init_{roll,pitch,yaw}_deg` | Initial object pose |

The eval script slices these keys per env to build `episodes.json`.

Observation normalization (`running_mean_std`), checkpoint loading (`restore_test`), and algorithm selection (`agent_cfg["algo"]`) follow sharpa's training-time conventions — no extra wiring needed.

---

## Config stanza

Paste under `runtime:` in `observer/configs/eval_config.yaml`:

```yaml
runtime:
  task: "Isaac-Inhand-Rotate-Sharpa-Wave-v0"
  eval_module: "rl_isaaclab.scripts.eval"
  record_script: "rl_isaaclab/scripts/record.py"
  num_envs: 4
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"

  # optional: forward sharpa-specific flags
  # extra_eval_args:
  #   - "--cache=/path/to/grasps.pkl"
```

---

## Running

```bash
cd /path/to/observer

# single checkpoint
python eval_runner.py --checkpoint /path/to/sharpa_run/model_5000.pth

# skip video (fast verification)
python eval_runner.py --checkpoint /path/to/model_5000.pth --skip_video
```

The metrics stage invokes `python -m rl_isaaclab.scripts.eval` from whichever Python environment has both sharpa-rl-lab and observer installed (`pip install -e .` in each).
The record stage runs sharpa's `record.py` under `isaaclab.sh`; sharpa in turn imports `observer.isaac.{camera_controller, recorder}` as utility libraries.

---

## Verifying the integration

```bash
# 1. headless eval only — fast turnaround
python eval_runner.py --checkpoint /path/to/model_5000.pth --skip_video

# 2. inspect outputs
cat eval_results/*/metrics.json  | python -m json.tool | grep -E "success_rate|num_episodes"
cat eval_results/*/episodes.json | python -m json.tool | python -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} episodes')"
```

`success_rate` should match what `rl_isaaclab/scripts/play.py` reports on the same checkpoint (± episode sampling noise).

---

## Troubleshooting

**`success_rate` differs from play.py result**

Likely causes:
- `running_mean_std` not restored → check the `restore_test` call
- gym `info` keys not populated → check that `sharpa_wave_env.py::_get_rewards` fills the keys listed above

**`episodes.json` is empty or has length 0**

→ The `eval/init_{roll,pitch,yaw}_deg` keys are missing from the info dict.
Check that the env records the initial object pose in the info dict on every episode reset.

**`ModuleNotFoundError: rl_isaaclab`**

→ sharpa-rl-lab is not installed in the current Python environment.
```bash
pip install -e /path/to/sharpa-rl-lab
python -c "import rl_isaaclab; print('ok')"
```

---

## Next steps

| Document | Content |
|:---|:---|
| [`../20_INTEGRATION_CONTRACT.md`](../20_INTEGRATION_CONTRACT.md) | Eval / record contract details |
| [`../21_ADAPTER_GUIDE.md`](../21_ADAPTER_GUIDE.md) | Guide for writing a new adapter |
| [`../30_METRICS_REFERENCE.md`](../30_METRICS_REFERENCE.md) | Collected metrics + failure taxonomy |
