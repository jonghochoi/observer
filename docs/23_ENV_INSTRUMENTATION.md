# 🔌 Env instrumentation — how info flows from env to observer

How a training environment must expose evaluation-relevant state, and how that state travels
from the env's step loop, through the eval subprocess, into `episodes.json`, and finally into
observer's failure-classification, coverage, and ranking stages.

This guide complements two neighbours:

- [`20_INTEGRATION_CONTRACT.md`](20_INTEGRATION_CONTRACT.md) — the *spec* for the eval/record
  subprocess CLI and `episodes.json` schema. Authoritative; this guide does not redefine it.
- [`21_ADAPTER_GUIDE.md`](21_ADAPTER_GUIDE.md) — the *how-to* for writing a new framework
  adapter. Focused on `eval.py` / `record.py`, not on the env-side data path.

What's been missing, and what this guide adds, is the **end-to-end picture**: a single page that
shows what data observer needs, where in the env it must originate, and how it reaches each
downstream consumer. Read this once before patching an unfamiliar env.

## Table of contents

- [Why instrumentation exists](#why-instrumentation-exists)
- [The data flow](#the-data-flow)
- [The contract — per-env tensors vs scalars](#the-contract--per-env-tensors-vs-scalars)
- [Required keys and who consumes them](#required-keys-and-who-consumes-them)
- [Worked example — minimal env patch](#worked-example--minimal-env-patch)
- [Common pitfalls](#common-pitfalls)
- [Probe script](#probe-script)
- [Cross-references](#cross-references)

---

## Why instrumentation exists

Observer is built around a **subprocess-only boundary**: the orchestrator never imports the
training repo's RL stack, torch, Isaac, or env code (see `CLAUDE.md` → "Subprocess-only
boundary"). The single channel observer has into the env's state is the gym `info` dict that
the user-supplied eval CLI sees on every step. The eval CLI snapshots that dict at episode
termination and writes one line into `episodes.json`. Everything observer does downstream
(failure classification, state-coverage heatmaps, ranking) consumes that one file.

This boundary is deliberate — it keeps observer reusable across frameworks (sharpa-rl-lab,
custom in-house stacks, third-party RL libs) without growing a fan-out of integrations. The
cost is that the env must surface evaluation-relevant state through the `info` channel, since
observer cannot reach in and read it.

The work of teaching the env to do this is what we call **instrumentation**. It is a one-time,
env-side patch — not a per-checkpoint runtime concern.

---

## The data flow

Single-direction fan-out, env on the left, central tracker on the right:

```
  ┌──────────────────────────┐
  │  env (DirectRLEnv etc.)  │
  │                          │
  │  _reset_idx()  ────────┐ │
  │      ↳ capture init    │ │
  │        pose, slip=0    │ │
  │                        │ │
  │  step() loop:          │ │
  │    _get_dones()  ──────┼─┼──▶  self.extras["eval/<key>"] = <per-env tensor>
  │      ↳ success,        │ │
  │        pos_err,        │ │
  │        rot_err          │
  │    compute_obs():      │
  │      ↳ slip transitions│
  └─────────────┬────────────┘
                │  IsaacLab forwards self.extras → gymnasium info
                ▼
  ┌──────────────────────────┐
  │  scripts/eval_cli.py     │
  │  (training repo)         │
  │                          │
  │  while ep < N:           │
  │    obs, *, dones, info   │
  │      = env.step(act)     │
  │    for d in done_idx:    │
  │      ep_records.append(  │
  │        { ...info[d]...}) │
  └─────────────┬────────────┘
                │  json.dump
                ▼
  ┌──────────────────────────┐         observer/pipeline/
  │  episodes.json           │          ├─ failure_classifier.py
  │  [{success, length,      │  ──▶     ├─ state_coverage.py
  │    pos_error, ...}, ...] │          ├─ auto_select.py
  └──────────────────────────┘          └─ report_generator.py
                                                  │
                                                  ▼
                                        ObserverResults dataclass
                                                  │
                                                  ▼
                                  nexus EvalLogger.upload(...)  ──▶  central MLflow
```

Three observations to internalize:

1. **The env owns instrumentation.** The eval CLI is just a transport layer — it copies fields
   out of `info[idx]` into a JSON record. If the env doesn't emit the keys, the eval CLI cannot
   invent them.
2. **observer is downstream-only.** It reads `episodes.json`. It does not call back into the
   env or the eval CLI. Anything observer needs must already be in that JSON.
3. **The boundary survives across runs.** A correctly instrumented env produces evaluable
   `episodes.json` files for every checkpoint, current and future, without further env edits.

---

## The contract — per-env tensors vs scalars

IsaacLab's `DirectRLEnv.step()` returns `(obs, reward, terminated, truncated, info)`. The `info`
dict is the same object as `self.extras` populated during the step. Two value shapes coexist
in `extras` and they mean *different* things:

| Shape | Used for | Example | Will eval pick this up correctly? |
|:---|:---|:---|:---:|
| **Scalar** (e.g. `tensor.mean()`) | Training-time logging — every env reports the same number to TensorBoard / MLflow. | `self.extras["alignment"] = alignment.mean()` | ❌ Per-episode value will be the cross-env mean, not the actual env's value |
| **Per-env tensor** (shape `(num_envs,)` or `(num_envs, ...)`) | Evaluation contract — each env's value is recoverable by indexing with the terminated env's index. | `self.extras["eval/success"] = success_flags` | ✅ |

This split is the most common source of "instrumentation looks done but `episodes.json` is
garbage" bugs. The fix is simple: any key under the `eval/` prefix is **always** a per-env
tensor; anything else can be either. Don't `.mean()` an `eval/*` value.

The slash in the key name (`eval/success`, not `eval_success`) survives gymnasium's wrappers —
verified in the sharpa adapter pattern. Use the slash form for consistency with the contract.

---

## Required keys and who consumes them

The schema is defined in [`20_INTEGRATION_CONTRACT.md §3`](20_INTEGRATION_CONTRACT.md). This
table adds the *consumer* column so you can decide which keys to prioritize when patching an
env incrementally:

| Key | Type | Required? | Consumed by |
|:---|:---|:---|:---|
| `eval/success` | bool/float | yes | `metrics_collector` (`success_rate`), `auto_select` (ranking) |
| `eval/length` | int/float | yes | `metrics_collector` (`episode_length_mean`), `auto_select` |
| `eval/pos_error` | float | yes | `metrics_collector` (`object_pose_error_mm`), `failure_classifier` (drop / pose-error rules), `auto_select` |
| `eval/rot_error_deg` | float | yes | `failure_classifier` (orientation-error rule) |
| `eval/init_roll_deg` | float | yes | `state_coverage` (roll axis of the heatmap) |
| `eval/init_pitch_deg` | float | yes | `state_coverage` (pitch axis of the heatmap) |
| `eval/init_yaw_deg` | float | yes | `state_coverage` (optional third axis on some presets) |
| `eval/init_pos_x` / `_y` / `_z` | float | yes | `state_coverage` position-domain presets |
| `eval/slip_count` | int/float | optional | `metrics_collector` (`slip_events_per_episode`), `failure_classifier` (`slip_failure` rule) |
| `eval/contact_forces` (list) | optional | `metrics_collector` (`contact_force_rms`) |
| `eval/joint_velocities` (list) | optional | `metrics_collector` (`joint_velocity_rms`) |

If you're going to ship instrumentation in stages, the practical priority order is:
**success → length → init_{roll,pitch}_deg → pos_error → rot_error_deg → init_{yaw,pos_*} →
slip_count**. The first four unlock 80% of observer's value (the success-rate scoreboard plus
the coverage heatmap).

---

## Worked example — minimal env patch

Concrete patch for an `IsaacLab DirectRLEnv` subclass with a goal-pose task. Adapt the field
names and success criterion to your env, but the *shape* of the patch is portable.

### ── 1. Per-env buffers in `__init__`

Allocated once; reset per episode. Store anything that needs to persist across the
`_reset_idx()` → `step()` boundary:

```python
# eval-time per-env state (observer episodes.json)
self._eval_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
self._eval_init_rpy_deg = torch.zeros((self.num_envs, 3), device=self.device)
self._eval_slip_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
self._eval_prev_contact = torch.zeros(
    (self.num_envs, self.num_fingertips), dtype=torch.bool, device=self.device
)
```

### ── 2. Capture init pose in `_reset_idx`

After the env writes the initial object pose into the simulator:

```python
from isaaclab.utils.math import euler_xyz_from_quat
import math

# After self.object.write_root_pose_to_sim(...) for env_ids:
self._eval_init_pos[env_ids] = (
    object_default_state[:, :3] - self.scene.env_origins[env_ids]
)
roll, pitch, yaw = euler_xyz_from_quat(object_default_state[:, 3:7])  # rad
self._eval_init_rpy_deg[env_ids] = torch.stack(
    [roll, pitch, yaw], dim=-1) * (180.0 / math.pi)
self._eval_slip_count[env_ids] = 0
self._eval_prev_contact[env_ids] = False
```

### ── 3. Track slip transitions during `step` (optional but recommended)

In whatever method already inspects contact signals (`compute_observations`, `_get_dones`, etc.):

```python
contact_active = sensed_contacts > self.cfg.contact_threshold
slip_now = self._eval_prev_contact & ~contact_active   # contact → no-contact
self._eval_slip_count += slip_now.any(dim=-1).long()
self._eval_prev_contact = contact_active
```

### ── 4. Emit per-env eval keys in `_get_dones`

The most important step. Run *every* step, not only on `done` — observer's eval CLI snapshots
`info` at termination, so the value at termination index is what lands in `episodes.json`:

```python
# Define success per your task. Example: didn't drop AND axis-aligned within tolerance.
align_cos = (some_axis_world * desired_axis).sum(-1).clamp(-1, 1)
align_ok = align_cos > math.cos(math.radians(30.0))
success = (~dropped) & align_ok

pos_err = torch.norm(self.object_pos - self.object_goal_pos, dim=-1)
rot_err_deg = torch.acos(align_cos) * (180.0 / math.pi)

self.extras["eval/success"]        = success.float()
self.extras["eval/length"]         = self.episode_length_buf.float()
self.extras["eval/pos_error"]      = pos_err
self.extras["eval/rot_error_deg"]  = rot_err_deg
self.extras["eval/init_roll_deg"]  = self._eval_init_rpy_deg[:, 0]
self.extras["eval/init_pitch_deg"] = self._eval_init_rpy_deg[:, 1]
self.extras["eval/init_yaw_deg"]   = self._eval_init_rpy_deg[:, 2]
self.extras["eval/init_pos_x"]     = self._eval_init_pos[:, 0]
self.extras["eval/init_pos_y"]     = self._eval_init_pos[:, 1]
self.extras["eval/init_pos_z"]     = self._eval_init_pos[:, 2]
self.extras["eval/slip_count"]     = self._eval_slip_count.float()
```

### ── 5. What the eval CLI sees

After the patch, one termination-step `info` dict, with `idx` being a terminated env index,
looks like:

```python
info["eval/success"][idx]       # tensor scalar, e.g. 1.0
info["eval/length"][idx]        # 187.0
info["eval/pos_error"][idx]     # 0.0042
info["eval/rot_error_deg"][idx] # 3.1
info["eval/init_roll_deg"][idx] # 12.0
# … etc
```

The eval CLI casts each to a Python `float`/`bool`/`int` and writes one record per terminated
env into `episodes.json`:

```json
{
  "checkpoint": "best.pth",
  "success": true, "length": 187,
  "final_pos_error_m": 0.0042, "final_rot_error_deg": 3.1,
  "init_roll_deg": 12.0, "init_pitch_deg": -4.3, "init_yaw_deg": 0.0,
  "init_pos_x": 0.0, "init_pos_y": 0.0, "init_pos_z": 0.05,
  "slip_count": 1,
  "failure_mode": "unknown"
}
```

`failure_mode` starts as `"unknown"`; observer's `failure_classifier` (step 2 of the pipeline)
fills it in based on the other fields. You don't compute it env-side.

---

## Common pitfalls

**`.mean()` on an `eval/*` value.** All envs report the same number; `episodes.json` ends up
with constant fields. Drop the `.mean()` — observer wants per-env tensors here. The rest of
`extras` (training-time logging keys like `alignment`, `total_reward`) can stay scalar.

**Eval-time randomization left on.** If `randomize_friction`, `contact_latency`,
`contact_sensor_noise`, or `gravity_curriculum` are active during eval, `success_rate` becomes
a measurement of the *eval*'s noise, not the policy. Force them off in `eval_cli.py` before
constructing the env. The user's own `play.py` already does this — mirror that switch.

**`episode_length_buf` read at the wrong moment.** IsaacLab resets `episode_length_buf` to zero
on `_reset_idx`. If you read it *after* the auto-reset, you get `0`. Read it inside
`_get_dones()` before any reset — that's where the per-env `eval/length` value should be set.

**Slash vs underscore key style.** observer's contract uses slashes: `eval/success`, not
`eval_success`. Mixing styles makes the eval CLI silently miss keys and yields empty
`episodes.json` records.

**Per-env mismatched shape.** `eval/length` shape `(num_envs,)` is correct;
`(num_envs, 1)` will pass through but indexing in eval CLI will produce shape-`(1,)` tensors
that aren't trivially castable. Squeeze before assigning to `extras`.

**Defining "success" too loosely.** The `failure_classifier` rules-chain assumes
`success=True` means "the task was actually accomplished." If you set `success = ~dropped`
without checking the goal, you'll see `success_rate ≈ 1.0` while half the rollouts didn't reach
the goal. Cross-check `success` against `pos_error` / `rot_error_deg` thresholds at minimum.

---

## Probe script

Drop this into the training repo at `scripts/instrumentation_probe.py` and run it under the
IsaacLab interpreter to verify which keys the env emits:

```python
"""scripts/instrumentation_probe.py — does the env emit the eval/* info keys?
Run via:  ./isaaclab.sh -p scripts/instrumentation_probe.py
"""
import argparse, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args, hydra_args = p.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
sim = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

# Replace these two imports with whatever your training repo uses.
import your_training_repo.envs  # noqa: F401
from your_training_repo.utils.env_wrapper import YourEnvWrapper

@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = YourEnvWrapper(env, clip_actions=1.0)
    obs, _ = env.reset()
    actions = torch.zeros(
        (args.num_envs, env.action_space.shape[-1]),
        device=env.unwrapped.device,
    )
    _, _, _, _, info = env.step(actions)
    required = ["eval/success", "eval/length", "eval/pos_error", "eval/rot_error_deg",
                "eval/init_roll_deg", "eval/init_pitch_deg", "eval/init_yaw_deg",
                "eval/init_pos_x", "eval/init_pos_y", "eval/init_pos_z"]
    sample = info if isinstance(info, dict) else info[0]
    missing = [k for k in required if k not in sample]
    print("present:", sorted(set(required) - set(missing)))
    print("missing:", missing or "none")
    print("all eval/ keys:", sorted(k for k in sample if k.startswith("eval/")))
    env.close()

if __name__ == "__main__":
    main()
    sim.close()
```

> ⚠️ Plain `python scripts/instrumentation_probe.py` will fail with
> `ModuleNotFoundError: No module named 'pxr'` — the USD lib only loads inside the IsaacSim
> interpreter. Use `./isaaclab.sh -p ...` (or your framework's equivalent wrapper).

---

## Cross-references

| Document | Why you'd read it after this |
|:---|:---|
| [`20_INTEGRATION_CONTRACT.md`](20_INTEGRATION_CONTRACT.md) | Authoritative `metrics.json` / `episodes.json` schemas and subprocess CLI shape |
| [`21_ADAPTER_GUIDE.md`](21_ADAPTER_GUIDE.md) | How to package your eval CLI into a framework adapter |
| [`22_EXTERNAL_LOGGER_HANDOFF.md`](22_EXTERNAL_LOGGER_HANDOFF.md) | How a downstream tracker (nexus, W&B, …) consumes observer's outputs |
| [`30_METRICS_REFERENCE.md`](30_METRICS_REFERENCE.md) | The 8-metric / 6-failure-mode table that `failure_classifier` and `auto_select` are built around |
| [`adapters/sharpa.md`](adapters/sharpa.md) | Reference adapter — sharpa-rl-lab's eval CLI showing the `info` keys in production |
