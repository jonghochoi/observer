# 🤖 Adapter — hand_expert (sharpa-derived custom PPO + nexus EvalLogger)

End-to-end worked example for a training repo built on the `hand_expert` package
(custom `PPO` / `ProprioAdapt` algorithms, `SharpaEnvWrapper`, IsaacLab `DirectRLEnv`)
that already streams metrics to **nexus** via `make_logger(mode="dual", ...)` and now wants to
close the loop: run observer's evaluation pipeline at the end of training and forward the
results onto the same central MLflow run as `artifacts/eval/<eval_id>/…`.

This adapter differs from [`sharpa.md`](sharpa.md) in two ways:

1. The training package is `hand_expert`, not upstream `rl_isaaclab` — the eval entrypoint must
   load `hand_expert.algo.PPO` checkpoints, so a small local `scripts/eval_cli.py` is needed.
2. The downstream logger is **nexus** — the integration uses `EvalLogger.from_run_info(...)`
   from [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)
   to attach observer's outputs to the central run.

## Table of contents

- [TL;DR](#tldr)
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Env instrumentation](#env-instrumentation)
- [Files to add or modify in the training repo](#files-to-add-or-modify-in-the-training-repo)
- [Running](#running)
- [Verifying the integration](#verifying-the-integration)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

- **Modify** `train.py` — add `central_tracking_uri=` to `make_logger(...)` and spawn a
  post-training eval subprocess.
- **Add** `scripts/eval_cli.py` — observer-contract metrics emitter, derived from
  `play.py:_play_custom_algo` but trimmed to a finite-episode loop that writes
  `metrics.json` + `episodes.json`.
- **Add** `configs/eval_config.yaml` — observer pipeline config; start with
  `skip_video: true`.
- **Add** `scripts/run_eval_and_upload.py` — verbatim from
  [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md);
  the only nexus↔observer glue and the same script doubles as a manual re-evaluation CLI.

---

## Prerequisites

- `hand_expert` training repo with a working `train.py` (this adapter assumes the structure of
  the user-supplied script — `hand_expert.algo.PPO` / `ProprioAdapt`, `SharpaEnvWrapper`,
  `make_logger(mode="dual", tb_dir=train_dir, ...)`).
- nexus installed in the same Python environment: `pip install nexus-logger` (or
  `pip install -e /path/to/nexus`).
- observer installed in the same Python environment: `pip install -e /path/to/observer`.
- The trainer's `.nexus_run.json` sidecar is being written into `train_dir` (default behavior of
  `make_logger(tb_dir=...)`). The training repo's `scheduled_sync` is already replicating runs
  to central MLflow — that flow does **not** change.

---

## Overview

`hand_expert` is a sharpa-derived stack: `Sharpa-InHandRotation-Direct-v0` on IsaacLab
`DirectRLEnv`, wrapped with `SharpaEnvWrapper`, driven by a custom `hand_expert.algo.ppo.PPO` /
`hand_expert.algo.proprio_adapt.ProprioAdapt`. Checkpoints are saved as `best.pth` / `last.pth`
in the algo subdir (probed by `play.py:get_ppo_checkpoint_path`).

| Item | Location |
|:---|:---|
| Metrics script | `scripts/eval_cli.py` (this adapter) |
| Record script | _deferred — start with `skip_video: true`_ |
| Env instrumentation | `hand_expert/envs/.../sharpa_wave_env.py` (or equivalent) |
| Logger integration | nexus `EvalLogger.from_run_info(target="central")` |

The two-process boundary is preserved end-to-end: `train.py` finishes, closes
`simulation_app`, then **spawns a subprocess** (`scripts/run_eval_and_upload.py`) that
re-launches Isaac under observer's metrics/record stages. This avoids GPU memory
contention between the still-loaded trainer and observer's own Isaac launch
([`docs/22_EXTERNAL_LOGGER_HANDOFF.md:#recommended-consumption-pattern`](../22_EXTERNAL_LOGGER_HANDOFF.md#recommended-consumption-pattern)).

```
train.py                                                             central MLflow
   │                                                                       ▲
   │ make_logger(central_tracking_uri=...)                                 │
   │     ↳ writes train_dir/.nexus_run.json with central URI               │
   │                                                                       │
   │ agent.train()                                                         │
   │ logger.close(); env.close(); simulation_app.close()                   │
   │                                                                       │
   ├──▶ subprocess: scripts/run_eval_and_upload.py                         │
   │       ├─ EvalConfig.from_yaml(configs/eval_config.yaml)              │
   │       ├─ PipelineOrchestrator.run_single(checkpoint)                  │
   │       │     ↳ subprocess: python -m scripts.eval_cli ...              │
   │       │           writes metrics.json + episodes.json                 │
   │       ├─ result_locator.locate_results(...) → ObserverResults         │
   │       ├─ result_locator.read_metrics(...)   → flat dict               │
   │       └─ EvalLogger.from_run_info(train_dir).upload(...) ─────────────┘
                                                  artifacts/eval/<eval_id>/…
```

---

## Env instrumentation

observer builds `episodes.json` by slicing per-step `info` dicts at episode termination. The env
must populate the following keys on every `done` step (taxonomy mirrors
[`adapters/sharpa.md:#-env-instrumentation-gym-info-keys`](sharpa.md#-env-instrumentation-gym-info-keys)):

| Key | Type | Description |
|:---|:---|:---|
| `eval/success` | bool | Episode success flag |
| `eval/length` | int | Step count at termination |
| `eval/pos_error` | float | Final object position error (m) |
| `eval/rot_error_deg` | float | Final object rotation error (deg) |
| `eval/init_roll_deg` | float | Initial object roll (deg) |
| `eval/init_pitch_deg` | float | Initial object pitch (deg) |
| `eval/init_yaw_deg` | float | Initial object yaw (deg) |
| `eval/init_pos_x` | float | Initial object position x (m) |
| `eval/init_pos_y` | float | Initial object position y (m) |
| `eval/init_pos_z` | float | Initial object position z (m) |
| `eval/slip_count` | int | (optional) Tactile slip events |

If the existing `Sharpa-InHandRotation-Direct-v0` env populates these in `_get_observations` /
`_get_rewards` / `_get_extras`, no env edits are needed. If not, the env must be patched to emit
them — observer cannot synthesize them after the fact, and missing keys cause `episodes.json` to
be empty (which silently disables observer's failure-classification and coverage stages).

> 📖 If you're new to *why* this `info`-dict channel exists and *how* the data flows from env →
> eval CLI → `episodes.json` → observer's failure-classification / coverage stages, read
> [`../23_ENV_INSTRUMENTATION.md`](../23_ENV_INSTRUMENTATION.md) first. It walks through the
> full path with a minimal worked patch.

Before wiring anything else, confirm what the env already emits. Use whichever check is faster —
the dynamic one needs an Isaac boot:

### ── Static check — grep the env source

No Isaac boot. Looks for the assignments observer expects to find:

```bash
grep -nE 'self\.extras\["eval/' \
    hand_expert/envs/in_hand_rotation/direct/v0/sharpa_wave_env.py
```

Ten lines (one per `eval/<key>`) → instrumentation is done. Zero lines → the env needs a patch.

### ── Dynamic check — `scripts/instrumentation_probe.py`

Mirrors `train.py` / `play.py`'s `AppLauncher` boot, runs one step, and reports which keys
arrive in the gym `info` dict. Save in the training repo at `scripts/instrumentation_probe.py`:

```python
"""scripts/instrumentation_probe.py — does the env emit the eval/* info keys?
Run via:  ./isaaclab.sh -p scripts/instrumentation_probe.py
"""
import argparse, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Sharpa-InHandRotation-Direct-v0")
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args, hydra_args = p.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
sim = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
import hand_expert.envs  # noqa: F401
from hand_expert.utils.env_wrapper import SharpaEnvWrapper

@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = SharpaEnvWrapper(env, clip_actions=1.0)
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

Run via `./isaaclab.sh -p scripts/instrumentation_probe.py` — plain `python` fails with
`ModuleNotFoundError: No module named 'pxr'` because IsaacLab `DirectRLEnv` only loads inside
the IsaacSim interpreter that ships the USD library.

---

## Files to add or modify in the training repo

### ── 1. `train.py` — pass `central_tracking_uri` and spawn the eval subprocess

The user-supplied `train.py` calls `make_logger(...)` without `central_tracking_uri`, which makes
`EvalLogger.from_run_info(target="central")` fail with the documented migration message
(see [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#migrating-trainers-without-central_tracking_uri`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#migrating-trainers-without-central_tracking_uri)).
Three small edits:

#### ▸ a. Add three CLI flags

```python
parser.add_argument(
    "--central_tracking_uri",
    type=str,
    default="http://nexus-server:5000",
    help="Central MLflow URI written into .nexus_run.json so the post-training eval can "
         "resolve the same run on central MLflow. Pass an empty string to disable.",
)
parser.add_argument(
    "--no_auto_eval", action="store_true",
    help="Skip the automatic post-training observer evaluation + upload.",
)
parser.add_argument(
    "--observer_config", type=str, default="configs/eval_config.yaml",
    help="Path to observer's EvalConfig YAML used by the post-training eval subprocess.",
)
```

#### ▸ b. Pass `central_tracking_uri=` to `make_logger`

```python
logger = make_logger(
    mode="dual",
    tb_dir=train_dir,
    run_name=run_name,
    tracking_uri=args_cli.tracking_uri,
    central_tracking_uri=args_cli.central_tracking_uri or None,   # ← NEW
    experiment_name=experiment_name,
    agent_params=agent_cfg,
    env_params=env_cfg,
    tags={
        "task": args_cli.task,
        "train": args_cli.algo,
        "hand": "sharpa_wave_22dof",
        "hand_side": args_cli.hand_side,
    },
)
```

#### ▸ c. Return a hand-off dict from `main()` and spawn the eval subprocess after `simulation_app.close()`

The eval pipeline must run in a fresh Python interpreter — observer relaunches IsaacLab via
`isaaclab.sh -p record.py` for the video stage, and co-locating that with the still-loaded
trainer crashes Isaac. Have `main()` return a small dict, then spawn the bridge after
`simulation_app.close()`:

```python
def main(env_cfg, agent_cfg):
    # … existing body up through the agent.train() / finally block …
    try:
        agent.train()
    finally:
        logger.close()
        env.close()

    checkpoint_for_eval = os.path.join(train_dir, "best.pth")
    if not os.path.exists(checkpoint_for_eval):
        checkpoint_for_eval = os.path.join(train_dir, "last.pth")

    return {
        "checkpoint": checkpoint_for_eval,
        "training_output_dir": train_dir,
        "observer_config": args_cli.observer_config,
        "no_auto_eval": args_cli.no_auto_eval,
    }


if __name__ == "__main__":
    eval_handoff = main()
    simulation_app.close()
    if eval_handoff and not eval_handoff["no_auto_eval"] and \
       os.path.exists(eval_handoff["checkpoint"]):
        import subprocess
        subprocess.run(
            [
                sys.executable, "scripts/run_eval_and_upload.py",
                "--checkpoint", eval_handoff["checkpoint"],
                "--training-output-dir", eval_handoff["training_output_dir"],
                "--observer-config", eval_handoff["observer_config"],
            ],
            check=False,
        )
```

> ⚠️ Do **not** invoke observer inline before `simulation_app.close()`. The trainer's Isaac
> instance must release the GPU before observer launches its own under `isaaclab.sh`.

### ── 2. `scripts/eval_cli.py` — observer-contract metrics emitter

This is the only non-trivial new file. It is the module observer calls as
`python -m scripts.eval_cli ...`. It restores a `hand_expert.algo.PPO` / `ProprioAdapt`
checkpoint, runs N episodes, and writes the two JSONs observer expects.

```python
"""scripts/eval_cli.py
=====================
Observer-contract eval entrypoint for hand_expert PPO/ProprioAdapt checkpoints.

Reads a checkpoint at --load_path, runs --num_episodes evaluation episodes, and writes
metrics.json + episodes.json at the paths given by --metrics_output / --episodes_output.

Invoked by observer as:
    python -m scripts.eval_cli \
        --task=<task> --load_path=<ckpt> --device=<dev> --seed=<seed> \
        --num_envs=<n> --num_episodes=<n> --headless \
        --metrics_output=<path> --episodes_output=<path> \
        --algo=PPO            # forwarded via runtime.extra_eval_args

The schema for both JSON files is documented in
observer/docs/20_INTEGRATION_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# ── 1. Argument parsing ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Observer-contract eval entrypoint for hand_expert.")
parser.add_argument("--task", required=True)
parser.add_argument("--load_path", required=True)
parser.add_argument("--metrics_output", required=True)
parser.add_argument("--episodes_output", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "ProprioAdapt"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--gravity", type=float, default=-9.81)
parser.add_argument("--cache", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force headless for the metrics stage — observer always passes this flag.
args.headless = True

simulation_app = AppLauncher(args).app

# ── 2. Late imports (must follow AppLauncher.app) ─────────────────────
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import hand_expert.envs  # noqa: F401, E402
from hand_expert.algo.ppo.ppo import PPO  # noqa: E402
from hand_expert.algo.proprio_adapt.proprio_adapt import ProprioAdapt  # noqa: E402
from hand_expert.utils.env_wrapper import SharpaEnvWrapper  # noqa: E402


# ── 3. Episode collection ─────────────────────────────────────────────
def _info_at(infos, idx):
    """Extract the per-env info entry; gymnasium VectorEnv flips list/dict layout."""
    if isinstance(infos, dict):
        return {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in infos.items()}
    return infos[idx]


def _episode_record(info: dict, ckpt: str) -> dict:
    """Map env info keys → contract schema. Missing keys raise KeyError loudly."""
    return {
        "checkpoint": Path(ckpt).name,
        "success": bool(info["eval/success"]),
        "length": int(info["eval/length"]),
        "final_pos_error_m": float(info["eval/pos_error"]),
        "final_rot_error_deg": float(info["eval/rot_error_deg"]),
        "init_roll_deg": float(info["eval/init_roll_deg"]),
        "init_pitch_deg": float(info["eval/init_pitch_deg"]),
        "init_yaw_deg": float(info["eval/init_yaw_deg"]),
        "init_pos_x": float(info["eval/init_pos_x"]),
        "init_pos_y": float(info["eval/init_pos_y"]),
        "init_pos_z": float(info["eval/init_pos_z"]),
        "slip_count": int(info.get("eval/slip_count", 0)),
        "failure_mode": "unknown",   # observer's failure_classifier fills this in step 2
    }


# ── 4. Main ───────────────────────────────────────────────────────────
@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    # Match play.py's _play_custom_algo overrides — keep eval-time variance off.
    if args.algo == "ProprioAdapt":
        agent_cfg["network"]["proprio_adapt"] = True
    torch.manual_seed(args.seed)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, args.gravity)
    env_cfg.gravity_curriculum = False
    if args.cache:
        env_cfg.grasp_cache_path = args.cache

    agent_cfg["seed"] = args.seed
    agent_cfg["device"] = args.device
    agent_cfg["algorithm"]["num_actors"] = args.num_envs
    agent_cfg["test"] = True

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    clip_actions = agent_cfg["algorithm"].get("clip_actions", 1.0)
    env = SharpaEnvWrapper(env, clip_actions=clip_actions)

    if args.algo == "ProprioAdapt":
        agent = ProprioAdapt(env, output_dir="", agent_cfg=agent_cfg, create_output_dir=False)
    else:
        agent = PPO(env, output_dir="", agent_cfg=agent_cfg, create_output_dir=False)
    agent.restore_test(args.load_path)

    ep_records: list[dict] = []
    obs, _ = env.reset()
    while len(ep_records) < args.num_episodes:
        with torch.inference_mode():
            actions = agent.act_inference(obs)   # see Troubleshooting if name differs
        obs, _, dones, _, infos = env.step(actions)
        done_idx = dones.nonzero().flatten().tolist() if hasattr(dones, "nonzero") else \
            [i for i, d in enumerate(dones) if d]
        for idx in done_idx:
            if len(ep_records) >= args.num_episodes:
                break
            ep_records.append(_episode_record(_info_at(infos, idx), args.load_path))

    success_rate = sum(int(e["success"]) for e in ep_records) / max(1, len(ep_records))
    metrics = {
        "checkpoint": Path(args.load_path).name,
        "num_episodes": len(ep_records),
        "success_rate": success_rate,
        "episode_length_mean": sum(e["length"] for e in ep_records) / max(1, len(ep_records)),
        "object_pose_error_mm": 1000.0 * (
            sum(e["final_pos_error_m"] for e in ep_records) / max(1, len(ep_records))
        ),
    }

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.episodes_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.episodes_output, "w") as f:
        json.dump(ep_records, f, indent=2)
    print(f"[eval_cli] wrote {len(ep_records)} episodes (sr={success_rate:.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

> 💡 The exact inference call (`agent.act_inference(obs)`) depends on the `hand_expert.algo.PPO`
> public API. If the class exposes `agent.policy(obs)` or expects a different shape, follow what
> `play.py:_play_custom_algo` does after `agent.restore_test(...)` — that's the canonical
> reference call.

### ── 3. `configs/eval_config.yaml` — observer pipeline config

Adapted from [`adapters/sharpa.md:#config-stanza`](sharpa.md#config-stanza). Keep
`skip_video: true` until a separate `record_script` is added.

```yaml
runtime:
  task: "Sharpa-InHandRotation-Direct-v0"
  eval_module: "scripts.eval_cli"
  record_script: "scripts/record.py"   # placeholder; only used when skip_video=False
  num_envs: 8
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
  extra_eval_args:
    - "--algo=PPO"          # forwarded verbatim to scripts/eval_cli.py

metrics:
  num_eval_episodes: 50

video:
  resolution: [1280, 720]
  fps: 30
  concat_views: true

cameras:
  - {name: "front", eye: [0.6, 0.0, 0.4], target: [0.0, 0.0, 0.05], record_steps: 600}
  - {name: "side",  eye: [0.0, 0.6, 0.4], target: [0.0, 0.0, 0.05], record_steps: 600}

skip_video: true
skip_report: false
dry_run: false
```

### ── 4. `scripts/run_eval_and_upload.py` — nexus↔observer bridge

Copy verbatim from
[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#worked-example-a-training-repo-glue-script`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#worked-example-a-training-repo-glue-script).
The template below is reproduced for convenience — diff against the canonical version before
copying, in case the upstream evolves.

```python
"""scripts/run_eval_and_upload.py
================================
Lives in the training repo. Imports observer + nexus and glues them together; observer and
nexus do not depend on each other.

Invoked automatically by train.py at the end of training, or manually for re-evaluation:

    python scripts/run_eval_and_upload.py \
        --checkpoint logs/<exp>/<run_name>/PPO/best.pth \
        --training-output-dir logs/<exp>/<run_name>/PPO \
        --observer-config configs/eval_config.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from nexus.logger.eval_logger import EvalLogger

from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics


def _eval_commit() -> str:
    env = os.environ.get("OBSERVER_COMMIT")
    if env:
        return env
    try:
        import observer
        repo = Path(observer.__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--training-output-dir", required=True, type=Path,
                   help="Dir holding .nexus_run.json from make_logger().")
    p.add_argument("--observer-config", required=True, type=Path)
    p.add_argument("--eval-output-dir", type=Path, default=None,
                   help="Defaults to <training-output-dir>/eval.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    eval_root = args.eval_output_dir or (args.training_output_dir / "eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig.from_yaml(str(args.observer_config))
    orch = PipelineOrchestrator(config=cfg, output_root=eval_root)
    result = orch.run_single(args.checkpoint)
    if not result.success:
        print(f"[eval] downstream run_single failed: {result.error_msg}", file=sys.stderr)
        return 2

    obs = locate_results(eval_root, result_dir=result.output_dir)
    metrics = read_metrics(obs)
    if not metrics:
        print("[eval] no metrics produced — aborting upload", file=sys.stderr)
        return 3

    tags = {
        "observer_commit": _eval_commit(),
        "checkpoint": args.checkpoint.name,
        "algo": os.environ.get("HAND_EXPERT_ALGO", "PPO"),
    }

    ev = EvalLogger.from_run_info(args.training_output_dir, target="central")
    eval_id = ev.upload(
        eval_dir=result.output_dir,
        metrics=metrics,
        tags=tags,
        generate_index=True,
        dry_run=args.dry_run,
    )
    print(f"[eval] uploaded eval_id={eval_id} ({len(metrics)} metrics) "
          f"under run_name={ev.run_name} on {ev.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Running

End of training auto-flow (default — `--no_auto_eval` to disable):

```bash
python train.py \
    --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name myexp \
    --run_name ppo_v17_seed3 \
    --central_tracking_uri http://nexus-server:5000
```

Manual re-evaluation against any past run (the `.nexus_run.json` sidecar persists):

```bash
python scripts/run_eval_and_upload.py \
    --checkpoint logs/myexp/ppo_v17_seed3/PPO/best.pth \
    --training-output-dir logs/myexp/ppo_v17_seed3/PPO \
    --observer-config configs/eval_config.yaml
```

`EvalLogger.from_run_info(target="central")` resolves the central run by `run_name` (not by
`run_id`) — see
[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#why-run_name-not-run_id`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#-why-run_name-not-run_id) — so it
keeps working long after the local-MLflow UUID rotates.

---

## Verifying the integration

Five-step smoke (run in order):

```bash
# 1. Sidecar carries central_tracking_uri.
python train.py --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4 --no_auto_eval
cat logs/nexus_eval_smoke/<run_name>/PPO/.nexus_run.json | python -m json.tool \
    | grep central_tracking_uri
#   → must show the central URL, not null.

# 2. observer dry-run round-trip without booting Isaac.
#    Set dry_run: true in configs/eval_config.yaml first.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml \
    --dry-run

# 3. Real metrics round-trip with skip_video: true (no record_script needed).
#    Flip dry_run: false; keep skip_video: true.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml
#   → check central MLflow UI for artifacts/eval/<eval_id>/metrics.json,
#     metrics tab for eval/success_rate, and the eval.last_id tag.

# 4. Auto-trigger from train.py (drop --no_auto_eval).
python train.py --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4

# 5. (later) Add scripts/record.py and flip skip_video: false.
```

`success_rate` from observer should match what `play.py` reports on the same checkpoint
(± episode sampling noise) — same caveat as
[`adapters/sharpa.md:#verifying-the-integration`](sharpa.md#verifying-the-integration).

---

## Troubleshooting

**`EvalLogger.from_run_info(...) raised ValueError: ... has no central_tracking_uri`**

→ `train.py` was run without `--central_tracking_uri` (or the kwarg was not threaded into
`make_logger`). Fix: re-run training with the flag, or pass `tracking_uri="<central-url>"`
to `from_run_info()` to override.

**`success_rate` differs from `play.py` result**

Likely causes (mirroring sharpa.md troubleshooting):

- `running_mean_std` not restored — confirm `agent.restore_test(...)` matches what
  `play.py:_play_custom_algo` does.
- Missing env `info` keys — re-run the env-instrumentation probe in
  [Env instrumentation](#env-instrumentation).

**`episodes.json` is empty or has length 0**

→ Either the env does not populate the `eval/init_*` keys, or `dones` was never `True` (episode
limit too short). Confirm via:

```bash
python -c "import json; d=json.load(open('logs/.../eval/<dir>/episodes.json')); print(len(d))"
```

**`AttributeError: 'PPO' object has no attribute 'act_inference'`**

→ Replace the inference call in `scripts/eval_cli.py` with whatever `play.py:_play_custom_algo`
uses on its restored agent. The user's `play.py` uses `agent.test()` for an infinite loop;
`eval_cli.py` instead needs a **single-step inference call** so the harness can count episodes.

**`ModuleNotFoundError: scripts.eval_cli`**

→ `python -m scripts.eval_cli` requires `scripts/` to be on `sys.path` and contain an
`__init__.py`. Add an empty `scripts/__init__.py`, or run from the training repo root with
`PYTHONPATH=.`.

**Run not found on central MLflow**

→ The training run has not yet been replicated by `scheduled_sync`. Either wait for the next
sync cycle or pass `target="local"` to `EvalLogger.from_run_info(...)` to land artifacts on the
GPU-node local relay first; they propagate to central with the next sync.

---

## Next steps

| Document | Content |
|:---|:---|
| [`../20_INTEGRATION_CONTRACT.md`](../20_INTEGRATION_CONTRACT.md) | Eval / record contract details |
| [`../21_ADAPTER_GUIDE.md`](../21_ADAPTER_GUIDE.md) | Generic adapter authoring guide |
| [`../22_EXTERNAL_LOGGER_HANDOFF.md`](../22_EXTERNAL_LOGGER_HANDOFF.md) | observer→consumer-logger contract |
| [`sharpa.md`](sharpa.md) | Upstream sharpa-rl-lab adapter (use when `rl_isaaclab.scripts.eval` works) |
| `nexus/docs/32_EVAL_ARTIFACT_INGESTION.md` | nexus `EvalLogger` API + sidecar schema |
