# 🔧 21 · Writing a New Adapter

## Table of contents

- [TL;DR](#tldr)
- [Prerequisites](#prerequisites)
- [Step 1 — Write eval.py](#step-1--write-evalpy)
- [Step 2 — Write record.py (optional)](#step-2--write-recordpy-optional)
- [Step 3 — Extract the config stanza](#step-3--extract-the-config-stanza)
- [Step 4 — Open an adapter doc PR](#step-4--open-an-adapter-doc-pr)
- [Worked example — sharpa-rl-lab](#worked-example--sharpa-rl-lab)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

1. Add `eval.py` to your framework repo — outputs `metrics.json` + `episodes.json`
2. (Optional) Add `record.py` — reads camera poses and saves video
3. Extract a `runtime:` stanza for `eval_config.yaml`
4. Open a PR to `observer/docs/adapters/<framework>.md`

---

## Prerequisites

- Observer installed (`pip install -e .`)
- Familiar with the integration contract: [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md)
- Pipeline structure verified with a dry run:
  ```bash
  python eval_runner.py --checkpoint_dir runs/ --dry_run
  ```

---

## Step 1 — Write eval.py

Add `eval.py` (or `scripts/eval.py`) to your framework repo.

### ── Accept the required CLI flags

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--load_path", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--metrics_output", required=True)
parser.add_argument("--episodes_output", required=True)
parser.add_argument("--headless", action="store_true")
# add framework-specific flags as needed
args, _ = parser.parse_known_args()   # silently ignore unknown flags
```

### ── Write the outputs

Using `MetricsCollector` (recommended):

```python
from observer.pipeline.metrics_collector import MetricsCollector, EpisodeStats

collector = MetricsCollector()

for episode in run_episodes(env, policy, args.num_episodes):
    collector.add(EpisodeStats(
        success=episode.success,
        length=episode.length,
        contact_forces=episode.contact_forces,   # per-step RMS list
        joint_velocities=episode.joint_vels,
        slip_count=episode.slip_count,
        final_pos_error_m=episode.pos_err,
        final_rot_error_deg=episode.rot_err,
        energy_J=episode.energy,
        init_roll_deg=episode.init_roll,
        init_pitch_deg=episode.init_pitch,
        init_yaw_deg=episode.init_yaw,
        init_pos_x=episode.init_x,
        init_pos_y=episode.init_y,
        init_pos_z=episode.init_z,
    ))

result = collector.aggregate(checkpoint_name=Path(args.load_path).stem)
result.save(args.metrics_output, args.episodes_output)
```

For the manual schema see [`20_INTEGRATION_CONTRACT.md §1`](./20_INTEGRATION_CONTRACT.md#1--eval-script-contract).

### ── Verify

```bash
python -m <your_pkg>.scripts.eval \
    --task=<task_id> --load_path=<ckpt.pth> \
    --num_episodes=5 \
    --metrics_output=/tmp/m.json --episodes_output=/tmp/ep.json --headless

# inspect outputs
cat /tmp/m.json  | python -m json.tool
cat /tmp/ep.json | python -m json.tool | head -30
```

`success_rate` should match what your policy actually achieves.

---

## Step 2 — Write record.py (optional)

Skip this step if you don't need video. Set `skip_video: true` in your config.

### ── Use the Observer utilities (recommended)

```python
import json
from observer.isaac.camera_controller import CameraController
from observer.isaac.recorder import record_all_views

with open(args.camera_config) as f:
    camera_poses = json.load(f)

controller = CameraController(viewport)
record_all_views(
    controller=controller,
    poses=camera_poses,
    output_dir=args.video_output_dir,
    fps=args.video_fps,
)
```

`CameraController` and `record_all_views` have no dependencies outside the Replicator module.

### ── Verify

```bash
python eval_runner.py --checkpoint <ckpt.pth> \
    --num_episodes 1 --video_steps 30
ls eval_results/*/videos/
```

`front.mp4`, `side.mp4`, `combined_grid.mp4` etc. should be present.

---

## Step 3 — Extract the config stanza

Prepare a YAML stanza to paste under `runtime:` in `observer/configs/eval_config.yaml`:

```yaml
runtime:
  task: "<your-task-id>"
  eval_module: "<your_pkg>.scripts.eval"        # python -m <eval_module>
  record_script: "<your_pkg>/scripts/record.py" # isaaclab.sh -p <record_script>
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
  num_envs: 4
  device: "cuda:0"
  seed: 42

  # framework-specific flags (optional)
  # extra_eval_args:
  #   - "--cache=/path/to/grasps.pkl"
```

---

## Step 4 — Open an adapter doc PR

Write `observer/docs/adapters/<framework>.md` and open a PR.

Include:
- One or two sentences describing the framework
- Metrics / record script locations
- Env instrumentation requirements (gym `info` keys, etc.)
- Config stanza (copy-pasteable)
- Run command
- Verification checklist

Use [`docs/adapters/sharpa.md`](./adapters/sharpa.md) as a template.

---

## Worked example — sharpa-rl-lab

sharpa-rl-lab is a PPO / ProprioAdapt stack built on `DirectRLEnv` + `GymStyleEnvWrapper`.

- **eval script**: `rl_isaaclab.scripts.eval` (invoked with python -m)
- **record script**: `rl_isaaclab/scripts/record.py`
- **env instrumentation**: `sharpa_wave_env.py` exposes `eval/success`, `eval/slip_detected`, etc. in the gym `info` dict
- **config stanza**: see [`docs/adapters/sharpa.md`](./adapters/sharpa.md)

---

## Troubleshooting

**`ModuleNotFoundError` when running `python -m <eval_module>`**

→ The framework package and observer are not in the same Python environment.
```bash
pip install -e /path/to/your-framework
pip install -e /path/to/observer
```

**`episodes.json` is empty or has length 0**

→ Missing `EpisodeStats` fields, or the env does not populate the init pose on reset.
Check that `init_roll_deg` / `init_pitch_deg` / `init_yaw_deg` reflect the actual sampled values.

**Camera does not move during the record stage**

→ Check that `--camera_config` points to the correct path and the JSON schema is `[{name, eye, target, record_steps}, ...]`.

---

## Next steps

| Document | Content |
|:---|:---|
| [`docs/adapters/sharpa.md`](./adapters/sharpa.md) | Complete sharpa adapter example |
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Full metrics + failure taxonomy reference |
| [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) | Detailed schema reference |
