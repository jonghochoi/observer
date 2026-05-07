# 🔌 20 · Integration Contract

## Table of contents

- [TL;DR](#tldr)
- [Prerequisites](#prerequisites)
- [§1 — Eval script contract](#1--eval-script-contract)
- [§2 — Record script contract (optional)](#2--record-script-contract-optional)
- [§3 — Minimum env instrumentation](#3--minimum-env-instrumentation)
- [§4 — Post-training upload glue](#4--post-training-upload-glue)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

- Observer runs **two kinds of subprocesses**: a headless eval script and a GUI record script.
- Each script just needs to satisfy the contract below. The RL framework doesn't matter.
- Minimum requirement: the eval script outputs `metrics.json` and `episodes.json` in the defined schema.

---

## Prerequisites

- Observer installed (`pip install -e .` in the observer root)
- The `runtime` section of `observer/configs/eval_config.yaml` is filled in
- Check existing adapters: [`docs/adapters/`](./adapters/) — if your framework is already there, you can use it directly

---

## §1 — Eval script contract

Observer invokes the eval module as:

```bash
python -m <runtime.eval_module> \
    --task=<runtime.task> \
    --load_path=<checkpoint.pth> \
    --device=<runtime.device> \
    --seed=<runtime.seed> \
    --num_envs=<runtime.num_envs> \
    --num_episodes=<metrics.num_eval_episodes> \
    --metrics_output=<out_dir>/metrics.json \
    --episodes_output=<out_dir>/episodes.json \
    --headless \
    <runtime.extra_eval_args...>
```

### ── Required CLI flags

| Flag | Description |
|:---|:---|
| `--task` | Task identifier. Forwarded from `runtime.task` verbatim. |
| `--load_path` | Path to the checkpoint file. |
| `--device` | Torch device (e.g. `cuda:0`). |
| `--seed` | Integer RNG seed. |
| `--num_envs` | Parallel env count. |
| `--num_episodes` | Target number of episodes to collect. |
| `--metrics_output` | Where to write the aggregated metrics JSON. |
| `--episodes_output` | Where to write the per-episode JSON. |
| `--headless` | Always passed by Observer. Your script may ignore it if irrelevant. |

Additional flags in `runtime.extra_eval_args` are forwarded as-is (e.g. grasp cache path).

### ── Required outputs

#### ▸ `metrics.json`

Using `observer.pipeline.metrics_collector.MetricsCollector` produces the correct schema automatically. If writing manually:

```json
{
  "checkpoint": "model_5000",
  "num_episodes": 50,
  "success_rate": 0.84,
  "contact_force_rms_mean": 0.41, "contact_force_rms_std": 0.07,
  "joint_velocity_rms_mean": 0.12, "joint_velocity_rms_std": 0.03,
  "slip_events_per_episode": 0.9,
  "mean_episode_length": 180.2, "std_episode_length": 22.1,
  "object_pos_error_mm_mean": 3.1,
  "object_rot_error_deg_mean": 6.4,
  "energy_J_mean": 1.32, "energy_J_std": 0.4,
  "failure_distribution": {"late_slip": 0.22, "timeout": 0.4},
  "dominant_failure_mode": "timeout",
  "extra": {}
}
```

Minimum fields (missing fields cause the corresponding analysis to be skipped):

```json
{"checkpoint": "...", "num_episodes": 50, "success_rate": 0.84}
```

#### ▸ `episodes.json`

A JSON array of per-episode dicts. Each entry:

```json
{
  "success": true,
  "length": 180,
  "contact_forces":   [0.41, 0.39],
  "joint_velocities": [0.12, 0.11],
  "slip_count": 1,
  "final_pos_error_m": 0.004,
  "final_rot_error_deg": 3.1,
  "energy_J": 1.28,
  "init_roll_deg": 12.0,
  "init_pitch_deg": -4.3,
  "init_yaw_deg": 0.0,
  "init_pos_x": 0.0, "init_pos_y": 0.0, "init_pos_z": 0.05,
  "failure_mode": "unknown"
}
```

> 💡 `failure_mode` is filled in downstream by `FailureModeClassifier`. Leaving it as `"unknown"` is fine.
> Empty arrays for `contact_forces` and `joint_velocities` cause failure classification and energy analysis to be skipped for that episode.

Schema definition: `observer.pipeline.metrics_collector.EpisodeStats`

---

## §2 — Record script contract (optional)

If video recording is not needed, set `skip_video: true` in your config and skip this section.

Observer invokes the record script as:

```bash
<runtime.isaac_lab_path> -p <runtime.record_script> \
    --task=<runtime.task> \
    --load_path=<checkpoint.pth> \
    --device=<runtime.device> \
    --seed=<runtime.seed> \
    --num_envs=<runtime.num_envs> \
    --record_mode \
    --camera_config=<out_dir>/camera_poses.json \
    --video_output_dir=<out_dir>/videos \
    --video_fps=<video.fps> \
    --video_resolution=<WxH> \
    <runtime.extra_record_args...>
```

The script must:

1. Launch Isaac Lab (or your simulator) in GUI mode.
2. Build the env/policy using your training-time stack.
3. Read camera poses from `--camera_config` (a JSON array written by Observer: `{name, eye, target, record_steps}`).
4. Sweep through all poses in order, saving `--video_output_dir/<name>.mp4`.

You can import the utility libraries (`observer.isaac.CameraController`, `observer.isaac.VideoRecorder`) to use them. They have no dependencies outside the Isaac Sim Replicator module.

---

## §3 — Minimum env instrumentation

`FailureModeClassifier` and `StateCoverageAnalyzer` consume per-episode fields. Your environment must expose at minimum:

| Episode field | Typical source |
|:---|:---|
| `success` | Terminal condition in your env / reward |
| `length` | Step counter until `done` |
| `final_pos_error_m`, `final_rot_error_deg` | Goal pose minus terminal pose |
| `init_roll/pitch/yaw_deg`, `init_pos_{x,y,z}` | Sampled initial object pose |
| `slip_count` *(optional)* | Detect via tactile / contact signal transitions |
| `contact_forces`, `joint_velocities` *(optional)* | Per-step RMS aggregates |

Observer never touches the env directly. Your eval script reads these values from the gym `info` dict, internal buffers, or whatever instrumentation you already have, and writes `episodes.json`.

---

## §4 — Post-training upload glue

Observer writes a deterministic output tree and exposes `locate_results` /
`read_metrics` helpers in `observer.pipeline.result_locator` so downstream
training repos can forward results to any experiment tracker without observer
knowing which one.

> 📖 Full API reference, output directory layout, `ObserverResults` field
> table, `read_metrics` behaviour, and a worked glue-script example are in
> [`22_EXTERNAL_LOGGER_HANDOFF.md`](./22_EXTERNAL_LOGGER_HANDOFF.md).

---

## Troubleshooting

**`coverage/` or failure distribution is empty**

→ Usually caused by `episodes.json` not being generated.
Check that the eval script is writing the file and that the schema matches `EpisodeStats`.

**`metrics.json` exists but `success_rate` is 0.0**

→ Checkpoint loading failure or observation normalization not restored.
Run the eval script standalone and inspect the output:
```bash
python -m <runtime.eval_module> --task=... --load_path=... --num_episodes=5 \
    --metrics_output=/tmp/test_metrics.json --episodes_output=/tmp/test_ep.json --headless
cat /tmp/test_metrics.json | python -m json.tool
```

**Unknown flags error**

→ A flag in `runtime.extra_eval_args` is not recognized by the eval script.
Add a `parse_known_args` pattern to the eval script to silently ignore unknown flags.

---

## Next steps

| Document | Content |
|:---|:---|
| [`21_ADAPTER_GUIDE.md`](./21_ADAPTER_GUIDE.md) | Step-by-step guide for writing a new framework adapter |
| [`docs/adapters/sharpa.md`](./adapters/sharpa.md) | Complete sharpa-rl-lab example |
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Full metrics + failure taxonomy reference |
