# Observer Integration Contract

Observer is intentionally framework-agnostic. It launches two kinds of
subprocesses — a headless **eval script** and a GUI **record script** — and
only requires that each satisfies the contract below. Anything else is up to
your RL stack (PPO / RSL-RL / CleanRL / …).

Already integrated frameworks are documented in [`adapters/`](./adapters/).

---

## 1. Eval script contract (metrics collection)

Observer invokes the eval module as:

```
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

### Required CLI flags

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
| `--headless` | Observer always passes this; your script may ignore it if irrelevant. |

Unknown flags supplied by `runtime.extra_eval_args` are forwarded as-is — use
them for framework-specific knobs (e.g. grasp cache paths).

### Required outputs

#### `metrics.json`

A single JSON object. If your script uses
`observer.pipeline.metrics_collector.MetricsCollector`, the schema is
`EvalResult.to_dict()`:

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
  "failure_distribution": {"late_slip": 0.22, "timeout": 0.4, "...": 0.0},
  "dominant_failure_mode": "timeout",
  "extra": {}
}
```

If you skip `MetricsCollector`, emit at minimum:

```json
{"checkpoint": "...", "num_episodes": N, "success_rate": 0.0}
```

The remaining fields are optional — observer will degrade gracefully.

#### `episodes.json`

A JSON **array** of per-episode dicts. Each entry must look like:

```json
{
  "success": true,
  "length": 180,
  "contact_forces":   [0.41, 0.39, ...],
  "joint_velocities": [0.12, 0.11, ...],
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

`failure_mode` is set downstream by `FailureModeClassifier`, so leaving it
as `"unknown"` is fine. `contact_forces` and `joint_velocities` are per-step
RMS scalars; pass empty lists if your env cannot provide them (failure
classification and energy analysis will just be skipped for that episode).

The schema matches `observer.pipeline.metrics_collector.EpisodeStats`.

---

## 2. Record script contract (optional, video stage)

Observer invokes the record script as:

```
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

1. Launch Isaac Lab (or whatever your simulator needs) with a GUI.
2. Build the env/policy using your training-time stack.
3. Read the camera poses from `--camera_config` (same JSON schema observer
   writes: an array of `{name, eye, target, record_steps}`).
4. Sweep the camera through all poses, capturing frames to
   `--video_output_dir/<name>.mp4` at `--video_fps` / `--video_resolution`.

You can reuse `observer.isaac.CameraController` and
`observer.isaac.VideoRecorder` / `record_all_views` as a utility library —
they expose a framework-neutral interface and do not import anything outside
of Isaac Sim's Replicator module.

If you do not need video recording, set `skip_video: true` in your config
and leave `runtime.record_script` blank.

---

## 3. Minimum env instrumentation

Observer's failure classifier and state coverage analyzer consume the
per-episode fields above. Your env must therefore expose, at minimum:

| Episode field | Where it usually comes from |
|:---|:---|
| `success` | Terminal condition in your env / reward |
| `length` | Step counter until `done` |
| `final_pos_error_m`, `final_rot_error_deg` | Goal pose minus terminal pose |
| `init_roll/pitch/yaw_deg`, `init_pos_{x,y,z}` | Sampled initial object pose |
| `slip_count` (optional) | Detect via tactile / contact signal transitions |
| `contact_forces`, `joint_velocities` (optional) | Per-step RMS aggregates |

How you surface these is your choice — your eval script can read them from
the gym `info` dict, from internal buffers, or from whatever instrumentation
you already have. Observer never touches the env directly.

---

## 4. Writing a new adapter

1. In your framework's repo, add an `eval.py` that satisfies §1.
2. (Optional) Add a `record.py` that satisfies §2.
3. Export the CLI invocation as a short YAML stanza that users can drop into
   `observer/configs/eval_config.yaml` under `runtime:`.
4. Open a PR to `observer/docs/adapters/<framework>.md` with the stanza and
   any framework-specific gotchas.

See [`adapters/sharpa.md`](./adapters/sharpa.md) for a worked example.
