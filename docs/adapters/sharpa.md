# Adapter — sharpa-rl-lab

[`sharpa-rl-lab`](../../..) ships its own PPO / ProprioAdapt stack built on
`DirectRLEnv` + `GymStyleEnvWrapper`. The adapter lives entirely inside the
sharpa repo; observer has zero sharpa-specific code.

- Metrics script: `rl_isaaclab.scripts.eval`
- Record script: `rl_isaaclab/scripts/record.py`
- Expected env instrumentation: `sharpa_wave_env.py` exposes the
  `eval/success`, `eval/slip_detected`, `eval/fingertip_forces`,
  `eval/joint_velocities`, `eval/joint_torques`, `eval/pos_error`,
  `eval/rot_error_deg`, and `eval/init_{roll,pitch,yaw}_deg` keys in the
  gym `info` dict — the eval script slices these per env to build
  `episodes.json`.

Observation normalization (`running_mean_std`), checkpoint loading
(`restore_test`), and algorithm selection (`agent_cfg["algo"]`) all follow
sharpa's training-time conventions — no extra wiring is needed.

## Config stanza

Paste into `observer/configs/eval_config.yaml`:

```yaml
runtime:
  task: "Isaac-Inhand-Rotate-Sharpa-Wave-v0"
  eval_module: "rl_isaaclab.scripts.eval"
  record_script: "rl_isaaclab/scripts/record.py"
  num_envs: 4
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"

  # Optional: forward sharpa-specific flags to eval.py
  # extra_eval_args:
  #   - "--cache=/path/to/grasps.pkl"
```

## Running

```bash
cd /path/to/observer
python eval_runner.py --checkpoint /path/to/sharpa_run/model_5000.pth
```

The metrics stage invokes `python -m rl_isaaclab.scripts.eval` from whichever
Python env has both `sharpa-rl-lab` and `observer` installed (`pip install
-e .` in each). The record stage launches sharpa's `record.py` under
`isaaclab.sh`; sharpa in turn imports `observer.isaac.{camera_controller,
recorder}` as utility libraries.

## Verifying the integration

A quick sanity check once both repos are installed:

```bash
# 1. Headless eval only — skip video for fast turnaround
python eval_runner.py --checkpoint /path/to/model_5000.pth --skip_video

# 2. Inspect outputs
cat eval_results/*/metrics.json     | jq '.success_rate, .num_episodes'
cat eval_results/*/episodes.json    | jq 'length'
```

The `success_rate` should match what `rl_isaaclab/scripts/play.py` reports
on the same checkpoint (± episode sampling noise). If it does not, the
likely culprits are (a) `running_mean_std` not being restored
(check the `restore_test` call) or (b) the gym `info` keys listed above not
being populated by `sharpa_wave_env.py::_get_rewards`.
