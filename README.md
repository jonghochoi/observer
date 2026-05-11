<div align="center">

# 👁️ OBSERVER · Automated Evaluation Pipeline

<img src="docs/LOGO.png" alt="OBSERVER Logo" width="600">

**Stop watching videos. Start reading data.**

*Automated checkpoint evaluation · Multi-view recording · Failure diagnosis · Experiment tracking*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

> ### 📖 New here? Start with the onboarding guide.
>
> **한글 온보딩 (Korean guide):** [`docs/observer_guide.html`](docs/observer_guide.html)
> — download and open in a browser for a full visual walkthrough of OBSERVER.
>
> ```bash
> git clone https://github.com/jonghochoi/observer.git
> open observer/docs/observer_guide.html   # macOS
> # xdg-open observer/docs/observer_guide.html   # Linux
> ```
>
> Every team-agreed rule and engineering invariant lives on **one page**: [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) — *5 min, English*.

---

## Why this exists

Dexterous manipulation research generates checkpoints far faster than humans can meaningfully review.
`success_rate = 0.91` looks great — but it may hide 3 slips/episode, 5× energy usage, and joint velocity spikes.
Manual review always introduces selection, recency, and fatigue bias.

> 💡 **Design invariant — subprocess-only boundary.** Observer never imports torch, Isaac, or your RL stack.
> Your eval / record scripts run as subprocesses; the only coupling is the JSON contract in
> [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md). That is what makes it
> truly framework-agnostic (PPO / RSL-RL / CleanRL / anything).

---

## What this pipeline does

**One command. Everything automated.**

```bash
observer --checkpoint_dir runs/ --recursive \
    --auto_select --select_weights hardware_safe
```

| # | Stage | What it does |
|:---:|:---|:---|
| 1 | **Metrics aggregation** | Reads `metrics.json` written by your eval script — schema is yours to define |
| 2 | **Failure mode classification** | Priority rule chain over `episodes.json` — no training data required |
| 3 | **State coverage analysis** | 2D success heatmap over whichever pose axes you emit |
| 4 | **Multi-view video recording** | Multi-viewpoint capture + grid mp4 (driven by your record script) |
| 5 | **Experiment tracking** | TensorBoard auto-detection (optional) |
| 6 | **Multi-objective ranking** | Weighted score across whichever metrics you opt in to |
| 7 | **HTML report** | Self-contained: charts, pie graphs, videos, heatmaps |

> ⚠️ **None of the stages have a fixed schema.** Observer is task-agnostic — what each stage
> consumes is determined by the JSON contract in
> [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md). The only universally required
> fields are `checkpoint`, `num_episodes`, and `success_rate`. Anything else is
> opt-in: emit it and the corresponding analysis lights up; omit it and that analysis is silently skipped.

### ── Bundled dexterous-manipulation defaults

Out of the box observer ships with a metric set and failure taxonomy tuned for **dexterous manipulation**
research. They are examples of what the contract enables, not a fixed pipeline — drop fields you don't
have, add your own metric keys, or extend `FailureModeClassifier` with new rules
(see the *"When adding new features"* checklists in [`CLAUDE.md`](CLAUDE.md)).

**Default metric keys** — `success_rate`, `contact_force_rms` (N), `joint_velocity_rms` (rad/s),
`slip_events_per_episode`, `mean_episode_length`, `object_pos_error_mm`, `object_rot_error_deg`, `energy_J`.
Full table with units and interpretation: [`docs/30_METRICS_REFERENCE.md`](docs/30_METRICS_REFERENCE.md).

**Default failure rules** — priority-ordered; first match wins, and the dominant mode hints at what to
fix in the reward / curriculum:

| Priority | Mode | If dominant → consider |
|:---:|:---|:---|
| 1 | `early_drop` | Grasp initialization / curriculum |
| 2 | `singularity_hit` | Joint velocity limit / singularity avoidance reward |
| 3 | `late_slip` | Slip penalty in reward |
| 4 | `contact_loss` | Contact force reward |
| 5 | `repose_failure` | Goal pose error reward |
| 6 | `timeout` | Simplify curriculum / improve exploration |

These are appropriate for *fingertip-grasp manipulation tasks*. For other domains (locomotion, navigation,
bimanual, etc.) you would replace rules and metric keys — observer's analysis code reads them generically.

**Default ranking presets** (`--select_weights`) — also tuned for the same manipulation use case;
override with your own weights when the trade-offs differ:

| Preset | When to use |
|:---|:---|
| `balanced` | Everyday experiment comparison — all metrics weighted evenly |
| `hardware_safe` | Pre-deployment selection — heavy penalties on slip and energy |
| `performance_first` | Ablation studies — prioritizes raw task success rate |

---

## Quick start

```bash
# 1) Install
pip install numpy pyyaml matplotlib && sudo apt install ffmpeg
pip install -e .

# 2) Validate configuration
observer doctor

# 3) Run
observer --checkpoint runs/exp_001/model_5000.pth         # single
observer --checkpoint_dir runs/exp_001/                   # directory
observer --checkpoint_dir runs/ --recursive --latest_only # recursive
observer --checkpoint_dir runs/ \                         # ranking + deploy
    --auto_select --select_weights hardware_safe --deploy_top_k 2      
observer --checkpoint_dir runs/ --dry_run                 # validation only
```

> **First time?** Run `observer doctor` to validate your configuration, then start with `--dry_run` to verify the pipeline structure before running Isaac.

Observer is **framework-agnostic**. It runs your eval/record scripts as subprocesses,
as long as those scripts satisfy the contract in [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md).

> 📖 Writing an adapter for a new framework? Start at [`docs/21_ADAPTER_GUIDE.md`](docs/21_ADAPTER_GUIDE.md).
> Already have an env but `episodes.json` looks empty? See
> [`docs/23_ENV_INSTRUMENTATION.md`](docs/23_ENV_INSTRUMENTATION.md) for the env→`info`→`episodes.json` flow
> and a minimal worked patch. Forwarding observer outputs to MLflow / W&B?
> [`docs/22_EXTERNAL_LOGGER_HANDOFF.md`](docs/22_EXTERNAL_LOGGER_HANDOFF.md) covers `result_locator`.

```yaml
# observer/configs/eval_config.yaml
runtime:
  task: "<your-task-id>"
  eval_module: "your_pkg.scripts.eval"
  record_script: "your_pkg/scripts/record.py"
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
```

---

## Architecture

```
observer.eval_runner
    ├── PipelineOrchestrator (per checkpoint)
    │     ├── MetricsCollector  → metrics.json
    │     ├── FailureModeClassifier
    │     ├── StateCoverageAnalyzer
    │     └── CameraController (subprocess → Isaac)
    ├── CheckpointSelector  (multi-objective ranking)
    └── ReportGenerator     (eval_report.html)
```

> 📖 Full architecture, file map, and output structure: [`docs/10_ARCHITECTURE.md`](docs/10_ARCHITECTURE.md)

---

## Further Reading

> Filename prefix conveys reading order. **Everyone reads `00_PRINCIPLES.md` first**, then picks up the relevant track below.

| # | Document | Audience | Content |
|:---:|:---|:---|:---|
| **HTML** | [`docs/observer_guide.html`](docs/observer_guide.html) | 🇰🇷 Team members | **Korean onboarding guide** — download and open locally for a full visual walkthrough |
| **00** | [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) | Everyone | **Read first.** Why evaluation matters, design principles |
| **10** | [`docs/10_ARCHITECTURE.md`](docs/10_ARCHITECTURE.md) | Engineers | System structure, file map, dependencies |
| **20** | [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md) | Developers | eval/record script contract |
| **21** | [`docs/21_ADAPTER_GUIDE.md`](docs/21_ADAPTER_GUIDE.md) | Developers | Writing a new framework adapter |
| **22** | [`docs/22_EXTERNAL_LOGGER_HANDOFF.md`](docs/22_EXTERNAL_LOGGER_HANDOFF.md) | Developers | Forwarding observer outputs to a downstream logger (MLflow, W&B, etc.) via `result_locator` |
| **23** | [`docs/23_ENV_INSTRUMENTATION.md`](docs/23_ENV_INSTRUMENTATION.md) | Developers | How `info` flows env → eval CLI → `episodes.json` → observer; required keys, per-env vs scalar contract, worked patch |
| **30** | [`docs/30_METRICS_REFERENCE.md`](docs/30_METRICS_REFERENCE.md) | Everyone | 8 metrics + 6-class failure classification |
| **31** | [`docs/31_CHECKPOINT_RANKING.md`](docs/31_CHECKPOINT_RANKING.md) | Everyone | Multi-objective ranking + state coverage |
| — | [`docs/adapters/sharpa.md`](docs/adapters/sharpa.md) | sharpa users | sharpa-rl-lab integration example |
