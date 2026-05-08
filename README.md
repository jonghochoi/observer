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

> ### 📖 New here? Read this first.
>
> Every team-agreed rule and engineering invariant lives on **one page**: [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) — *5 min, English*.
>
> 한글 온보딩 트랙은 [`docs/ko/`](docs/ko/)에서 시작하세요.

---

## Why this exists

Dexterous manipulation research generates checkpoints far faster than humans can meaningfully review.
`success_rate = 0.91` looks great — but it may hide 3 slips/episode, 5× energy usage, and joint velocity spikes.
Manual review always introduces selection, recency, and fatigue bias.

---

## What this pipeline does

**One command. Everything automated.**

```bash
observer --checkpoint_dir runs/ --recursive \
    --auto_select --select_weights hardware_safe
```

| # | Step | Output |
|:---:|:---|:---|
| 1 | **Metrics collection** | `metrics.json` — 8 quantitative metrics |
| 2 | **Failure mode classification** | 6-class rule chain, no training data required |
| 3 | **State coverage analysis** | roll × pitch success heatmap |
| 4 | **Multi-view video recording** | 5 viewpoints + 2×3 grid mp4 |
| 5 | **Experiment tracking** | TensorBoard auto-detection |
| 6 | **Multi-objective ranking** | Weighted score: success rate, slip, energy, pose error |
| 7 | **HTML report** | Includes charts, pie graphs, videos, and heatmaps |

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

> **First time?** Run `./scripts/setup.sh` to check prerequisites,
> validate your configuration with `observer doctor`, then start with `--dry_run`.

Observer is **framework-agnostic**. It runs your eval/record scripts as subprocesses,
as long as those scripts satisfy the contract in [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md).

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

## Documentation

| Document | Audience | Content |
|:---|:---|:---|
| [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) | Everyone | Why evaluation matters, design principles |
| [`docs/10_ARCHITECTURE.md`](docs/10_ARCHITECTURE.md) | Engineers | System structure, file map, dependencies |
| [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md) | Developers | eval/record script contract |
| [`docs/21_ADAPTER_GUIDE.md`](docs/21_ADAPTER_GUIDE.md) | Developers | Writing a new framework adapter |
| [`docs/22_EXTERNAL_LOGGER_HANDOFF.md`](docs/22_EXTERNAL_LOGGER_HANDOFF.md) | Developers | Forwarding observer outputs to a downstream logger (MLflow, W&B, etc.) via `result_locator` |
| [`docs/23_ENV_INSTRUMENTATION.md`](docs/23_ENV_INSTRUMENTATION.md) | Developers | How `info` flows env → eval CLI → `episodes.json` → observer; required keys, per-env vs scalar contract, worked patch |
| [`docs/30_METRICS_REFERENCE.md`](docs/30_METRICS_REFERENCE.md) | Everyone | 8 metrics + 6-class failure classification |
| [`docs/31_CHECKPOINT_RANKING.md`](docs/31_CHECKPOINT_RANKING.md) | Everyone | Multi-objective ranking + state coverage |
| [`docs/adapters/sharpa.md`](docs/adapters/sharpa.md) | sharpa users | sharpa-rl-lab integration example |
| [`docs/adapters/hand_expert.md`](docs/adapters/hand_expert.md) | hand_expert users | hand_expert + nexus `EvalLogger` end-to-end worked example |
| [`docs/ko/01_INTRO.md`](docs/ko/01_INTRO.md) | 🇰🇷 Team members | Full Korean onboarding |
