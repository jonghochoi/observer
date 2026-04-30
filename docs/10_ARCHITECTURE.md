# 🏗️ 10 · Architecture

## 📑 Table of Contents

- [⚡ TL;DR](#-tldr)
- [🔄 Pipeline Overview](#-pipeline-overview)
- [🗂️ Repository File Map](#️-repository-file-map)
- [📹 Output Directory Structure](#-output-directory-structure)
- [🛠️ Dependencies](#️-dependencies)
- [🗺️ Next Steps](#️-next-steps)

---

## ⚡ TL;DR

- `eval_runner.py` → `PipelineOrchestrator` → [MetricsCollector, FailureModeClassifier, StateCoverageAnalyzer, CameraController] → `CheckpointSelector` → `ReportGenerator`
- Isaac is invoked as a subprocess only. Observer code never imports Isaac directly.
- Outputs are written under `eval_results/` in a per-checkpoint directory.

---

## 🔄 Pipeline Overview

```
eval_runner.py
    │
    ├── 📡 ExperimentTracker    W&B + TensorBoard (auto-detected)
    │
    ├── 🔄 PipelineOrchestrator (per checkpoint)
    │     ├── [1] 📦 MetricsCollector        → metrics.json
    │     ├── [2] 🔍 FailureModeClassifier   → failure distribution
    │     ├── [3] 🗺️ StateCoverageAnalyzer   → PNG heatmaps
    │     ├── [4] 🎬 CameraController        → viewport sweep
    │     └── [5] 📡 ExperimentTracker       → logging
    │
    ├── 🏆 CheckpointSelector   Multi-objective scoring + deploy
    │
    └── 📄 ReportGenerator      eval_report.html
```

### Component Roles

| Component | File | Role |
|:---|:---|:---|
| `PipelineOrchestrator` | `pipeline/orchestrator.py` | Coordinates the full per-checkpoint cycle |
| `MetricsCollector` | `pipeline/metrics_collector.py` | Collects and aggregates per-step episode metrics |
| `FailureModeClassifier` | `pipeline/failure_classifier.py` | Classifies failure types via a priority rule chain |
| `StateCoverageAnalyzer` | `pipeline/state_coverage.py` | Analyzes initial pose space coverage over roll × pitch |
| `ExperimentTracker` | `pipeline/experiment_tracker.py` | Auto-detects and logs to W&B / TensorBoard |
| `CheckpointSelector` | `pipeline/auto_select.py` | Selects top-k checkpoints via multi-objective scoring |
| `CameraController` | `isaac/camera_controller.py` | Isaac Sim viewport control (utility library) |
| `VideoRecorder` | `isaac/recorder.py` | Replicator-based video capture (utility library) |
| `ReportGenerator` | `report/report_generator.py` | Generates a self-contained HTML report |

> [!IMPORTANT]
> `CameraController` and `VideoRecorder` are **utility libraries**. Observer does not call them directly —
> the user's record script imports and uses them. Isaac is only ever invoked as a subprocess.

---

## 🗂️ Repository File Map

```
observer/
├── 🚀 eval_runner.py              Entry point (exposed as `observer` CLI after install)
├── 🎨 brand.py                    Console banner / branding
├── 📦 requirements.txt            Core runtime dependencies
├── 🏗️ setup.py                    Package install script
├── configs/
│   ├── 📝 eval_config.py          Config dataclass
│   └── ⚙️ eval_config.yaml        ← Edit this per experiment
├── pipeline/
│   ├── 🔄 orchestrator.py         Per-checkpoint cycle coordinator
│   ├── 📦 metrics_collector.py    Per-step metric accumulator
│   ├── 🔍 failure_classifier.py   Rule-based failure mode taxonomy
│   ├── 🗺️ state_coverage.py       Initial pose coverage analysis
│   ├── 📡 experiment_tracker.py   W&B / TensorBoard integration
│   └── 🏆 auto_select.py          Multi-objective checkpoint scoring
├── isaac/
│   ├── 🎥 camera_controller.py    Isaac Sim viewport control (utility)
│   └── 🎬 recorder.py             Replicator-based video capture (utility)
├── report/
│   └── 📄 report_generator.py     HTML report generator
├── tactile/
│   └── 👆 overlay.py              Deform map video overlay
└── docs/
    ├── 00_PRINCIPLES.md
    ├── 10_ARCHITECTURE.md          ← you are here
    ├── 20_INTEGRATION_CONTRACT.md
    ├── 21_ADAPTER_GUIDE.md
    ├── 30_METRICS_REFERENCE.md
    ├── 31_CHECKPOINT_RANKING.md
    ├── adapters/
    │   └── sharpa.md
    └── ko/
        ├── README.md
        └── 01_INTRO.md
```

> 💡 **Top 3 files you'll visit most**
> 1. `configs/eval_config.yaml` — edit per experiment
> 2. `docs/20_INTEGRATION_CONTRACT.md` — the only contract your framework needs to satisfy
> 3. `eval_runner.py` — CLI flag reference

---

## 📹 Output Directory Structure

After a run, outputs are written under `eval_results/`:

```
📁 eval_results/
├── 📄 eval_report.html                         ← open in browser (start here)
├── 🏆 best/
│   ├── 🥇 rank01__model_6000.pth               symlink to original
│   ├── 🥈 rank02__model_4000.pth
│   └── 📋 selection_meta.json
└── 📁 exp_001__model_5000__20240117_143022/
    ├── ⚙️  eval_config_snapshot.yaml           reproducibility record
    ├── 📊 metrics.json
    ├── 📝 episodes.json                        per-episode data (when available)
    ├── 📷 camera_poses.json
    ├── 📁 coverage/
    │   ├── 🌡️ success_heatmap.png              success rate over roll × pitch
    │   ├── 🔵 coverage_scatter.png             per-episode scatter by failure mode
    │   └── 📊 pose_histogram.png               sampling distribution
    └── 📁 videos/
        ├── 🎬 front.mp4
        ├── 🎬 side.mp4
        ├── 🎬 top.mp4
        └── 🎬 combined_grid.mp4                all views in one file
```

---

## 🛠️ Dependencies

### Core (required)

```bash
pip install numpy pyyaml matplotlib
sudo apt install ffmpeg          # video encoding
pip install -e .                 # install the observer CLI
```

### Optional

```bash
# experiment tracking (either or both)
pip install wandb
pip install tensorboard

# tactile overlay
pip install opencv-python
```

### Isaac (provided by Isaac Lab installation)

```
omni.isaac.lab
omni.replicator.core
omni.kit.viewport.utility
```

> ⚠️ When optional packages are absent, the corresponding feature is gracefully disabled. The rest of the pipeline continues normally.

### Verify installation

```bash
observer doctor   # validates config and dependencies
```

---

## 🗺️ Next Steps

| Document | Content |
|:---|:---|
| [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) | Contract your eval / record scripts must satisfy |
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Collected metrics + failure mode taxonomy |
| [`31_CHECKPOINT_RANKING.md`](./31_CHECKPOINT_RANKING.md) | Multi-objective ranking + state coverage |
