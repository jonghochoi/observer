# 🏗️ 10 · Architecture

## Table of contents

- [TL;DR](#tldr)
- [Pipeline overview](#pipeline-overview)
- [Repository file map](#repository-file-map)
- [Output directory structure](#output-directory-structure)
- [Dependencies](#dependencies)
- [Next steps](#next-steps)

---

## TL;DR

- `observer.eval_runner` → `PipelineOrchestrator` → [MetricsCollector, FailureModeClassifier, StateCoverageAnalyzer, CameraController] → `CheckpointSelector` → `ReportGenerator`
- Isaac is invoked as a subprocess only. Observer code never imports Isaac directly.
- Outputs are written under `eval_results/` in a per-checkpoint directory.

---

## Pipeline overview

```
observer.eval_runner
    │
    ├── 📡 ExperimentTracker    TensorBoard (auto-detected)
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

### ── Component roles

| Component | File | Role |
|:---|:---|:---|
| `PipelineOrchestrator` | `observer/pipeline/orchestrator.py` | Coordinates the full per-checkpoint cycle |
| `MetricsCollector` | `observer/pipeline/metrics_collector.py` | Collects and aggregates per-step episode metrics |
| `FailureModeClassifier` | `observer/pipeline/failure_classifier.py` | Classifies failure types via a priority rule chain |
| `StateCoverageAnalyzer` | `observer/pipeline/state_coverage.py` | Analyzes initial pose space coverage over roll × pitch |
| `ExperimentTracker` | `observer/pipeline/experiment_tracker.py` | Auto-detects and logs to TensorBoard |
| `CheckpointSelector` | `observer/pipeline/auto_select.py` | Selects top-k checkpoints via multi-objective scoring |
| `CameraController` | `observer/isaac/camera_controller.py` | Isaac Sim viewport control (utility library) |
| `VideoRecorder` | `observer/isaac/recorder.py` | Replicator-based video capture (utility library) |
| `ReportGenerator` | `observer/pipeline/report_generator.py` | Generates a self-contained HTML report |

> [!IMPORTANT]
> `CameraController` and `VideoRecorder` are **utility libraries**. Observer does not call them directly —
> the user's record script imports and uses them. Isaac is only ever invoked as a subprocess.

---

## Repository file map

```
observer/                            ← repository root (metadata only)
├── 📜 pyproject.toml                Package metadata, deps, console script entry
├── 🛠️ setup.sh                      Optional venv bootstrap helper
├── 🛠️ Makefile                      Convenience targets (doctor, eval, sweep)
├── observer/                        ← Python package — discovered via PYTHONPATH=<repo_root>
│   ├── 🚀 eval_runner.py            Entry point (`observer` CLI / `python -m observer.eval_runner`)
│   ├── 🎨 brand.py                  Console banner / branding
│   ├── 🩺 doctor.py                 Pre-flight environment validator
│   ├── configs/
│   │   ├── 📝 eval_config.py        Config dataclass
│   │   └── ⚙️ eval_config.yaml      ← Edit this per experiment
│   ├── pipeline/
│   │   ├── 🔄 orchestrator.py       Per-checkpoint cycle coordinator
│   │   ├── 📦 metrics_collector.py  Per-step metric accumulator
│   │   ├── 🔍 failure_classifier.py Rule-based failure mode taxonomy
│   │   ├── 🗺️ state_coverage.py     Initial pose coverage analysis
│   │   ├── 📡 experiment_tracker.py TensorBoard integration
│   │   ├── 🏆 auto_select.py        Multi-objective checkpoint scoring
│   │   ├── 📄 report_generator.py   HTML report generator
│   │   └── 🔎 result_locator.py     Output layout discovery for glue scripts
│   ├── isaac/
│   │   ├── 🎥 camera_controller.py  Isaac Sim viewport control (utility)
│   │   └── 🎬 recorder.py           Replicator-based video capture (utility)
│   └── viz/
│       └── 👆 tactile_overlay.py    Deform map video overlay
└── docs/
    ├── 00_PRINCIPLES.md
    ├── 10_ARCHITECTURE.md           ← you are here
    ├── 20_INTEGRATION_CONTRACT.md
    ├── 21_ADAPTER_GUIDE.md
    ├── 22_EXTERNAL_LOGGER_HANDOFF.md
    ├── 30_METRICS_REFERENCE.md
    ├── 31_CHECKPOINT_RANKING.md
    ├── adapters/
    │   └── sharpa.md
    └── ko/
        ├── README.md
        └── 01_INTRO.md
```

> 💡 **Top 3 files you'll visit most**
> 1. `observer/configs/eval_config.yaml` — edit per experiment
> 2. `docs/20_INTEGRATION_CONTRACT.md` — the only contract your framework needs to satisfy
> 3. `observer/eval_runner.py` — CLI flag reference

---

## Packaging boundary

Every subdirectory in this repository is a Python package. `pipeline/`, `isaac/`, `viz/`, and `configs/` all carry an `__init__.py` and are picked up by the discovery rule in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["observer*"]
```

This is not accidental — observer is intentionally a **library + CLI hybrid**. Both surfaces are first-class.

### ── Library + CLI hybrid usage

- **CLI** — `[project.scripts]` registers `observer = "observer.eval_runner:main"`. After `pip install -e .` the `observer` console script is available, alongside `python -m observer.eval_runner` for environments where `pip` isn't an option.
- **Library** — external trainer / evaluation scripts import submodules directly:

  ```python
  from observer.pipeline.result_locator    import locate_results       # find result dirs
  from observer.pipeline.metrics_collector import MetricsCollector     # reuse aggregation
  from observer.isaac.camera_controller    import CameraController     # adapter utility
  ```

  The handoff to external trackers (MLflow, W&B, ...) is fully specified in [`22_EXTERNAL_LOGGER_HANDOFF.md`](./22_EXTERNAL_LOGGER_HANDOFF.md).

> [!IMPORTANT]
> `observer/isaac/` is part of the public surface even though Observer itself "never imports Isaac directly" (see [TL;DR](#tldr)). The user's record script imports from it (`from observer.isaac.recorder import VideoRecorder`) — that's exactly why it must ship as a library.

### ── Contrast with the sibling nexus repo

`jonghochoi/nexus` packages **only** `nexus.logger` and intentionally leaves `post_upload/`, `scheduled_sync/`, `chart_settings/` outside the wheel — those are operator-invoked tools that run on a GPU server or MLflow host, not inside another Python process. Observer is the opposite: every component is a candidate for a downstream `import`, so the entire tree is shipped.

### ── Decision rule for new modules

- **Default to packaging** — put new code under `observer/<subpkg>/` with an `__init__.py`. Adapter authors may want to reuse it (see [`21_ADAPTER_GUIDE.md`](./21_ADAPTER_GUIDE.md)).
- **Exception** — only an internal build/release script that **must not** be importable belongs outside the package, and such scripts are very rare in this repo. When in doubt, package it.

---

## Output directory structure

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

## Dependencies

### ── Core (required)

```bash
pip install numpy pyyaml matplotlib
sudo apt install ffmpeg          # video encoding
pip install -e .                 # install the observer CLI
```

### ── Optional

```bash
# experiment tracking
pip install tensorboard

# tactile overlay
pip install opencv-python
```

### ── Isaac (provided by Isaac Lab installation)

```
omni.isaac.lab
omni.replicator.core
omni.kit.viewport.utility
```

> ⚠️ When optional packages are absent, the corresponding feature is gracefully disabled. The rest of the pipeline continues normally.

### ── Verify installation

```bash
observer doctor   # validates config and dependencies
```

---

## Next steps

| Document | Content |
|:---|:---|
| [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) | Contract your eval / record scripts must satisfy |
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Collected metrics + failure mode taxonomy |
| [`31_CHECKPOINT_RANKING.md`](./31_CHECKPOINT_RANKING.md) | Multi-objective ranking + state coverage |
