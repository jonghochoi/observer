# 🔌 External Logger Handoff

How a downstream training repo collects observer's outputs and forwards them to its own experiment-tracking system, **without observer knowing or caring which one**.

> 📖 The subprocess CLI contract that observer expects from your eval/record scripts is in [`20_INTEGRATION_CONTRACT.md`](20_INTEGRATION_CONTRACT.md). This guide is its mirror image — the contract observer offers to **consumers** of its outputs.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [What observer produces](#what-observer-produces)
- [Library API for consumers](#library-api-for-consumers)
- [Locating outputs](#locating-outputs)
- [Flattening metrics for a scalar logger](#flattening-metrics-for-a-scalar-logger)
- [Reading additional outputs](#reading-additional-outputs)
- [Recommended consumption pattern](#recommended-consumption-pattern)
- [Failure modes](#failure-modes)

---

## Why this exists

Observer's `PipelineOrchestrator` does not push results to MLflow, W&B, or any other centralised tracker. The only optional integration is `observer/pipeline/experiment_tracker.py`, which writes to a TensorBoard log dir or W&B run that **observer itself owns** — it cannot inject artifacts into a run that some other process opened.

When the consumer is a training repo that already has an open MLflow / W&B / custom-logger run, the right boundary is observer's **on-disk layout** plus the small `result_locator` helper. The training repo imports observer + its own logger and glues them in a few lines. Observer stays unaware of which logger is on the receiving end:

- Observer writes a deterministic directory tree.
- Observer exposes `result_locator.locate_results()` and `result_locator.read_metrics()` for consumers.
- Consumers map those into their own logger's API.

The principle: **observer never imports your logger; your logger never imports observer.** A small glue script in your training repo imports both. This way observer can be reused by any team without growing a fan-out of tracker integrations.

---

## What observer produces

After `PipelineOrchestrator.run_single(checkpoint)` returns, the on-disk tree is exactly:

```
output_root/                                 ← passed to PipelineOrchestrator(output_root=...)
├── <parent>__<stem>__<timestamp>/           ← per-checkpoint result dir
│   ├── metrics.json
│   ├── episodes.json
│   ├── eval_config_snapshot.yaml
│   ├── coverage/
│   │   ├── heatmap_roll_pitch.png
│   │   └── ...
│   └── videos/
│       ├── front.mp4
│       ├── side.mp4
│       └── combined_grid.mp4              ← only when video.concat_views=True
└── eval_report.html                         ← multi-checkpoint roll-up
```

Two location rules to remember:

- **The per-checkpoint dir** sits under `output_root/`. Its name is `f"{checkpoint.parent.name}__{checkpoint.stem}__{stamp}"` (see [`observer/pipeline/orchestrator.py:356-361`](../observer/pipeline/orchestrator.py)). One dir per `run_single()` call.
- **`eval_report.html` lives at `output_root` itself**, not inside the per-checkpoint dir. It's a multi-checkpoint roll-up rendered after a sweep — uploading it as part of a single-checkpoint result misrepresents what's inside.

When `dry_run: true` is set in the config, `metrics.json` is filled by `_dummy_metrics()` ([`observer/pipeline/orchestrator.py:383-397`](../observer/pipeline/orchestrator.py)) and the videos / Isaac subprocess are skipped — useful for end-to-end smoke-testing the consumer side without booting Isaac.

---

## Library API for consumers

```python
from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics
```

### ── Running one checkpoint

```python
config = EvalConfig.from_yaml("configs/eval_config.yaml")
orch   = PipelineOrchestrator(config=config, output_root=Path("eval_results"))
result = orch.run_single(checkpoint_path)   # → CheckpointResult
```

`CheckpointResult` ([`observer/pipeline/orchestrator.py:36-49`](../observer/pipeline/orchestrator.py)) is a dataclass:

| Field | Type | Note |
|---|---|---|
| `checkpoint` | `Path` | The input checkpoint |
| `output_dir` | `Path` | The per-checkpoint result dir; pass this verbatim to `locate_results(..., result_dir=)` |
| `metrics` | `dict \| None` | Aggregate `metrics.json` contents (also includes `failure_distribution`, `dominant_failure_mode` after step 2) |
| `video_paths` | `list[Path]` | Per-camera mp4s (empty when `skip_video=True`) |
| `combined_video` | `Path \| None` | The 2×N grid mp4, when `video.concat_views=True` |
| `coverage_plots` | `list[Path]` | The PNGs under `coverage/` |
| `success` | `bool` | **Check this** — see below |
| `error_msg` | `str` | Populated when `success=False` |
| `elapsed_sec` | `float` | Wall-clock duration |
| `failure_analysis` | `FailureAnalysis \| None` | `mode_fractions` and `dominant_failure` |

> ⚠️ `run_single()` **swallows internal exceptions** ([`observer/pipeline/orchestrator.py:103-105`](../observer/pipeline/orchestrator.py)): it catches everything, sets `result.success = False`, records the message in `result.error_msg`, and returns the partially-populated dataclass. Consumers **must** check `result.success` before reading metrics or uploading anything — observer never raises out of `run_single`. This is by design (one bad checkpoint never aborts a sweep) but trips people who expect Pythonic exception flow.

---

## Locating outputs

```python
results = locate_results(output_root, *, result_dir=None, metrics_filename="metrics.json")
```

Definition at [`observer/pipeline/result_locator.py:64`](../observer/pipeline/result_locator.py).

Two modes:

- **Default (sweep / interactive):** `result_dir=None` selects the most recently modified canonical `<a>__<b>__<c>/` subdirectory by mtime. Convenient for "show me the last result" interactive analysis.
- **Explicit (single-checkpoint runs):** `result_dir=result.output_dir` pins the bundle. **Recommended for any glue script** — the mtime default is racy under concurrent runs and surprises you when a hand-renamed dir grows back to the top of the mtime list.

Returns `ObserverResults` ([`observer/pipeline/result_locator.py:40-58`](../observer/pipeline/result_locator.py)) — all the file-path slots are populated only if the file exists, so missing pieces are reported as `None` / empty list, not as errors. Observer does not promise that every bundle is complete (a failed checkpoint may produce partial output).

| `ObserverResults` field | Populated from |
|---|---|
| `root` | The chosen per-checkpoint dir |
| `output_root` | The `output_root` argument |
| `metrics_json` | `<root>/metrics.json` |
| `episodes_json` | `<root>/episodes.json` |
| `eval_config_snapshot` | `<root>/eval_config_snapshot.yaml` |
| `report_html` | **`<output_root>/eval_report.html`** (top-level, not per-checkpoint) |
| `videos` | `<root>/videos/*.mp4` minus `combined_grid.mp4` |
| `combined_video` | `<root>/videos/combined_grid.mp4` |
| `coverage_plots` | `<root>/coverage/*.png` |

---

## Flattening metrics for a scalar logger

Most centralised loggers want `{name: float}` scalar pairs, not the nested JSON observer writes. Use:

```python
metrics = read_metrics(results)   # → dict[str, float]
```

Definition at [`observer/pipeline/result_locator.py:137`](../observer/pipeline/result_locator.py). Behaviour:

- Walks nested dicts with `.` separators — `{"failure_distribution": {"success": 0.9}}` becomes `{"failure_distribution.success": 0.9}`.
- Booleans cast to `0.0` / `1.0`.
- **Strings, lists, and `None` are silently dropped** — `dominant_failure_mode`, `checkpoint`, and `note` (in dry-run) do not appear in the output. If you want those, read `results.metrics_json` directly.
- Returns `{}` when `metrics.json` is absent. Treat that as "evaluator produced no metrics" — typically an upstream failure that the consumer should surface, not paper over.

The output keys are bare (no namespace prefix). If your downstream logger requires a prefix (e.g. `eval/`, `observer.`), apply it at the consumer side; observer deliberately does not bake one in.

Worked example. Given a `metrics.json` like:

```json
{
  "checkpoint": "model_5000.pth",
  "success_rate": 0.91,
  "failure_distribution": {
    "success": 0.91,
    "early_drop": 0.02,
    "late_slip": 0.03,
    "contact_loss": 0.02,
    "repose_failure": 0.02
  },
  "dominant_failure_mode": "success",
  "energy_J_per_episode": 2.3
}
```

`read_metrics(...)` returns:

```python
{
    "success_rate": 0.91,
    "failure_distribution.success": 0.91,
    "failure_distribution.early_drop": 0.02,
    "failure_distribution.late_slip": 0.03,
    "failure_distribution.contact_loss": 0.02,
    "failure_distribution.repose_failure": 0.02,
    "energy_J_per_episode": 2.3,
}
```

The string `"checkpoint"` and `"dominant_failure_mode"` are dropped. Record them as **tags** in your logger, not metrics.

---

## Reading additional outputs

`read_metrics()` only handles `metrics.json`. The other artefacts are reachable by path off the `ObserverResults` dataclass:

```python
results = locate_results(output_root, result_dir=result.output_dir)

# Per-camera and grid videos
for v in results.videos:
    your_logger.upload_video(v)
if results.combined_video:
    your_logger.upload_video(results.combined_video)

# Coverage heatmaps
for p in results.coverage_plots:
    your_logger.upload_image(p)

# Other files
if results.eval_config_snapshot:
    your_logger.upload_file(results.eval_config_snapshot)
if results.episodes_json:
    your_logger.upload_file(results.episodes_json)

# Multi-checkpoint roll-up — only meaningful after a sweep
if results.report_html:
    your_logger.upload_file(results.report_html)
```

Most centralised trackers can just upload a directory recursively. Observer's per-checkpoint dir (`results.root`) is self-contained and safe to upload as a single bundle — the consumer doesn't need to enumerate file types.

---

## Recommended consumption pattern

Two-process model — the cleanest separation of concerns:

1. **Training process** — opens its own logger run early, ideally writing a sidecar that records the run identity (run name, experiment, server URI) into a fixed file path. Trains. Exits.
2. **Eval process** — invoked **after** training completes, in a separate Python interpreter. Imports observer + the training repo's logger. Reads the sidecar to recover the run identity, runs `PipelineOrchestrator.run_single(...)`, then uses `result_locator` to forward outputs to the existing run.

Why two processes:

- Observer's video stage spawns the framework's `record_script` under the simulator wrapper (e.g. `isaaclab.sh`). Co-locating that with a still-running trainer risks GPU memory contention and dependency conflicts.
- The training repo's logger and observer can have independent dependency closures — no need to install observer on training nodes that don't run eval, and vice versa.
- Failure isolation — a crash in the eval pipeline cannot corrupt training state.

> 📖 An example of this pattern: in deployments using NEXUS as the centralised logger, the training repo writes a `.nexus_run.json` sidecar via `make_logger()`, and a small post-training script reads that sidecar to forward observer's outputs to the same MLflow run. See the **NEXUS** project's `docs/32_EVAL_ARTIFACT_INGESTION.md` (the consumer-side mirror of this guide). Observer itself contains no code that knows about NEXUS.

A minimal glue script lives in the training repo (not in observer, not in the logger package). Sketch:

```python
from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics

# from <your_logger> import ...   ← imported by the training repo only

def evaluate_and_upload(checkpoint, training_output_dir, observer_config_yaml):
    cfg    = EvalConfig.from_yaml(str(observer_config_yaml))
    orch   = PipelineOrchestrator(config=cfg, output_root=training_output_dir / "eval")
    result = orch.run_single(checkpoint)
    if not result.success:
        raise RuntimeError(f"observer eval failed: {result.error_msg}")

    obs     = locate_results(orch.output_root, result_dir=result.output_dir)
    metrics = read_metrics(obs)

    # Hand off to your logger — this is the only line the training repo
    # tailors to its tracking backend. Observer doesn't know what's here.
    your_logger.upload_eval_bundle(
        eval_dir=result.output_dir,
        metrics=metrics,
        tags={"checkpoint": checkpoint.name},
    )
```

---

## Failure modes

Things that surprise consumers, in observer's own words:

| When | What you see | Why |
|---|---|---|
| `result.success == False` after `run_single()` | Set `error_msg` populated; partial outputs may exist | Internal exceptions are caught — by design, so a sweep doesn't abort on one bad checkpoint. Consumers must branch on `success` |
| `read_metrics()` returns `{}` | `metrics.json` was not produced (or was unreadable) | Step 1 (metrics collection) crashed in the subprocess. Check `result.error_msg` and the subprocess's stderr |
| Wrong bundle picked up by `locate_results()` | A previously-renamed dir or a parallel `run_single()` ahead in mtime | Pass `result_dir=result.output_dir` explicitly. The mtime default is for interactive use only |
| Dotted metric keys collide on the consumer side | Centralised tracker treats `.` as a hierarchy separator differently from observer | Most loggers (MLflow, W&B) accept `.` in metric names; if yours doesn't, replace `.` with `_` at the consumer boundary |
| `failure_distribution` missing from `metrics` | Step 2 (failure analysis) couldn't find `episodes.json` | Either the eval subprocess didn't produce `episodes.json`, or `dry_run=True` skipped it. Step 2 falls back to copying `failure_distribution` from `metrics` itself if present |

For consumer-side issues (sidecar missing, run not found on the central server, etc.), consult the consumer logger's own documentation — observer has no view into that side of the handoff.
