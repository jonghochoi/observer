"""
observer/pipeline/result_locator.py
====================================
Discover the on-disk layout produced by ``PipelineOrchestrator.run_single``.

This module exists so callers (e.g. a post-training "evaluate + upload"
glue script) can hand observer's outputs to a generic uploader without
re-implementing observer's layout knowledge. Observer doesn't import any
uploader; the uploader doesn't import observer. The glue script imports
both.

Layout produced by ``PipelineOrchestrator``:

    output_root/
    ├── <parent>__<stem>__<timestamp>/      ← per-checkpoint result dir
    │   ├── metrics.json
    │   ├── episodes.json
    │   ├── eval_config_snapshot.yaml
    │   ├── coverage/*.png
    │   └── videos/*.mp4 (+ combined_grid.mp4 when concat_views=True)
    └── eval_report.html                    ← multi-checkpoint roll-up

``locate_results(output_root)`` walks this tree and returns an
``ObserverResults`` dataclass. ``read_metrics(results)`` flattens the
``metrics.json`` for that result into a {dotted_key: float} dict suitable
for promotion as MLflow scalars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Public dataclass ─────────────────────────────────────────────────────────


@dataclass
class ObserverResults:
    """Materialized view of a single PipelineOrchestrator output bundle."""

    root: Path
    """The selected per-checkpoint result directory (``<parent>__<stem>__<ts>``)."""

    output_root: Path
    """The parent ``output_root=`` passed to ``PipelineOrchestrator``."""

    metrics_json: Optional[Path] = None
    episodes_json: Optional[Path] = None
    eval_config_snapshot: Optional[Path] = None
    report_html: Optional[Path] = None
    """Top-level ``eval_report.html`` (lives at ``output_root``, not ``root``)."""

    videos: list[Path] = field(default_factory=list)
    combined_video: Optional[Path] = None
    coverage_plots: list[Path] = field(default_factory=list)


# ── Public interface ─────────────────────────────────────────────────────────


def locate_results(
    output_root: Path,
    *,
    result_dir: Optional[Path] = None,
    metrics_filename: str = "metrics.json",
) -> ObserverResults:
    """Resolve observer's output layout into a populated ``ObserverResults``.

    ``output_root`` is the same directory passed to
    ``PipelineOrchestrator(output_root=...)``.

    By default the most recently modified ``<parent>__<stem>__<ts>/``
    subdirectory is selected. Pass ``result_dir`` explicitly to pin a
    specific bundle (useful when you ran multiple checkpoints in one
    session and want the second-to-last).

    Missing files are reported as ``None`` / empty list — the caller
    decides whether that's an error. Observer itself never aborts a sweep
    on a single failed checkpoint, so a partial bundle is a real case.
    """
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"output_root not found: {output_root}")
    if not output_root.is_dir():
        raise NotADirectoryError(f"output_root is not a directory: {output_root}")

    if result_dir is not None:
        chosen = Path(result_dir)
        # Accept either an absolute path, a CWD-relative full path that already
        # contains `output_root`, or a bare basename that should be joined to
        # `output_root`. The earlier blind prefix produced a doubled path when
        # callers (e.g. `run_eval_and_upload.py`) passed back the
        # `result.output_dir` they got from the orchestrator verbatim.
        if not chosen.is_absolute() and not chosen.exists():
            chosen = output_root / chosen
    else:
        chosen = _latest_result_dir(output_root)
        if chosen is None:
            raise FileNotFoundError(
                f"No PipelineOrchestrator result subdirectory found under {output_root}. "
                "Expected <parent>__<stem>__<timestamp>/ — has the orchestrator run yet?"
            )

    results = ObserverResults(root=chosen, output_root=output_root)

    metrics_path = chosen / metrics_filename
    if metrics_path.exists():
        results.metrics_json = metrics_path

    for name, attr in (
        ("episodes.json", "episodes_json"),
        ("eval_config_snapshot.yaml", "eval_config_snapshot"),
    ):
        p = chosen / name
        if p.exists():
            setattr(results, attr, p)

    # Top-level report sits at output_root, not inside the per-checkpoint dir.
    report = output_root / "eval_report.html"
    if report.exists():
        results.report_html = report

    coverage_dir = chosen / "coverage"
    if coverage_dir.is_dir():
        results.coverage_plots = sorted(coverage_dir.glob("*.png"))

    video_dir = chosen / "videos"
    if video_dir.is_dir():
        all_videos = sorted(video_dir.glob("*.mp4"))
        combined = video_dir / "combined_grid.mp4"
        results.combined_video = combined if combined.exists() else None
        # combined_grid.mp4 is a derived artifact; expose per-camera videos
        # separately so callers can decide whether to upload the grid only.
        results.videos = [v for v in all_videos if v != combined]

    return results


def read_metrics(results: ObserverResults) -> dict:
    """Flatten ``metrics.json`` into a ``{dotted_key: float}`` dict.

    Walks nested dicts with '.' separators. Non-numeric leaves (strings
    like ``checkpoint``, ``dominant_failure_mode``) and lists are skipped
    silently — they aren't meaningful as MLflow scalars. Returns an empty
    dict if no ``metrics.json`` was found.
    """
    if results.metrics_json is None or not results.metrics_json.exists():
        return {}

    with open(results.metrics_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}

    out: dict[str, float] = {}

    def walk(obj: dict, prefix: str) -> None:
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, (int, float)):
                out[key] = float(v)
            # Strings, lists, None — silently skipped.

    walk(data, "")
    return out


# ── Internal helpers ─────────────────────────────────────────────────────────


def _latest_result_dir(output_root: Path) -> Optional[Path]:
    """Return the most recently modified ``<parent>__<stem>__<ts>/`` subdir.

    Selection is by mtime rather than parsing the trailing timestamp so we
    survive clock skew or hand-renamed directories. Subdirs with the
    canonical ``__`` shape are preferred; if none match, fall back to any
    direct subdirectory so single-checkpoint test setups still work.
    """
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    canonical = [p for p in candidates if p.name.count("__") >= 2]
    pool = canonical or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)
