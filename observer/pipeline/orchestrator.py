"""
observer/pipeline/orchestrator.py
==================================
Coordinates the full evaluation cycle for a single checkpoint.

Execution order (per checkpoint):
  1. Metrics collection   — headless Isaac subprocess -> metrics.json
  2. Failure analysis     — FailureModeClassifier on episode data
  3. Coverage analysis    — StateCoverageAnalyzer -> PNG plots
  4. Video recording      — Isaac GUI subprocess + camera sweep -> mp4 files
  5. Experiment tracking  — W&B / TensorBoard logging
"""

from __future__ import annotations
import json
import logging
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from observer.configs.eval_config import EvalConfig
from observer.pipeline.failure_classifier import (
    analyze_episodes,
    FailureAnalysis,
    ClassifierThresholds,
)
from observer.pipeline.state_coverage import StateCoverageAnalyzer, PosedEpisode
from observer.pipeline.experiment_tracker import ExperimentTracker
from observer.pipeline.utils import extract_step

log = logging.getLogger("observer.orchestrator")


@dataclass
class CheckpointResult:
    checkpoint: Path
    output_dir: Path
    metrics: Optional[dict] = None
    video_paths: list[Path] = field(default_factory=list)
    combined_video: Optional[Path] = None
    success: bool = False
    error_msg: str = ""
    elapsed_sec: float = 0.0
    failure_analysis: Optional[object] = None   # FailureAnalysis
    coverage_stats: Optional[object] = None     # CoverageStats
    scored_rank: int = 0
    coverage_plots: list[Path] = field(default_factory=list)


class PipelineOrchestrator:
    def __init__(
        self,
        config: EvalConfig,
        output_root: Path,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.config      = config
        self.output_root = output_root
        self.tracker     = tracker

    # ── Public entry point ────────────────────────────────────────────
    def run_single(self, checkpoint: Path) -> CheckpointResult:
        t0      = time.time()
        out_dir = self._make_output_dir(checkpoint)
        result  = CheckpointResult(checkpoint=checkpoint, output_dir=out_dir)

        try:
            self.config.to_yaml(str(out_dir / "eval_config_snapshot.yaml"))

            log.info("  [1/5] Metrics collection (headless)")
            metrics = self._run_metrics_collection(checkpoint, out_dir)
            result.metrics = metrics

            log.info("  [2/5] Failure mode analysis")
            failure_analysis = self._run_failure_analysis(metrics, out_dir)
            result.failure_analysis = failure_analysis
            if failure_analysis:
                metrics["failure_distribution"]  = failure_analysis.mode_fractions
                metrics["dominant_failure_mode"] = failure_analysis.dominant_failure

            log.info("  [3/5] State coverage analysis")
            coverage_stats, coverage_plots = self._run_coverage_analysis(metrics, out_dir)
            result.coverage_stats = coverage_stats
            result.coverage_plots = coverage_plots

            if not self.config.skip_video:
                log.info("  [4/5] Video recording (camera sweep)")
                video_paths = self._run_video_recording(checkpoint, out_dir)
                result.video_paths = video_paths
                if self.config.video.concat_views and len(video_paths) > 1:
                    log.info("       Grid concat")
                    result.combined_video = self._concat_videos(video_paths, out_dir)
            else:
                log.info("  [4/5] Video recording skipped")

            log.info("  [5/5] Experiment tracking")
            self._push_to_tracker(result)

            result.success = True

        except Exception as e:
            log.error(f"  X Failed: {e}", exc_info=True)
            result.error_msg = str(e)

        result.elapsed_sec = time.time() - t0
        log.info(f"  Done: {result.elapsed_sec:.1f}s | success={result.success}")
        return result

    # ── Step 1: metrics collection ────────────────────────────────────
    def _run_metrics_collection(self, checkpoint: Path, out_dir: Path) -> dict:
        if self.config.dry_run:
            log.info("    [dry_run] Returning dummy metrics")
            return _dummy_metrics(checkpoint.name)

        metrics_path = out_dir / self.config.metrics_filename
        episodes_path = out_dir / "episodes.json"
        cmd = self._build_metrics_cmd(
            checkpoint=checkpoint,
            metrics_path=metrics_path,
            episodes_path=episodes_path,
        )
        log.debug(f"    CMD: {' '.join(cmd)}")
        _run_subprocess(cmd, label="metrics")

        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        log.warning("    metrics.json was not created — returning empty dict")
        return {}

    # ── Step 2: failure analysis ──────────────────────────────────────
    def _run_failure_analysis(self, metrics: dict, out_dir: Path):
        try:
            episodes_path = out_dir / "episodes.json"
            thresholds = ClassifierThresholds(
                **self.config.failure_thresholds.__dict__
            )
            if episodes_path.exists():
                from observer.pipeline.metrics_collector import EpisodeStats
                with open(episodes_path) as f:
                    raw_eps = json.load(f)
                episodes = [EpisodeStats(**ep) for ep in raw_eps]
                return analyze_episodes(
                    episodes,
                    checkpoint=metrics.get("checkpoint", ""),
                    thresholds=thresholds,
                )
            else:
                dist     = metrics.get("failure_distribution", {})
                dominant = metrics.get("dominant_failure_mode", "unknown")
                if dist:
                    fa = FailureAnalysis(
                        checkpoint=metrics.get("checkpoint", ""),
                        total_episodes=int(metrics.get("num_episodes", 0)),
                        mode_fractions=dist,
                        dominant_failure=dominant,
                    )
                    log.info(f"  Failure distribution restored from aggregate: dominant={dominant}")
                    return fa
                log.info("  No episode data — failure analysis skipped")
                return None
        except Exception as e:
            log.warning(f"  Failure analysis error: {e}")
            return None

    # ── Step 3: state coverage analysis ───────────────────────────────
    def _run_coverage_analysis(self, metrics: dict, out_dir: Path):
        try:
            episodes_path = out_dir / "episodes.json"
            if not episodes_path.exists():
                log.info("  No episode data — coverage analysis skipped")
                return None, []

            with open(episodes_path) as f:
                raw_eps = json.load(f)

            posed = [
                PosedEpisode(
                    success=ep.get("success", False),
                    failure_mode=ep.get("failure_mode", "unknown"),
                    init_roll_deg=ep.get("init_roll_deg", 0.0),
                    init_pitch_deg=ep.get("init_pitch_deg", 0.0),
                    init_yaw_deg=ep.get("init_yaw_deg", 0.0),
                    init_pos_x=ep.get("init_pos_x", 0.0),
                    init_pos_y=ep.get("init_pos_y", 0.0),
                    init_pos_z=ep.get("init_pos_z", 0.0),
                    episode_length=ep.get("length", 0),
                    slip_count=ep.get("slip_count", 0),
                )
                for ep in raw_eps
            ]

            coverage_dir = out_dir / "coverage"
            stats  = StateCoverageAnalyzer(output_dir=coverage_dir).analyze(posed)
            plots  = sorted(coverage_dir.glob("*.png"))
            log.info(f"  Coverage analysis complete: {len(plots)} plot(s)")
            return stats, plots

        except Exception as e:
            log.warning(f"  Coverage analysis error: {e}")
            return None, []

    # ── Step 4: video recording ───────────────────────────────────────
    def _run_video_recording(self, checkpoint: Path, out_dir: Path) -> list[Path]:
        if self.config.dry_run:
            log.info("    [dry_run] Video recording skipped")
            return []

        self.config.runtime.validate_for_record()

        video_dir    = out_dir / self.config.video_subdir
        video_dir.mkdir(exist_ok=True)
        cam_cfg_path = out_dir / "camera_poses.json"
        _write_camera_json(self.config, cam_cfg_path)

        cmd = self._build_record_cmd(
            checkpoint=checkpoint,
            extra_args=[
                f"--num_envs={self.config.runtime.num_envs}",
                "--record_mode",
                f"--camera_config={cam_cfg_path}",
                f"--video_output_dir={video_dir}",
                f"--video_fps={self.config.video.fps}",
                f"--video_resolution="
                f"{self.config.video.resolution[0]}x{self.config.video.resolution[1]}",
            ]
        )
        log.debug(f"    CMD: {' '.join(cmd)}")
        _run_subprocess(cmd, label="recording")

        video_paths = sorted(video_dir.glob("*.mp4"))
        log.info(f"    Recording complete: {len(video_paths)} video(s)")
        return video_paths

    # ── Video grid concat (ffmpeg) ────────────────────────────────────
    def _concat_videos(self, video_paths: list[Path], out_dir: Path) -> Optional[Path]:
        if not video_paths:
            return None

        n    = len(video_paths)
        cols = 2
        rows = (n + 1) // cols
        out_path = out_dir / self.config.video_subdir / "combined_grid.mp4"

        inputs = []
        for vp in video_paths:
            inputs += ["-i", str(vp)]

        w, h = (self.config.video.resolution[0] // cols,
                self.config.video.resolution[1] // rows)
        filter_parts = [f"[{i}:v]scale={w}:{h}[v{i}]" for i in range(n)]

        padded_n = rows * cols
        if padded_n > n:
            filter_parts.append(f"color=black:s={w}x{h}:r={self.config.video.fps}[vpad]")

        row_stacks = []
        for r in range(rows):
            row_inputs = []
            for c in range(cols):
                idx = r * cols + c
                row_inputs.append(f"[v{idx}]" if idx < n else "[vpad]")
            row_label = f"[row{r}]"
            filter_parts.append(
                f"{''.join(row_inputs)}hstack=inputs={cols}{row_label}"
            )
            row_stacks.append(row_label)

        if rows > 1:
            filter_parts.append(f"{''.join(row_stacks)}vstack=inputs={rows}[out]")
            map_label = "[out]"
        else:
            map_label = row_stacks[0]

        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", map_label,
            "-c:v", self.config.video.codec,
            "-pix_fmt", self.config.video.pix_fmt,
            "-crf", str(self.config.video.crf),
            str(out_path),
        ]
        _run_subprocess(cmd, label="ffmpeg-concat")
        log.info(f"    Grid video saved: {out_path}")
        return out_path

    # ── Step 5: experiment tracking ───────────────────────────────────
    def _push_to_tracker(self, result: CheckpointResult):
        if self.tracker is None or result.metrics is None:
            return
        try:
            step = extract_step(result.checkpoint.name)
            self.tracker.log_eval_result(
                metrics=result.metrics,
                checkpoint_name=result.checkpoint.name,
                failure_analysis=result.failure_analysis,
            )
            for plot_path in result.coverage_plots:
                self.tracker.log_image(
                    f"eval/coverage/{plot_path.stem}", plot_path, step
                )
            if result.combined_video and result.combined_video.exists():
                self.tracker.log_video(
                    "eval/video/combined", result.combined_video, step
                )
        except Exception as e:
            log.warning(f"  Experiment tracking failed: {e}")

    # ── Internal helpers ──────────────────────────────────────────────
    def _build_metrics_cmd(
        self,
        checkpoint: Path,
        metrics_path: Path,
        episodes_path: Path,
    ) -> list[str]:
        """Build the metrics subprocess command.

        Observer only guarantees the CLI flag set documented in
        ``docs/INTEGRATION.md``; anything framework-specific must be supplied
        via ``runtime.extra_eval_args`` in the YAML.
        """
        rt = self.config.runtime
        rt.validate_for_metrics()
        cmd = [
            "python", "-m", rt.eval_module,
            f"--task={rt.task}",
            f"--load_path={checkpoint}",
            f"--device={rt.device}",
            f"--seed={rt.seed}",
            f"--num_envs={rt.num_envs}",
            f"--num_episodes={self.config.metrics.num_eval_episodes}",
            f"--metrics_output={metrics_path}",
            f"--episodes_output={episodes_path}",
            "--headless",
        ]
        cmd.extend(rt.extra_eval_args)
        return cmd

    def _build_record_cmd(self, checkpoint: Path, extra_args: list[str]) -> list[str]:
        """Build the video recording subprocess command."""
        rt = self.config.runtime
        cmd = [
            rt.resolve_isaac_lab_path(), "-p", rt.record_script,
            f"--task={rt.task}",
            f"--load_path={checkpoint}",
            f"--device={rt.device}",
            f"--seed={rt.seed}",
            *extra_args,
        ]
        cmd.extend(rt.extra_record_args)
        return cmd

    def _make_output_dir(self, checkpoint: Path) -> Path:
        stamp    = time.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{checkpoint.parent.name}__{checkpoint.stem}__{stamp}"
        out_dir  = self.output_root / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


# ── Helper functions ──────────────────────────────────────────────────
def _run_subprocess(cmd: list[str], label: str):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] subprocess failed (code={result.returncode})\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )


def _write_camera_json(config: EvalConfig, path: Path):
    cameras = [
        {"name": c.name, "eye": c.eye, "target": c.target, "record_steps": c.record_steps}
        for c in config.cameras
    ]
    with open(path, "w") as f:
        json.dump(cameras, f, indent=2)


def _dummy_metrics(ckpt_name: str) -> dict:
    """Dummy metrics for dry-run mode."""
    import random
    random.seed(hash(ckpt_name) % (2**31))
    return {
        "checkpoint":              ckpt_name,
        "success_rate":            round(random.uniform(0.5, 0.95), 3),
        "contact_force_rms":       round(random.uniform(0.1, 1.5), 4),
        "joint_velocity_rms":      round(random.uniform(0.05, 0.3), 4),
        "slip_events_per_episode": round(random.uniform(0.0, 3.0), 2),
        "mean_episode_length":     round(random.uniform(80, 200), 1),
        "object_pose_error_mm":    round(random.uniform(1.0, 15.0), 2),
        "energy_J_per_episode":    round(random.uniform(0.5, 5.0), 3),
        "note":                    "dry_run dummy data",
    }
