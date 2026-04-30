"""
experiment_tracker.py
======================
W&B / TensorBoard integration module.

Design principles:
  - Auto-detects whichever backend(s) are installed and uses them simultaneously.
  - Silently no-ops when neither backend is available.
  - Uses the training step number extracted from the checkpoint filename as the
    shared x-axis, so eval points overlay directly on training curves.

Logged metrics:
  eval/success_rate, eval/contact_force_rms, eval/slip_per_episode,
  eval/pos_error_mm, eval/energy_J, eval/dominant_failure_mode,
  eval/failure/<mode> (per-mode fraction), eval/coverage/* (images)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("observer.pipeline.experiment_tracker")

# ── Backend detection ─────────────────────────────────────────────────
_WANDB_AVAILABLE = False
_TB_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
    log.info("W&B detected")
except ImportError:
    pass

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
    log.info("TensorBoard detected")
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        _TB_AVAILABLE = True
        log.info("TensorBoardX detected")
    except ImportError:
        pass

if not _WANDB_AVAILABLE and not _TB_AVAILABLE:
    log.warning(
        "Neither W&B nor TensorBoard is installed — experiment tracking disabled.\n"
        "  Install: pip install wandb   or   pip install tensorboard"
    )


from observer.pipeline.utils import extract_step as _extract_step


class ExperimentTracker:
    """
    Records eval results to W&B and/or TensorBoard simultaneously.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str | None
        W&B run name. Auto-generated if None.
    tb_log_dir : str | Path
        TensorBoard log directory.
    tags : list[str]
        W&B run tags.
    config : dict
        Experiment hyperparameters logged to W&B config.
    enabled : bool
        Master switch. Set False to disable all tracking without
        changing call sites.
    """

    def __init__(
        self,
        project: str = "sharpa-hand-eval",
        run_name: Optional[str] = None,
        tb_log_dir: str | Path = "tb_logs",
        tags: Optional[list[str]] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and (_WANDB_AVAILABLE or _TB_AVAILABLE)
        self._wandb_run = None
        self._tb_writer = None

        if not self.enabled:
            return

        # W&B init
        if _WANDB_AVAILABLE:
            try:
                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    tags=tags or [],
                    config=config or {},
                    resume="allow",
                )
                log.info(f"W&B run started: {self._wandb_run.url}")
            except Exception as e:
                log.warning(f"W&B init failed: {e}")
                self._wandb_run = None

        # TensorBoard init
        if _TB_AVAILABLE:
            try:
                self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
                log.info(f"TensorBoard log dir: {tb_log_dir}")
            except Exception as e:
                log.warning(f"TensorBoard init failed: {e}")
                self._tb_writer = None

    # ── Logging ───────────────────────────────────────────────────────
    def log_eval_result(
        self,
        metrics: dict,
        checkpoint_name: str,
        failure_analysis=None,
        step: Optional[int] = None,
    ):
        """
        Log a single checkpoint's evaluation result.

        Parameters
        ----------
        metrics : dict
            EvalResult.to_dict() or contents of metrics.json.
        checkpoint_name : str
            Used to extract step number if `step` is not provided.
        failure_analysis : FailureAnalysis | None
        step : int | None
            Explicit global step. Extracted from checkpoint_name if None.
        """
        if not self.enabled:
            return

        global_step = step if step is not None else _extract_step(checkpoint_name)

        log_data = {
            "eval/success_rate":      metrics.get("success_rate", 0),
            "eval/contact_force_rms": metrics.get("contact_force_rms_mean",
                                      metrics.get("contact_force_rms", 0)),
            "eval/joint_vel_rms":     metrics.get("joint_velocity_rms_mean",
                                      metrics.get("joint_velocity_rms", 0)),
            "eval/slip_per_episode":  metrics.get("slip_events_per_episode", 0),
            "eval/pos_error_mm":      metrics.get("object_pos_error_mm_mean",
                                      metrics.get("object_pose_error_mm", 0)),
            "eval/rot_error_deg":     metrics.get("object_rot_error_deg_mean", 0),
            "eval/energy_J":          metrics.get("energy_J_mean",
                                      metrics.get("energy_J_per_episode", 0)),
            "eval/episode_length":    metrics.get("mean_episode_length", 0),
        }

        if failure_analysis is not None:
            log_data["eval/dominant_failure"] = _mode_to_int(
                failure_analysis.dominant_failure
            )
            for mode, frac in failure_analysis.mode_fractions.items():
                log_data[f"eval/failure/{mode}"] = frac

        self._write(log_data, global_step)
        log.info(
            f"  [tracker] step={global_step} | "
            f"success={log_data['eval/success_rate']:.3f} | "
            f"backends={'wandb ' if self._wandb_run else ''}"
            f"{'tb' if self._tb_writer else ''}"
        )

    def log_coverage_stats(self, stats, step: int):
        """Log StateCoverageAnalyzer summary stats."""
        if not self.enabled:
            return
        self._write({
            "eval/coverage_uniformity": stats.coverage_uniformity,
            "eval/n_high_risk_zones":   len(stats.high_risk_zones),
        }, step)

    def log_image(self, key: str, image_path: Path, step: int):
        """Upload a coverage/heatmap image to W&B."""
        if not self.enabled or not self._wandb_run:
            return
        if not image_path.exists():
            return
        try:
            wandb.log({key: wandb.Image(str(image_path))}, step=step)
        except Exception as e:
            log.warning(f"Image upload failed ({key}): {e}")

    def log_video(self, key: str, video_path: Path, step: int, fps: int = 30):
        """Upload a recorded video to W&B."""
        if not self.enabled or not self._wandb_run:
            return
        if not video_path.exists():
            return
        try:
            wandb.log({key: wandb.Video(str(video_path), fps=fps)}, step=step)
            log.info(f"  [W&B] video uploaded: {key}")
        except Exception as e:
            log.warning(f"Video upload failed ({key}): {e}")

    # ── Internal write ────────────────────────────────────────────────
    def _write(self, data: dict, step: int):
        if self._wandb_run:
            try:
                wandb.log(data, step=step)
            except Exception as e:
                log.warning(f"W&B log failed: {e}")
        if self._tb_writer:
            for key, value in data.items():
                try:
                    if isinstance(value, (int, float)):
                        self._tb_writer.add_scalar(key, value, global_step=step)
                except Exception as e:
                    log.debug(f"TB scalar failed ({key}): {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────
    def finish(self):
        if self._wandb_run:
            wandb.finish()
        if self._tb_writer:
            self._tb_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()


# ── Utility ───────────────────────────────────────────────────────────
_MODE_INT = {
    "success": 0, "early_drop": 1, "singularity_hit": 2,
    "late_slip": 3, "contact_loss": 4, "repose_failure": 5,
    "timeout": 6, "unknown": 7,
}

def _mode_to_int(mode: str) -> int:
    """Convert failure mode string to integer for TensorBoard scalar logging."""
    return _MODE_INT.get(mode, 7)
