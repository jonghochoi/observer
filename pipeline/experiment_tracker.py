"""
observer/pipeline/experiment_tracker.py
========================================
TensorBoard integration module for logging eval results.

Design principles:
  - Auto-detects TensorBoard and silently no-ops when unavailable.
  - Uses the training step number extracted from the checkpoint filename as the
    shared x-axis, so eval points overlay directly on training curves.
  - Videos are stored locally; their path is logged rather than uploaded.

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
_TB_AVAILABLE = False

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

if not _TB_AVAILABLE:
    log.warning(
        "TensorBoard is not installed — experiment tracking disabled.\n"
        "  Install: pip install tensorboard"
    )


from observer.pipeline.utils import extract_step as _extract_step


class ExperimentTracker:
    """
    Records eval results to TensorBoard.

    Parameters
    ----------
    tb_log_dir : str | Path
        TensorBoard log directory.
    enabled : bool
        Master switch. Set False to disable all tracking without
        changing call sites.
    """

    def __init__(
        self,
        tb_log_dir: str | Path = "tb_logs",
        enabled: bool = True,
    ):
        self.enabled = enabled and _TB_AVAILABLE
        self._tb_writer = None

        if not self.enabled:
            return

        try:
            self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            log.info(f"TensorBoard log dir: {tb_log_dir}")
        except Exception as e:
            log.warning(f"TensorBoard init failed: {e}")
            self._tb_writer = None
            self.enabled = False

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
            f"backend={'tb' if self._tb_writer else 'none'}"
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
        """Upload a coverage/heatmap image to TensorBoard."""
        if not self.enabled or not self._tb_writer:
            return
        if not image_path.exists():
            return
        try:
            import numpy as np
            import matplotlib.image as mpimg
            img = mpimg.imread(str(image_path))  # (H, W, C) float32
            if img.ndim == 2:
                img = img[np.newaxis, :]          # (1, H, W) grayscale
            else:
                img = img[:, :, :3].transpose(2, 0, 1)  # (C, H, W), drop alpha
            self._tb_writer.add_image(key, img, global_step=step)
        except Exception as e:
            log.warning(f"Image log failed ({key}): {e}")

    def log_video(self, key: str, video_path: Path, step: int, fps: int = 30):
        """Log the path of a recorded video — TensorBoard does not support file-based video upload."""
        if not video_path.exists():
            return
        log.info(f"  [tracker] video saved locally: {video_path}  (key={key}, step={step})")

    # ── Internal write ────────────────────────────────────────────────
    def _write(self, data: dict, step: int):
        if self._tb_writer:
            for key, value in data.items():
                try:
                    if isinstance(value, (int, float)):
                        self._tb_writer.add_scalar(key, value, global_step=step)
                except Exception as e:
                    log.debug(f"TB scalar failed ({key}): {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────
    def finish(self):
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
