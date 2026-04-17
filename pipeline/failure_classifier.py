"""
failure_classifier.py
======================
Automatically classifies failure causes by analyzing episode trajectories.

Classification taxonomy:
  success          -> Episode succeeded
  early_drop       -> Grasp failure at episode start (< early_drop_threshold steps)
  singularity_hit  -> Joint velocity spike before failure (near-singularity / control instability)
  late_slip        -> Gradual slip accumulation during sustained grasp
  contact_loss     -> Sudden loss of fingertip contact force
  repose_failure   -> Grasp maintained but target pose not reached
  timeout          -> Maximum step budget exhausted

Implemented as a rule-based classifier with no Isaac dependency.
Interface is designed for drop-in replacement with a learned classifier
once sufficient episode data has been collected.
"""

from __future__ import annotations
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("observer.pipeline.failure_classifier")


# ── Failure mode label map ────────────────────────────────────────────
FAILURE_MODES = {
    "success":         "Success",
    "early_drop":      "Early Drop (grasp failure)",
    "singularity_hit": "Singularity Hit",
    "late_slip":       "Late Slip Accumulation",
    "contact_loss":    "Fingertip Contact Loss",
    "repose_failure":  "Repose Goal Not Reached",
    "timeout":         "Timeout",
    "unknown":         "Unclassified",
}


@dataclass
class ClassifierThresholds:
    """
    Classification threshold configuration.
    All values can be overridden via eval_config.yaml.
    """
    early_drop_steps: int    = 50     # Failure before this step count -> early_drop
    singularity_jv: float    = 5.0    # Max joint velocity threshold (rad/s)
    late_slip_count: int     = 3      # Accumulated slip event threshold
    contact_loss_cf: float   = 0.01   # Mean fingertip force threshold in tail window (N)
    contact_loss_window: int = 10     # Number of final steps used for contact-loss check
    repose_pos_err_m: float  = 0.02   # Position error threshold (m)
    repose_rot_err_deg: float = 15.0  # Rotation error threshold (deg)


@runtime_checkable
class EpisodeStatsProtocol(Protocol):
    """Duck-type protocol compatible with MetricsCollector.EpisodeStats."""
    success: bool
    length: int
    contact_forces: list[float]
    joint_velocities: list[float]
    slip_count: int
    final_pos_error_m: float
    final_rot_error_deg: float


class FailureModeClassifier:
    """
    Maps a single episode to a failure mode label.

    Rules are applied in priority order:
      1. success
      2. early_drop       (too short to evaluate other conditions)
      3. singularity_hit  (highest hardware risk — checked before slip)
      4. late_slip
      5. contact_loss
      6. repose_failure
      7. timeout          (catch-all)
      8. unknown          (fallback)
    """

    def __init__(self, thresholds: ClassifierThresholds | None = None):
        self.th = thresholds or ClassifierThresholds()

    def classify(self, ep: EpisodeStatsProtocol) -> str:
        if ep.success:
            return "success"

        # Rule 1: early_drop
        if ep.length < self.th.early_drop_steps:
            return "early_drop"

        jv = np.array(ep.joint_velocities) if ep.joint_velocities else np.array([])
        cf = np.array(ep.contact_forces)   if ep.contact_forces   else np.array([])

        # Rule 2: singularity_hit
        if len(jv) > 0 and jv.max() > self.th.singularity_jv:
            return "singularity_hit"

        # Rule 3: late_slip
        if ep.slip_count >= self.th.late_slip_count:
            return "late_slip"

        # Rule 4: contact_loss (based on tail-window mean force)
        if len(cf) >= self.th.contact_loss_window:
            if cf[-self.th.contact_loss_window:].mean() < self.th.contact_loss_cf:
                return "contact_loss"

        # Rule 5: repose_failure
        pos_fail = (
            not math.isnan(ep.final_pos_error_m)
            and ep.final_pos_error_m > self.th.repose_pos_err_m
        )
        rot_fail = (
            not math.isnan(ep.final_rot_error_deg)
            and ep.final_rot_error_deg > self.th.repose_rot_err_deg
        )
        if pos_fail or rot_fail:
            return "repose_failure"

        # Rule 6: timeout (catch-all when no other rule matches)
        return "timeout"

    def classify_batch(self, episodes: list) -> list[str]:
        return [self.classify(ep) for ep in episodes]


# ── Aggregation ───────────────────────────────────────────────────────

@dataclass
class FailureAnalysis:
    """Per-checkpoint failure analysis result."""
    checkpoint: str
    total_episodes: int
    mode_counts: dict[str, int]        = field(default_factory=dict)
    mode_fractions: dict[str, float]   = field(default_factory=dict)
    dominant_failure: str              = "unknown"
    # Mean episode length per mode (sanity-check: early_drop should be short)
    mode_mean_length: dict[str, float] = field(default_factory=dict)
    mode_mean_slip: dict[str, float]   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint":       self.checkpoint,
            "total_episodes":   self.total_episodes,
            "mode_counts":      self.mode_counts,
            "mode_fractions":   self.mode_fractions,
            "dominant_failure": self.dominant_failure,
            "mode_mean_length": self.mode_mean_length,
            "mode_mean_slip":   self.mode_mean_slip,
        }

    def summary_str(self) -> str:
        lines = [f"  Failure analysis [{self.checkpoint}]"]
        for mode, frac in sorted(self.mode_fractions.items(), key=lambda x: -x[1]):
            label = FAILURE_MODES.get(mode, mode)
            bar = "█" * int(frac * 20)
            lines.append(f"    {label:35s} {frac*100:5.1f}% {bar}")
        lines.append(
            f"  -> Dominant failure: {FAILURE_MODES.get(self.dominant_failure, '?')}"
        )
        return "\n".join(lines)


def analyze_episodes(
    episodes: list,
    checkpoint: str = "",
    thresholds: ClassifierThresholds | None = None,
) -> FailureAnalysis:
    """
    Classify a list of episodes and return a FailureAnalysis summary.

    Parameters
    ----------
    episodes : list[EpisodeStats]
        Episode records from MetricsCollector._episodes.
    checkpoint : str
        Checkpoint name used for display in reports.
    thresholds : ClassifierThresholds | None
        Custom thresholds; falls back to defaults if None.
    """
    clf = FailureModeClassifier(thresholds)
    labels = clf.classify_batch(episodes)
    counts = Counter(labels)
    n = len(episodes)

    fractions = {mode: counts.get(mode, 0) / n for mode in FAILURE_MODES}

    # Dominant failure mode (excluding success)
    failure_only = {k: v for k, v in fractions.items() if k != "success" and v > 0}
    dominant = max(failure_only, key=failure_only.get) if failure_only else "unknown"

    mode_episodes: dict[str, list] = {m: [] for m in FAILURE_MODES}
    for ep, label in zip(episodes, labels):
        mode_episodes[label].append(ep)

    mode_mean_length = {
        m: float(np.mean([e.length for e in eps])) if eps else 0.0
        for m, eps in mode_episodes.items()
    }
    mode_mean_slip = {
        m: float(np.mean([e.slip_count for e in eps])) if eps else 0.0
        for m, eps in mode_episodes.items()
    }

    analysis = FailureAnalysis(
        checkpoint=checkpoint,
        total_episodes=n,
        mode_counts=dict(counts),
        mode_fractions=fractions,
        dominant_failure=dominant,
        mode_mean_length=mode_mean_length,
        mode_mean_slip=mode_mean_slip,
    )
    log.info(f"\n{analysis.summary_str()}")
    return analysis
