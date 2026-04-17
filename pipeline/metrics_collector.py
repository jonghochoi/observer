"""
metrics_collector.py
====================
Collects per-step performance metrics inside the Isaac Lab inference loop.

Tracked metrics:
  - success_rate          : Episode success rate
  - contact_force_rms     : Fingertip RMS contact force (grasp stability indicator)
  - joint_velocity_rms    : Joint velocity RMS (jerk proxy)
  - slip_events           : Tactile-based slip event count
  - episode_length        : Mean episode length (steps)
  - object_pose_error     : Goal pose deviation (mm / deg)
  - energy                : Joint torque x velocity integral (energy consumption)
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("metrics_collector")


@dataclass
class EpisodeStats:
    """Per-episode statistics record."""
    success: bool = False
    length: int = 0
    contact_forces: list[float] = field(default_factory=list)    # per-step RMS
    joint_velocities: list[float] = field(default_factory=list)  # per-step RMS
    slip_count: int = 0
    final_pos_error_m: float = float("nan")
    final_rot_error_deg: float = float("nan")
    energy_J: float = 0.0

    # Initial object pose fields (used by StateCoverageAnalyzer)
    init_roll_deg: float = 0.0
    init_pitch_deg: float = 0.0
    init_yaw_deg: float = 0.0
    init_pos_x: float = 0.0
    init_pos_y: float = 0.0
    init_pos_z: float = 0.0

    # Failure mode label (set by FailureModeClassifier inside aggregate())
    failure_mode: str = "unknown"


@dataclass
class EvalResult:
    """Aggregated evaluation result across all episodes for one checkpoint."""
    checkpoint: str = ""
    num_episodes: int = 0

    # Success rate
    success_rate: float = 0.0

    # Contact force (N) — lower and more stable is better
    contact_force_rms_mean: float = 0.0
    contact_force_rms_std: float = 0.0

    # Joint velocity (rad/s) — jerk proxy
    joint_velocity_rms_mean: float = 0.0
    joint_velocity_rms_std: float = 0.0

    # Slip events per episode
    slip_events_per_episode: float = 0.0

    # Episode length (steps)
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0

    # Goal pose deviation
    object_pos_error_mm_mean: float = 0.0
    object_rot_error_deg_mean: float = 0.0

    # Energy consumption (J / episode)
    energy_J_mean: float = 0.0
    energy_J_std: float = 0.0

    # Failure mode distribution (from FailureModeClassifier)
    failure_distribution: dict = field(default_factory=dict)
    dominant_failure_mode: str = "unknown"

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Metrics saved: {path}")

    def summary_str(self) -> str:
        lines = [
            f"  Checkpoint    : {self.checkpoint}",
            f"  Episodes      : {self.num_episodes}",
            f"  Success rate  : {self.success_rate*100:.1f}%",
            f"  Contact RMS   : {self.contact_force_rms_mean:.4f} +/- {self.contact_force_rms_std:.4f} N",
            f"  Joint vel RMS : {self.joint_velocity_rms_mean:.4f} +/- {self.joint_velocity_rms_std:.4f} rad/s",
            f"  Slip events   : {self.slip_events_per_episode:.2f} / ep",
            f"  Episode length: {self.mean_episode_length:.1f} +/- {self.std_episode_length:.1f} steps",
            f"  Pos error     : {self.object_pos_error_mm_mean:.2f} mm",
            f"  Rot error     : {self.object_rot_error_deg_mean:.2f} deg",
            f"  Energy        : {self.energy_J_mean:.3f} +/- {self.energy_J_std:.3f} J",
            f"  Dominant fail : {self.dominant_failure_mode}",
        ]
        return "\n".join(lines)


class MetricsCollector:
    """
    Integrates with the Isaac Lab inference loop to collect per-step data.

    Usage pattern (inside Isaac eval script):
        collector = MetricsCollector(config.metrics)
        collector.on_episode_start()
        for step in range(max_steps):
            obs, reward, done, info = env.step(action)
            collector.on_step(info=info)
            if done.any():
                collector.on_episode_end(success=info["success"])
        result = collector.aggregate(checkpoint_name="model_5000")
        result.to_json("metrics.json")
    """

    def __init__(self, config=None):
        self.config = config
        self._episodes: list[EpisodeStats] = []
        self._current: Optional[EpisodeStats] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def on_episode_start(self, init_pose: Optional[dict] = None):
        """
        Parameters
        ----------
        init_pose : dict, optional
            Initial object pose. Expected keys:
            {"roll": 30.0, "pitch": 15.0, "yaw": -45.0,
             "pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.05}
        """
        ep = EpisodeStats()
        if init_pose:
            ep.init_roll_deg  = float(init_pose.get("roll",  0.0))
            ep.init_pitch_deg = float(init_pose.get("pitch", 0.0))
            ep.init_yaw_deg   = float(init_pose.get("yaw",   0.0))
            ep.init_pos_x     = float(init_pose.get("pos_x", 0.0))
            ep.init_pos_y     = float(init_pose.get("pos_y", 0.0))
            ep.init_pos_z     = float(init_pose.get("pos_z", 0.0))
        self._current = ep

    def on_step(
        self,
        obs: Optional["np.ndarray"] = None,
        info: Optional[dict] = None,
        done: Optional["np.ndarray"] = None,
        joint_torques: Optional["np.ndarray"] = None,
        joint_velocities: Optional["np.ndarray"] = None,
        contact_forces: Optional["np.ndarray"] = None,
        dt: float = 1.0 / 60.0,
    ):
        """
        Record data from a single environment step.

        Parameters
        ----------
        obs : observation tensor (batch_size, obs_dim)
        info : environment info dict. Expected keys:
                 "fingertip_forces"  : (n_envs, n_fingers, 3)
                 "slip_detected"     : (n_envs,) bool
                 "joint_velocities"  : (n_envs, n_joints)
                 "joint_torques"     : (n_envs, n_joints)
        done : (n_envs,) bool
        joint_torques    : explicit override over info["joint_torques"]
        joint_velocities : explicit override over info["joint_velocities"]
        contact_forces   : (n_envs, n_fingers, 3) explicit override
        dt : simulation timestep (seconds)
        """
        if self._current is None:
            return

        self._current.length += 1

        # Contact force RMS
        _cf = contact_forces if contact_forces is not None else (
            info.get("fingertip_forces") if info else None
        )
        if _cf is not None:
            cf = np.asarray(_cf)
            self._current.contact_forces.append(float(np.sqrt(np.mean(cf ** 2))))

        # Joint velocity RMS
        _jv = joint_velocities if joint_velocities is not None else (
            info.get("joint_velocities") if info else None
        )
        if _jv is not None:
            jv = np.asarray(_jv)
            self._current.joint_velocities.append(float(np.sqrt(np.mean(jv ** 2))))

        # Slip events
        if info and "slip_detected" in info:
            self._current.slip_count += int(np.asarray(info["slip_detected"]).any())

        # Energy: tau * omega * dt
        _jt = joint_torques if joint_torques is not None else (
            info.get("joint_torques") if info else None
        )
        if _jt is not None and _jv is not None:
            power = float(np.abs(np.asarray(_jt) * np.asarray(_jv)).sum())
            self._current.energy_J += power * dt

    def on_episode_end(
        self,
        success: bool = False,
        final_pos_error_m: float = float("nan"),
        final_rot_error_deg: float = float("nan"),
        info: Optional[dict] = None,
    ):
        if self._current is None:
            return

        ep = self._current
        ep.success = success
        ep.final_pos_error_m = final_pos_error_m
        ep.final_rot_error_deg = final_rot_error_deg

        if info:
            if not success and "success" in info:
                ep.success = bool(info["success"])
            if math.isnan(ep.final_pos_error_m) and "pos_error" in info:
                ep.final_pos_error_m = float(info["pos_error"])
            if math.isnan(ep.final_rot_error_deg) and "rot_error" in info:
                ep.final_rot_error_deg = float(info["rot_error"])

        self._episodes.append(ep)
        self._current = None
        log.debug(
            f"  ep #{len(self._episodes):03d} | "
            f"success={ep.success} | len={ep.length} | slip={ep.slip_count}"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, checkpoint_name: str = "") -> EvalResult:
        """Aggregate collected episodes into an EvalResult."""
        eps = self._episodes
        n = len(eps)

        if n == 0:
            log.warning("No episodes to aggregate.")
            return EvalResult(checkpoint=checkpoint_name)

        def _rms_mean(series_list):
            ep_means = [
                float(np.mean(s)) if s else float("nan")
                for s in series_list
            ]
            valid = [v for v in ep_means if not math.isnan(v)]
            if not valid:
                return 0.0, 0.0
            return float(np.mean(valid)), float(np.std(valid))

        cf_mean, cf_std = _rms_mean([ep.contact_forces for ep in eps])
        jv_mean, jv_std = _rms_mean([ep.joint_velocities for ep in eps])

        lengths  = [ep.length for ep in eps]
        pos_errs = [ep.final_pos_error_m * 1000 for ep in eps
                    if not math.isnan(ep.final_pos_error_m)]
        rot_errs = [ep.final_rot_error_deg for ep in eps
                    if not math.isnan(ep.final_rot_error_deg)]
        energies = [ep.energy_J for ep in eps]

        result = EvalResult(
            checkpoint=checkpoint_name,
            num_episodes=n,
            success_rate=float(np.mean([ep.success for ep in eps])),
            contact_force_rms_mean=cf_mean,
            contact_force_rms_std=cf_std,
            joint_velocity_rms_mean=jv_mean,
            joint_velocity_rms_std=jv_std,
            slip_events_per_episode=float(np.mean([ep.slip_count for ep in eps])),
            mean_episode_length=float(np.mean(lengths)),
            std_episode_length=float(np.std(lengths)),
            object_pos_error_mm_mean=float(np.mean(pos_errs)) if pos_errs else 0.0,
            object_rot_error_deg_mean=float(np.mean(rot_errs)) if rot_errs else 0.0,
            energy_J_mean=float(np.mean(energies)),
            energy_J_std=float(np.std(energies)),
        )

        # Auto-classify failure modes
        try:
            from observer.pipeline.failure_classifier import FailureModeClassifier, analyze_episodes
            clf = FailureModeClassifier()
            for ep in eps:
                ep.failure_mode = clf.classify(ep)
            analysis = analyze_episodes(eps, checkpoint_name)
            result.failure_distribution = analysis.mode_fractions
            result.dominant_failure_mode = analysis.dominant_failure
        except Exception as e:
            log.warning(f"Failure classification skipped: {e}")

        log.info(f"Aggregation complete:\n{result.summary_str()}")
        return result

    def reset(self):
        """Clear all collected data (call before evaluating the next checkpoint)."""
        self._episodes = []
        self._current = None
