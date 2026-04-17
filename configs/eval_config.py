"""
observer/configs/eval_config.py
================================
Central configuration dataclass for the evaluation pipeline.
Loaded from eval_config.yaml; all framework-specific fields are required so
observer stays agnostic to the underlying RL stack.

See ``docs/INTEGRATION.md`` for the contract observer expects from the
metrics/record scripts it launches, and ``docs/adapters/`` for ready-made
configs pointing at specific frameworks (e.g. sharpa-rl-lab).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class CameraConfig:
    """Single viewpoint definition."""
    name: str
    eye: list[float]          # Camera position [x, y, z] (world frame, meters)
    target: list[float]       # Look-at point  [x, y, z]
    record_steps: int = 200   # Simulation steps to record at this viewpoint


@dataclass
class VideoConfig:
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 18
    concat_views: bool = True


@dataclass
class MetricsConfig:
    """Controls which metrics are collected during evaluation."""
    collect_success_rate: bool = True
    collect_contact_force: bool = True
    collect_joint_velocity: bool = True
    collect_slip_events: bool = True
    collect_episode_length: bool = True
    collect_object_pose_error: bool = True
    collect_energy: bool = True
    num_eval_episodes: int = 50


@dataclass
class FailureThresholdsConfig:
    """Thresholds forwarded to failure_classifier.ClassifierThresholds."""
    early_drop_steps: int = 50
    singularity_jv: float = 5.0
    late_slip_count: int = 3
    contact_loss_cf: float = 0.01
    contact_loss_window: int = 10
    repose_pos_err_m: float = 0.02
    repose_rot_err_deg: float = 15.0


@dataclass
class RuntimeConfig:
    """Framework-agnostic runtime parameters for the metrics/record subprocesses.

    Observer only knows it needs to launch *something* that satisfies the
    integration contract (see ``docs/INTEGRATION.md``). Pick values that make
    sense for your stack and wire them in through the YAML.

    Required:
      - ``task``         : identifier passed to the eval/record scripts.
      - ``eval_module``  : module invoked as ``python -m <eval_module> ...``
                           for headless metrics collection.
      - ``record_script``: script launched under the Isaac Lab launcher for
                           GUI-based video recording (used only when video is
                           enabled). Pass an empty string to hard-disable.

    Optional:
      - ``extra_eval_args``/``extra_record_args`` : forwarded verbatim to each
        subprocess (e.g. framework-specific flags like ``--cache=...``).
    """
    task: str = ""
    num_envs: int = 4
    headless: bool = False
    device: str = "cuda:0"
    seed: int = 42

    eval_module: str = ""
    record_script: str = ""

    # Isaac Lab launcher, used only for record_script. Env vars are expanded lazily.
    isaac_lab_path: str = "${ISAACLAB_PATH}/isaaclab.sh"

    extra_eval_args: list[str] = field(default_factory=list)
    extra_record_args: list[str] = field(default_factory=list)

    def resolve_isaac_lab_path(self) -> str:
        return os.path.expandvars(self.isaac_lab_path)

    def validate_for_metrics(self) -> None:
        missing = [n for n, v in (("task", self.task), ("eval_module", self.eval_module)) if not v]
        if missing:
            raise ValueError(
                f"runtime.{missing[0]} is required — set it in eval_config.yaml. "
                f"See observer/docs/INTEGRATION.md for the expected contract."
            )

    def validate_for_record(self) -> None:
        missing = [n for n, v in (("task", self.task), ("record_script", self.record_script)) if not v]
        if missing:
            raise ValueError(
                f"runtime.{missing[0]} is required when video recording is enabled. "
                f"Either set it or run with skip_video: true."
            )


@dataclass
class EvalConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    failure_thresholds: FailureThresholdsConfig = field(default_factory=FailureThresholdsConfig)
    cameras: list[CameraConfig] = field(default_factory=list)

    skip_video: bool = False
    skip_report: bool = False
    dry_run: bool = False

    save_raw_observations: bool = False
    frame_subdir: str = "frames"
    video_subdir: str = "videos"
    metrics_filename: str = "metrics.json"

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        p = Path(path)
        if not p.exists():
            import logging
            logging.getLogger("observer.configs.eval_config").warning(
                f"Config file '{path}' not found. Using defaults."
            )
            cfg = cls()
            cfg._set_default_cameras()
            return cfg

        with open(p) as f:
            raw = yaml.safe_load(f)

        cfg = cls()

        # runtime block; legacy `isaac:` key is still accepted as an alias
        runtime_raw = raw.get("runtime") or raw.get("isaac")
        if runtime_raw:
            cfg.runtime = RuntimeConfig(**runtime_raw)
        if video_raw := raw.get("video"):
            vr = video_raw.copy()
            if "resolution" in vr:
                vr["resolution"] = tuple(vr["resolution"])
            cfg.video = VideoConfig(**vr)
        if metrics_raw := raw.get("metrics"):
            cfg.metrics = MetricsConfig(**metrics_raw)
        if ft_raw := raw.get("failure_thresholds"):
            cfg.failure_thresholds = FailureThresholdsConfig(**ft_raw)
        if cameras_raw := raw.get("cameras"):
            cfg.cameras = [CameraConfig(**c) for c in cameras_raw]
        else:
            cfg._set_default_cameras()

        for key in ("skip_video", "skip_report", "dry_run",
                    "save_raw_observations", "frame_subdir",
                    "video_subdir", "metrics_filename"):
            if key in raw:
                setattr(cfg, key, raw[key])

        return cfg

    def _set_default_cameras(self):
        self.cameras = [
            CameraConfig("front",      [0.6,  0.0, 0.4], [0.0, 0.0, 0.1], 200),
            CameraConfig("front_left", [0.4,  0.5, 0.4], [0.0, 0.0, 0.1], 200),
            CameraConfig("side",       [0.0,  0.7, 0.3], [0.0, 0.0, 0.1], 200),
            CameraConfig("rear",       [-0.5, 0.0, 0.4], [0.0, 0.0, 0.1], 200),
            CameraConfig("top",        [0.0,  0.0, 0.9], [0.0, 0.0, 0.0], 200),
        ]

    def to_yaml(self, path: str):
        import dataclasses
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(i) for i in obj]
            return obj
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, allow_unicode=True, default_flow_style=False)
