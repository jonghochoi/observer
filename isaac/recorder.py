"""
recorder.py
===========
Isaac Sim Replicator-based video recording wrapper.

Responsibilities:
  - Frame capture via omni.replicator.core
  - Frame sequence -> mp4 conversion via ffmpeg
  - Per-viewpoint video file management

Isaac Replicator API reference:
  https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/
"""

from __future__ import annotations
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("observer.isaac.recorder")

_REPLICATOR_AVAILABLE = False
try:
    import omni.replicator.core as rep
    _REPLICATOR_AVAILABLE = True
    log.info("omni.replicator.core loaded successfully.")
except ImportError:
    log.warning("omni.replicator.core not available — running in mock mode.")


class VideoRecorder:
    """
    Manages the recording lifecycle for a single camera view.

    Usage pattern:
        recorder = VideoRecorder(output_dir=video_dir, fps=30)
        recorder.start("front_view")
        for _ in range(200):
            sim.step()
            recorder.capture_frame()
        recorder.stop()      # -> front_view.mp4
    """

    def __init__(
        self,
        output_dir: Path,
        fps: int = 30,
        resolution: tuple[int, int] = (1920, 1080),
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        crf: int = 18,
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        self.output_dir       = Path(output_dir)
        self.fps              = fps
        self.resolution       = resolution
        self.codec            = codec
        self.pix_fmt          = pix_fmt
        self.crf              = crf
        self.camera_prim_path = camera_prim_path

        self._current_name: Optional[str] = None
        self._frame_dir: Optional[Path]   = None
        self._frame_count: int = 0
        self._writer = None
        self._render_product = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def start(self, view_name: str):
        """Begin a recording session. Any active session is stopped first."""
        if self._current_name is not None:
            log.warning(f"Previous session ({self._current_name}) not stopped — force-stopping.")
            self.stop()

        self._current_name = view_name
        self._frame_dir    = self.output_dir / f"frames_{view_name}"
        self._frame_dir.mkdir(exist_ok=True)
        self._frame_count  = 0

        log.info(f"  Recording started: {view_name} -> {self._frame_dir}")

        if _REPLICATOR_AVAILABLE:
            self._render_product = rep.create.render_product(
                self.camera_prim_path, resolution=self.resolution
            )
            self._writer = rep.WriterRegistry.get("BasicWriter")
            self._writer.initialize(
                output_dir=str(self._frame_dir), rgb=True, frame_padding=5
            )
            self._writer.attach([self._render_product])

    def capture_frame(self):
        """Capture the current render frame. Call immediately after sim.step()."""
        if _REPLICATOR_AVAILABLE and self._writer:
            rep.orchestrator.step(pause_timeline=False)
        else:
            # Mock: create empty placeholder
            if self._frame_dir:
                (self._frame_dir / f"rgb_{self._frame_count:05d}.png").touch()
        self._frame_count += 1

    def stop(self) -> Optional[Path]:
        """
        End the current recording session and convert frames to mp4.

        Returns
        -------
        Path | None
            Path to the generated mp4 file, or None on failure.
        """
        if self._current_name is None:
            return None

        view_name  = self._current_name
        frame_dir  = self._frame_dir
        self._current_name = None
        self._frame_dir    = None

        log.info(f"  Recording stopped: {view_name} ({self._frame_count} frames)")

        if _REPLICATOR_AVAILABLE and self._writer:
            self._writer.detach()
            self._render_product.destroy()
            self._writer = None
            self._render_product = None

        return self._frames_to_mp4(frame_dir, view_name)

    # ------------------------------------------------------------------
    # ffmpeg conversion
    # ------------------------------------------------------------------
    def _frames_to_mp4(self, frame_dir: Path, view_name: str) -> Optional[Path]:
        """Convert a frame directory (rgb_NNNNN.png) to mp4 via ffmpeg."""
        if not frame_dir or not frame_dir.exists():
            log.error(f"Frame directory not found: {frame_dir}")
            return None

        frames = sorted(frame_dir.glob("rgb_*.png"))
        if not frames:
            log.warning(f"No frames found in: {frame_dir}")
            return None

        out_mp4 = self.output_dir / f"{view_name}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(frame_dir / "rgb_%05d.png"),
            "-c:v", self.codec,
            "-pix_fmt", self.pix_fmt,
            "-crf", str(self.crf),
            "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}",
            str(out_mp4),
        ]
        log.info(f"  ffmpeg: {len(frames)} frames -> {out_mp4.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log.error(f"  ffmpeg failed:\n{result.stderr[-1000:]}")
            return None

        shutil.rmtree(frame_dir, ignore_errors=True)
        log.info(f"  Video saved: {out_mp4} ({out_mp4.stat().st_size // 1024} KB)")
        return out_mp4

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._current_name:
            self.stop()


# ── Integrated helper for Isaac eval scripts ──────────────────────────

def record_all_views(sim, policy, camera_controller, recorder: VideoRecorder, step_fn=None):
    """
    Run camera sweep and recording in a single call.

    Parameters
    ----------
    sim : IsaacLab SimulationApp
    policy : callable — inference policy
    camera_controller : CameraController instance
    recorder : VideoRecorder instance
    step_fn : callable | None — custom step function; defaults to sim.step()
    """
    if step_fn is None:
        step_fn = sim.step

    def on_pose_set(pose: dict):
        view_name = pose["name"]
        n_steps   = pose.get("record_steps", 200)
        recorder.start(view_name)
        for _ in range(n_steps):
            with _torch_no_grad():
                obs    = sim.get_observations()
                action = policy(obs)
                sim.step(action)
            recorder.capture_frame()
        recorder.stop()

    camera_controller.sweep(on_pose_set_callback=on_pose_set)


def _torch_no_grad():
    """torch.no_grad() context that gracefully falls back when torch is absent."""
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()
