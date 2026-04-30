"""
tactile_overlay.py
==================
Overlays Fingertip Deform Map data onto rendered video frames.

Target hardware: Sharpa Hand (22-DOF, 5 fingers)
  - Each fingertip provides a (H_d, W_d) float32 pressure distribution map.
  - Heatmaps are composited onto designated UI regions of the RGB video frame.

Dependencies:
  - opencv-python (cv2)
  - numpy

No Isaac dependency — can be applied as an offline post-processing step.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("observer.tactile.overlay")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    log.warning("opencv-python not found — tactile overlay disabled.")


# ── Sharpa Hand finger definitions ────────────────────────────────────
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class FingerUILayout:
    """Screen-space position of a single finger's tactile panel."""
    name: str
    panel_x: int    # Top-left x (pixels)
    panel_y: int    # Top-left y (pixels)
    width: int  = 60
    height: int = 60


# Default panel layouts by resolution
DEFAULT_LAYOUT_1080P = [
    FingerUILayout("thumb",  20,  920, 60, 60),
    FingerUILayout("index",  90,  920, 60, 60),
    FingerUILayout("middle", 160, 920, 60, 60),
    FingerUILayout("ring",   230, 920, 60, 60),
    FingerUILayout("pinky",  300, 920, 60, 60),
]

DEFAULT_LAYOUT_720P = [
    FingerUILayout("thumb",  20,  640, 50, 50),
    FingerUILayout("index",  80,  640, 50, 50),
    FingerUILayout("middle", 140, 640, 50, 50),
    FingerUILayout("ring",   200, 640, 50, 50),
    FingerUILayout("pinky",  260, 640, 50, 50),
]


class TactileOverlayRenderer:
    """
    Composites per-fingertip tactile heatmaps onto a single video frame.

    Parameters
    ----------
    layout : list[FingerUILayout] | None
        UI panel positions per finger. Auto-selected by resolution if None.
    alpha : float
        Background blend weight (0 = heatmap only, 1 = original only).
    colormap : int
        OpenCV colormap constant (default: COLORMAP_JET).
    show_labels : bool
        Render finger name text in each panel.
    show_border : bool
        Draw panel border; red if slip is detected.
    normalize_per_finger : bool
        True  -> each finger normalized independently (emphasizes relative pressure).
        False -> global normalization across all fingers (enables absolute comparison).
    """

    def __init__(
        self,
        layout: Optional[list[FingerUILayout]] = None,
        alpha: float = 0.35,
        colormap: int = None,
        show_labels: bool = True,
        show_border: bool = True,
        normalize_per_finger: bool = True,
    ):
        self.layout   = layout
        self.alpha    = alpha
        self.colormap = colormap if colormap is not None else (
            cv2.COLORMAP_JET if _CV2_AVAILABLE else 2
        )
        self.show_labels         = show_labels
        self.show_border         = show_border
        self.normalize_per_finger = normalize_per_finger

    def _get_layout(self, frame_h: int) -> list[FingerUILayout]:
        if self.layout:
            return self.layout
        return DEFAULT_LAYOUT_1080P if frame_h >= 900 else DEFAULT_LAYOUT_720P

    # ── Single-frame rendering ────────────────────────────────────────
    def render_frame(
        self,
        frame: np.ndarray,
        deform_maps: np.ndarray,
        slip_flags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply tactile overlay to one video frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8  BGR or RGB image
        deform_maps : (n_fingers, H_d, W_d) float32
            Pressure distribution per finger. Arbitrary value range.
        slip_flags : (n_fingers,) bool, optional
            If True for a finger, its panel border turns red.

        Returns
        -------
        (H, W, 3) uint8  frame with overlay applied.
        """
        if not _CV2_AVAILABLE or deform_maps is None or len(deform_maps) == 0:
            return frame

        output = frame.copy()
        h, w = frame.shape[:2]
        layout = self._get_layout(h)

        if not self.normalize_per_finger:
            global_min = deform_maps.min()
            global_rng = deform_maps.max() - global_min + 1e-8

        for i, fl in enumerate(layout):
            if i >= len(deform_maps):
                break

            patch = deform_maps[i].astype(np.float32)

            # Normalization
            if self.normalize_per_finger:
                pmin = patch.min()
                rng  = patch.max() - pmin + 1e-8
                patch_norm = ((patch - pmin) / rng * 255).astype(np.uint8)
            else:
                patch_norm = ((patch - global_min) / global_rng * 255).astype(np.uint8)

            # Colormap + resize
            heatmap = cv2.applyColorMap(patch_norm, self.colormap)
            heatmap_resized = cv2.resize(
                heatmap, (fl.width, fl.height), interpolation=cv2.INTER_LINEAR
            )

            # Bounds check
            y1, y2 = fl.panel_y, fl.panel_y + fl.height
            x1, x2 = fl.panel_x, fl.panel_x + fl.width
            if y2 > h or x2 > w:
                continue

            # Alpha blend
            roi = output[y1:y2, x1:x2]
            output[y1:y2, x1:x2] = cv2.addWeighted(
                roi, self.alpha, heatmap_resized, 1 - self.alpha, 0
            )

            # Border (red on slip)
            if self.show_border:
                slip_detected = (
                    slip_flags is not None
                    and i < len(slip_flags)
                    and slip_flags[i]
                )
                border_color = (0, 0, 255) if slip_detected else (200, 200, 200)
                cv2.rectangle(output, (x1, y1), (x2, y2), border_color, 2)

            # Finger label
            if self.show_labels:
                label = FINGER_NAMES[i][:3] if i < len(FINGER_NAMES) else f"f{i}"
                cv2.putText(
                    output, label, (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA
                )

            # Peak pressure value
            cv2.putText(
                output, f"{float(deform_maps[i].max()):.2f}",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1, cv2.LINE_AA
            )

        # Global slip warning
        if slip_flags is not None and any(slip_flags[:len(layout)]):
            cv2.putText(
                output, "SLIP DETECTED", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )

        return output

    # ── Offline video post-processing ─────────────────────────────────
    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        deform_sequence: list[np.ndarray],
        slip_sequence: Optional[list[np.ndarray]] = None,
    ) -> bool:
        """
        Apply tactile overlay to an existing video file and save the result.

        Parameters
        ----------
        input_path : Path
            Source video.
        output_path : Path
            Destination video.
        deform_sequence : list[(n_fingers, H_d, W_d)]
            Per-frame deform maps.
        slip_sequence : list[(n_fingers,)] | None
            Per-frame slip flags (optional).
        """
        if not _CV2_AVAILABLE:
            log.error("cv2 not available — cannot process video.")
            return False

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            log.error(f"Failed to open video: {input_path}")
            return False

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            deform = deform_sequence[frame_idx] if frame_idx < len(deform_sequence) else None
            slip   = (slip_sequence[frame_idx]
                      if slip_sequence and frame_idx < len(slip_sequence) else None)

            if deform is not None:
                frame = self.render_frame(frame, deform, slip)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        log.info(f"Tactile overlay complete: {output_path} ({frame_idx} frames)")
        return True

    # ── Colormap legend ───────────────────────────────────────────────
    def generate_legend(self, output_path: Path, width: int = 400, height: int = 60):
        """Generate a colormap legend image for use in reports."""
        if not _CV2_AVAILABLE:
            return
        gradient = np.tile(np.arange(256, dtype=np.uint8), (height // 2, 1))
        cmap_img = cv2.applyColorMap(gradient, self.colormap)
        legend   = np.zeros((height, width, 3), dtype=np.uint8)
        legend[:height//2, :] = cv2.resize(cmap_img, (width, height // 2))
        cv2.putText(legend, "Low Pressure",  (5, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(legend, "High Pressure", (width - 120, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imwrite(str(output_path), legend)
        log.info(f"Legend saved: {output_path}")
