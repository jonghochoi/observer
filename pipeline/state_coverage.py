"""
state_coverage.py
=================
Analyzes and visualizes the success/failure distribution over initial object pose space.

Core insight:
  Aggregate success rate hides *where* failures concentrate.
  If failures cluster in the roll=30-60 deg, pitch=20-40 deg region,
  that zone should receive higher sampling weight in the next curriculum iteration.

Outputs:
  - success_heatmap.png   : 2-D success rate heatmap (roll x pitch bins)
  - coverage_scatter.png  : Per-episode scatter colored by failure mode
  - pose_histogram.png    : Roll / pitch / yaw distribution histograms
  - coverage_stats.json   : Numerical summary (worst zone, uniformity, high-risk zones)
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("observer.pipeline.state_coverage")

try:
    import matplotlib
    matplotlib.use("Agg")   # headless-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    log.warning("matplotlib not available — coverage plots will be skipped.")


@dataclass
class PosedEpisode:
    """
    Episode record extended with initial object pose information.
    Compatible with MetricsCollector.EpisodeStats (superset of fields).
    """
    success: bool
    failure_mode: str  = "unknown"
    init_roll_deg: float  = 0.0
    init_pitch_deg: float = 0.0
    init_yaw_deg: float   = 0.0
    init_pos_x: float     = 0.0
    init_pos_y: float     = 0.0
    init_pos_z: float     = 0.0
    episode_length: int   = 0
    slip_count: int       = 0


@dataclass
class CoverageStats:
    """Numerical coverage analysis summary."""
    n_episodes: int          = 0
    success_rate_overall: float = 0.0
    worst_roll_bin: str      = ""
    worst_pitch_bin: str     = ""
    coverage_uniformity: float = 0.0   # Entropy-based score: 0=concentrated, 1=uniform
    high_risk_zones: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class StateCoverageAnalyzer:
    """
    Analyzes initial pose coverage and generates diagnostic visualizations.

    Parameters
    ----------
    output_dir : Path
        Directory where plots and stats are saved.
    roll_bins : int
        Number of bins along the roll axis.
    pitch_bins : int
        Number of bins along the pitch axis.
    """

    # Failure mode color palette (matches report_generator)
    COLORS = {
        "success":         "#22c55e",
        "early_drop":      "#ef4444",
        "singularity_hit": "#f97316",
        "late_slip":       "#a855f7",
        "contact_loss":    "#06b6d4",
        "repose_failure":  "#f59e0b",
        "timeout":         "#6b7280",
        "unknown":         "#374151",
    }

    def __init__(
        self,
        output_dir: Path,
        roll_bins: int = 12,
        pitch_bins: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.roll_bins  = roll_bins
        self.pitch_bins = pitch_bins

    # ── Main entry point ──────────────────────────────────────────────
    def analyze(self, episodes: list[PosedEpisode]) -> CoverageStats:
        """Run full analysis, save all outputs, return numerical summary."""
        if not episodes:
            log.warning("No episodes provided — coverage analysis skipped.")
            return CoverageStats()

        stats = self._compute_stats(episodes)

        if _MPL_AVAILABLE:
            self._plot_success_heatmap(episodes)
            self._plot_scatter(episodes)
            self._plot_histogram(episodes)
        else:
            log.info("matplotlib unavailable — skipping visualizations, saving JSON only.")

        with open(self.output_dir / "coverage_stats.json", "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        log.info(f"Coverage analysis saved to: {self.output_dir}")
        return stats

    # ── Numerical statistics ──────────────────────────────────────────
    def _compute_stats(self, episodes: list[PosedEpisode]) -> CoverageStats:
        n  = len(episodes)
        sr = sum(ep.success for ep in episodes) / n

        rolls  = np.array([ep.init_roll_deg  for ep in episodes])
        pitchs = np.array([ep.init_pitch_deg for ep in episodes])

        roll_edges  = np.linspace(rolls.min(),  rolls.max(),  self.roll_bins + 1)
        pitch_edges = np.linspace(pitchs.min(), pitchs.max(), self.pitch_bins + 1)

        worst_sr    = 1.0
        worst_roll  = ""
        worst_pitch = ""
        high_risk   = []

        for ri in range(self.roll_bins):
            for pi in range(self.pitch_bins):
                mask = (
                    (rolls  >= roll_edges[ri])  & (rolls  < roll_edges[ri+1]) &
                    (pitchs >= pitch_edges[pi]) & (pitchs < pitch_edges[pi+1])
                )
                cnt = mask.sum()
                if cnt == 0:
                    continue
                bin_sr = sum(ep.success for ep, m in zip(episodes, mask) if m) / cnt
                roll_range  = f"{roll_edges[ri]:.0f}°~{roll_edges[ri+1]:.0f}°"
                pitch_range = f"{pitch_edges[pi]:.0f}°~{pitch_edges[pi+1]:.0f}°"
                if bin_sr < worst_sr:
                    worst_sr    = bin_sr
                    worst_roll  = roll_range
                    worst_pitch = pitch_range
                if bin_sr < 0.5 and cnt >= 3:
                    high_risk.append({
                        "roll_range":   roll_range,
                        "pitch_range":  pitch_range,
                        "success_rate": round(bin_sr, 3),
                        "n_episodes":   int(cnt),
                    })

        # Uniformity via entropy
        hist, _ = np.histogramdd(
            np.stack([rolls, pitchs], axis=1),
            bins=[self.roll_bins, self.pitch_bins]
        )
        hist_flat = hist.flatten()
        hist_norm = hist_flat[hist_flat > 0] / hist_flat.sum()
        max_entropy = math.log(len(hist_flat))
        entropy = -np.sum(hist_norm * np.log(hist_norm))
        uniformity = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return CoverageStats(
            n_episodes=n,
            success_rate_overall=round(sr, 4),
            worst_roll_bin=worst_roll,
            worst_pitch_bin=worst_pitch,
            coverage_uniformity=round(uniformity, 4),
            high_risk_zones=high_risk[:10],
        )

    # ── Visualizations ────────────────────────────────────────────────
    def _plot_success_heatmap(self, episodes: list[PosedEpisode]):
        """2-D success rate heatmap binned by roll x pitch."""
        rolls  = np.array([ep.init_roll_deg  for ep in episodes])
        pitchs = np.array([ep.init_pitch_deg for ep in episodes])
        successes = np.array([ep.success for ep in episodes], dtype=float)

        roll_edges  = np.linspace(rolls.min(),  rolls.max(),  self.roll_bins + 1)
        pitch_edges = np.linspace(pitchs.min(), pitchs.max(), self.pitch_bins + 1)

        heatmap   = np.full((self.pitch_bins, self.roll_bins), np.nan)
        count_map = np.zeros((self.pitch_bins, self.roll_bins))

        for ri in range(self.roll_bins):
            for pi in range(self.pitch_bins):
                mask = (
                    (rolls  >= roll_edges[ri])  & (rolls  < roll_edges[ri+1]) &
                    (pitchs >= pitch_edges[pi]) & (pitchs < pitch_edges[pi+1])
                )
                cnt = mask.sum()
                count_map[pi, ri] = cnt
                if cnt > 0:
                    heatmap[pi, ri] = successes[mask].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")

        cmap = plt.cm.RdYlGn
        cmap.set_bad(color="#2a2d3e")
        im = axes[0].imshow(
            heatmap, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower",
            extent=[roll_edges[0], roll_edges[-1], pitch_edges[0], pitch_edges[-1]]
        )
        axes[0].set_xlabel("Initial Roll (deg)", color="#e2e8f0")
        axes[0].set_ylabel("Initial Pitch (deg)", color="#e2e8f0")
        axes[0].set_title("Success Rate by Initial Pose", color="#e2e8f0", fontsize=13)
        plt.colorbar(im, ax=axes[0])

        im2 = axes[1].imshow(
            count_map, cmap="Blues", aspect="auto", origin="lower",
            extent=[roll_edges[0], roll_edges[-1], pitch_edges[0], pitch_edges[-1]]
        )
        axes[1].set_xlabel("Initial Roll (deg)", color="#e2e8f0")
        axes[1].set_ylabel("Initial Pitch (deg)", color="#e2e8f0")
        axes[1].set_title("Sample Count by Pose Bin", color="#e2e8f0", fontsize=13)
        plt.colorbar(im2, ax=axes[1])

        for ax in axes:
            ax.tick_params(colors="#8892a4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")

        plt.tight_layout()
        out = self.output_dir / "success_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Success rate heatmap: {out}")

    def _plot_scatter(self, episodes: list[PosedEpisode]):
        """Scatter plot of episodes in roll-pitch space, colored by failure mode."""
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")

        mode_groups: dict[str, list] = {}
        for ep in episodes:
            mode = ep.failure_mode if hasattr(ep, "failure_mode") else (
                "success" if ep.success else "unknown"
            )
            mode_groups.setdefault(mode, []).append(ep)

        legend_patches = []
        for mode, eps in mode_groups.items():
            color  = self.COLORS.get(mode, "#6b7280")
            xs     = [ep.init_roll_deg  for ep in eps]
            ys     = [ep.init_pitch_deg for ep in eps]
            alpha  = 0.9 if mode == "success" else 0.7
            size   = 30  if mode == "success" else 50
            zorder = 2   if mode == "success" else 3
            ax.scatter(xs, ys, c=color, s=size, alpha=alpha,
                       zorder=zorder, edgecolors="none")
            legend_patches.append(Patch(color=color, label=f"{mode} ({len(eps)})"))

        ax.set_xlabel("Initial Roll (deg)", color="#e2e8f0", fontsize=11)
        ax.set_ylabel("Initial Pitch (deg)", color="#e2e8f0", fontsize=11)
        ax.set_title("Episode Outcome by Initial Object Pose", color="#e2e8f0", fontsize=13)
        ax.tick_params(colors="#8892a4")
        ax.legend(
            handles=legend_patches, loc="upper right",
            facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#e2e8f0", fontsize=9
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

        plt.tight_layout()
        out = self.output_dir / "coverage_scatter.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Coverage scatter: {out}")

    def _plot_histogram(self, episodes: list[PosedEpisode]):
        """Roll / pitch / yaw distribution histograms, split by success/failure."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0f1117")

        data_list = [
            ([ep.init_roll_deg  for ep in episodes], "Initial Roll (deg)"),
            ([ep.init_pitch_deg for ep in episodes], "Initial Pitch (deg)"),
            ([ep.init_yaw_deg   for ep in episodes], "Initial Yaw (deg)"),
        ]
        success_mask = np.array([ep.success for ep in episodes], dtype=bool)

        for ax, (data, label) in zip(axes, data_list):
            ax.set_facecolor("#1a1d27")
            data_arr  = np.array(data)
            succ_data = data_arr[success_mask]
            fail_data = data_arr[~success_mask]
            bins = np.linspace(data_arr.min(), data_arr.max(), 20)
            ax.hist(fail_data,  bins=bins, color="#ef4444", alpha=0.7, label="Fail")
            ax.hist(succ_data,  bins=bins, color="#22c55e", alpha=0.7, label="Success")
            ax.set_xlabel(label, color="#e2e8f0")
            ax.set_ylabel("Count", color="#e2e8f0")
            ax.tick_params(colors="#8892a4")
            ax.legend(facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#e2e8f0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")

        fig.suptitle("Initial Pose Distribution", color="#e2e8f0", fontsize=13)
        plt.tight_layout()
        out = self.output_dir / "pose_histogram.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Pose histogram: {out}")
