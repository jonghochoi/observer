"""
auto_select.py
==============
Multi-objective scoring for automatic best-checkpoint selection and deployment.

Motivation:
  Ranking by success rate alone is misleading.
  A policy with 95% success rate but 10x energy consumption, or high slip
  frequency, may be unsafe or unreliable on real hardware.
  A weighted multi-objective score captures this trade-off explicitly.

Scoring formula:
  Score = w_sr    *  success_rate       (higher is better)
        - w_slip   *  slip_per_ep_norm  (lower is better)
        - w_energy *  energy_norm       (lower is better)
        - w_pos    *  pos_error_norm     (lower is better)
        - w_cf     *  contact_force_norm (lower is better)

All metrics are min-max normalized before weighting so that scales are comparable.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("observer.pipeline.auto_select")


@dataclass
class ScoringWeights:
    """
    Per-metric importance weights.
    Values do not need to sum to 1 — normalization is applied internally.
    """
    success_rate:   float = 1.0
    slip:           float = 0.3
    energy:         float = 0.2
    pos_error:      float = 0.2
    contact_force:  float = 0.1   # Lower contact RMS -> more stable grasp

    @classmethod
    def hardware_safe(cls) -> "ScoringWeights":
        """Emphasizes hardware safety: penalizes slip and energy heavily."""
        return cls(success_rate=1.0, slip=0.6, energy=0.4, pos_error=0.1, contact_force=0.3)

    @classmethod
    def performance_first(cls) -> "ScoringWeights":
        """Prioritizes task success rate above all else."""
        return cls(success_rate=2.0, slip=0.1, energy=0.05, pos_error=0.3, contact_force=0.05)

    @classmethod
    def balanced(cls) -> "ScoringWeights":
        """Balanced default weights."""
        return cls()


@dataclass
class ScoredCheckpoint:
    """Scoring result for a single checkpoint."""
    checkpoint: Path
    output_dir: Path
    metrics: dict
    score: float
    rank: int = 0
    score_breakdown: dict = field(default_factory=dict)

    def summary_str(self) -> str:
        sr   = self.metrics.get("success_rate", 0)
        slip = self.metrics.get("slip_events_per_episode", 0)
        eng  = self.metrics.get("energy_J_mean", self.metrics.get("energy_J_per_episode", 0))
        return (
            f"  Rank #{self.rank:02d} | Score={self.score:+.4f} | "
            f"SR={sr*100:.1f}% | Slip={slip:.2f}/ep | Energy={eng:.3f}J | "
            f"{self.checkpoint.name}"
        )


class CheckpointSelector:
    """
    Ranks checkpoints by multi-objective score and deploys the top-k.

    Parameters
    ----------
    weights : ScoringWeights
        Per-metric importance weights.
    output_root : Path
        Root evaluation directory. The 'best/' subdirectory is created here.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        output_root: Optional[Path] = None,
    ):
        self.weights = weights or ScoringWeights.balanced()
        self.output_root = output_root

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_all(self, results: list) -> list[ScoredCheckpoint]:
        """
        Score and rank a list of CheckpointResult objects.

        Returns
        -------
        list[ScoredCheckpoint]
            Sorted by score descending (rank 1 = best).
        """
        valid = [r for r in results if r.success and r.metrics]
        if not valid:
            log.warning("No scoreable results found.")
            return []

        def _collect(key, *alt_keys):
            vals = []
            for r in valid:
                for k in [key] + list(alt_keys):
                    if k in r.metrics:
                        vals.append(r.metrics[k])
                        break
            return np.array(vals) if vals else np.zeros(len(valid))

        sr_arr   = _collect("success_rate")
        slip_arr = _collect("slip_events_per_episode")
        eng_arr  = _collect("energy_J_mean", "energy_J_per_episode")
        pos_arr  = _collect("object_pos_error_mm_mean", "object_pose_error_mm")
        cf_arr   = _collect("contact_force_rms_mean", "contact_force_rms")

        def _norm(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 1e-8 else np.zeros_like(arr)

        sr_n  = _norm(sr_arr)
        slip_n = _norm(slip_arr)
        eng_n  = _norm(eng_arr)
        pos_n  = _norm(pos_arr)
        cf_n   = _norm(cf_arr)

        w = self.weights
        scores = (
             w.success_rate  * sr_n
            - w.slip          * slip_n
            - w.energy        * eng_n
            - w.pos_error     * pos_n
            - w.contact_force * cf_n
        )

        scored = []
        for i, (r, sc) in enumerate(zip(valid, scores)):
            scored.append(ScoredCheckpoint(
                checkpoint=r.checkpoint,
                output_dir=r.output_dir,
                metrics=r.metrics,
                score=float(sc),
                score_breakdown={
                    "success_rate_contrib":   float( w.success_rate  * sr_n[i]),
                    "slip_contrib":           float(-w.slip          * slip_n[i]),
                    "energy_contrib":         float(-w.energy        * eng_n[i]),
                    "pos_error_contrib":      float(-w.pos_error     * pos_n[i]),
                    "contact_force_contrib":  float(-w.contact_force * cf_n[i]),
                }
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        for rank, s in enumerate(scored, 1):
            s.rank = rank

        log.info(f"\n{'='*60}")
        log.info(f"Checkpoint ranking ({len(scored)} total)")
        for s in scored[:5]:
            log.info(s.summary_str())
        if len(scored) > 5:
            log.info(f"  ... and {len(scored)-5} more")
        log.info(f"{'='*60}")

        return scored

    def select_best(self, results: list) -> Optional[ScoredCheckpoint]:
        scored = self.score_all(results)
        return scored[0] if scored else None

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------
    def deploy_best(
        self,
        results: list,
        deploy_dir: Optional[Path] = None,
        top_k: int = 1,
    ) -> list[Path]:
        """
        Symlink (or copy) the top-k checkpoints to a deployment directory.

        Parameters
        ----------
        results : list[CheckpointResult]
        deploy_dir : Path | None
            Defaults to output_root/best/.
        top_k : int
            Number of top checkpoints to deploy.

        Returns
        -------
        list[Path]
            Paths of deployed checkpoint files.
        """
        scored = self.score_all(results)
        if not scored:
            return []

        dst_root = deploy_dir or (
            self.output_root / "best" if self.output_root else Path("best")
        )
        dst_root.mkdir(parents=True, exist_ok=True)

        deployed = []
        for s in scored[:top_k]:
            src = s.checkpoint
            dst = dst_root / f"rank{s.rank:02d}__{src.name}"

            if dst.exists() or dst.is_symlink():
                dst.unlink()

            try:
                os.symlink(src.resolve(), dst)
                log.info(f"  Symlink created: {dst} -> {src}")
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
                log.info(f"  File copied: {src} -> {dst}")

            deployed.append(dst)

        # Save selection metadata for reproducibility
        meta = {
            "top_k": top_k,
            "weights": {
                "success_rate":  self.weights.success_rate,
                "slip":          self.weights.slip,
                "energy":        self.weights.energy,
                "pos_error":     self.weights.pos_error,
                "contact_force": self.weights.contact_force,
            },
            "ranking": [
                {
                    "rank": s.rank,
                    "checkpoint": str(s.checkpoint),
                    "score": s.score,
                    "score_breakdown": s.score_breakdown,
                    "metrics": {
                        k: v for k, v in s.metrics.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                }
                for s in scored
            ],
        }
        with open(dst_root / "selection_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"Selection metadata saved: {dst_root / 'selection_meta.json'}")

        return deployed

    # ------------------------------------------------------------------
    # Ranking table (plain text)
    # ------------------------------------------------------------------
    def ranking_table(self, results: list) -> str:
        scored = self.score_all(results)
        if not scored:
            return "No ranking data available."

        header = (
            f"{'Rank':>4} | {'Score':>8} | {'SR':>6} | "
            f"{'Slip/ep':>7} | {'Energy J':>8} | {'PosErr mm':>9} | Checkpoint"
        )
        sep  = "-" * len(header)
        rows = [header, sep]
        for s in scored:
            m    = s.metrics
            sr   = m.get("success_rate", 0)
            slip = m.get("slip_events_per_episode", 0)
            eng  = m.get("energy_J_mean", m.get("energy_J_per_episode", 0))
            pos  = m.get("object_pos_error_mm_mean", m.get("object_pose_error_mm", 0))
            rows.append(
                f"{s.rank:>4} | {s.score:>+8.4f} | {sr*100:>5.1f}% | "
                f"{slip:>7.2f} | {eng:>8.3f} | {pos:>9.2f} | {s.checkpoint.name}"
            )
        return "\n".join(rows)
