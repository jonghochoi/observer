"""
eval_runner.py
==============
OBSERVER — entry point for the checkpoint evaluation pipeline.

Usage:
    python eval_runner.py --checkpoint runs/exp_001/model_5000.pth
    python eval_runner.py --checkpoint_dir runs/exp_001/
    python eval_runner.py --checkpoint_dir runs/ --recursive --latest_only
    python eval_runner.py --checkpoint_dir runs/ --auto_select --select_weights hardware_safe
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.experiment_tracker import ExperimentTracker
from observer.pipeline.auto_select import CheckpointSelector, ScoringWeights
from observer.report.report_generator import ReportGenerator
from observer.brand import print_banner, print_flow, rule, log as obs_log, VERSION_STRING

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("observer.runner")


def parse_args():
    parser = argparse.ArgumentParser(description="OBSERVER — Automated Evaluation Pipeline")

    # Checkpoint source
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint",     type=str, help="Single .pth file path")
    ckpt_group.add_argument("--checkpoint_dir", type=str, help="Directory containing .pth files")

    # Directory traversal
    parser.add_argument("--recursive",    action="store_true",
                        help="Recursively search subdirectories for .pth files")
    parser.add_argument("--latest_only",  action="store_true",
                        help="Evaluate only the most recent checkpoint per directory")

    # Config
    parser.add_argument("--config", type=str,
                        default="observer/configs/eval_config.yaml",
                        help="Path to eval_config.yaml")

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="observer-eval",
                        help="W&B project name")
    parser.add_argument("--wandb_run",     type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    parser.add_argument("--tb_log_dir",    type=str, default="tb_logs",
                        help="TensorBoard log directory")
    parser.add_argument("--no_tracking",   action="store_true",
                        help="Disable W&B / TensorBoard logging")

    # Auto checkpoint selection
    parser.add_argument("--auto_select",    action="store_true",
                        help="Automatically select and deploy the best checkpoint after evaluation")
    parser.add_argument("--select_weights", type=str, default="balanced",
                        choices=["balanced", "hardware_safe", "performance_first"],
                        help="Multi-objective scoring weight preset")
    parser.add_argument("--deploy_top_k",  type=int, default=1,
                        help="Number of top checkpoints to deploy")

    # Execution modes
    parser.add_argument("--dry_run",     action="store_true",
                        help="Validate pipeline structure without running Isaac")
    parser.add_argument("--skip_video",  action="store_true",
                        help="Skip video recording (metrics only)")
    parser.add_argument("--skip_report", action="store_true",
                        help="Skip HTML report generation")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Root directory for all evaluation outputs")

    return parser.parse_args()


def collect_checkpoints(args) -> list[Path]:
    """Gather .pth files from the specified source."""
    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.exists():
            log.error(f"Checkpoint file not found: {p}")
            sys.exit(1)
        return [p]

    root = Path(args.checkpoint_dir)
    if not root.exists():
        log.error(f"Directory not found: {root}")
        sys.exit(1)

    pattern = "**/*.pth" if args.recursive else "*.pth"
    ckpts   = sorted(root.glob(pattern))

    if not ckpts:
        log.error(f"No .pth files found in '{root}'.")
        sys.exit(1)

    if args.latest_only:
        from itertools import groupby
        ckpts = [
            max(group, key=lambda p: p.stat().st_mtime)
            for _, group in groupby(ckpts, key=lambda p: p.parent)
        ]
        log.info(f"latest_only: selected {len(ckpts)} checkpoint(s)")

    log.info(f"Found {len(ckpts)} checkpoint(s):")
    for ck in ckpts:
        log.info(f"  * {ck}")
    return ckpts


def main():
    args = parse_args()

    # Config
    config = EvalConfig.from_yaml(args.config)
    config.skip_video  = args.skip_video  or config.skip_video
    config.skip_report = args.skip_report or config.skip_report
    config.dry_run     = args.dry_run

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = collect_checkpoints(args)

    # Startup display
    print_banner()
    print_flow()
    print(rule())
    log.info(obs_log(f"Version     : {VERSION_STRING}", "info"))
    log.info(obs_log(f"Checkpoints : {len(checkpoints)}", "info"))
    log.info(obs_log(f"Output root : {output_root.resolve()}", "info"))
    log.info(obs_log(f"Dry-run     : {config.dry_run}", "info"))
    log.info(obs_log(
        f"Tracking    : {'OFF' if args.no_tracking else 'ON (W&B/TB auto-detected)'}",
        "info"
    ))
    log.info(obs_log(
        f"Auto-select : {'ON (' + args.select_weights + ')' if args.auto_select else 'OFF'}",
        "info"
    ))
    print(rule())

    # Experiment tracker
    tracker = ExperimentTracker(
        project=args.wandb_project,
        run_name=args.wandb_run,
        tb_log_dir=Path(args.tb_log_dir),
        tags=[config.runtime.task, f"n_ckpt={len(checkpoints)}"],
        config={
            "task":       config.runtime.task,
            "num_envs":   config.runtime.num_envs,
            "n_episodes": config.metrics.num_eval_episodes,
            "n_cameras":  len(config.cameras),
        },
        enabled=not args.no_tracking and not config.dry_run,
    )

    _weights_map = {
        "balanced":          ScoringWeights.balanced(),
        "hardware_safe":     ScoringWeights.hardware_safe(),
        "performance_first": ScoringWeights.performance_first(),
    }

    # Run pipeline
    orchestrator = PipelineOrchestrator(
        config=config, output_root=output_root, tracker=tracker
    )
    all_results = []

    t0 = time.time()
    for idx, ckpt in enumerate(checkpoints):
        log.info(obs_log(f"[{idx+1}/{len(checkpoints)}] Scanning: {ckpt.name}", "info"))
        result = orchestrator.run_single(ckpt)
        all_results.append(result)

    log.info(obs_log(f"All scans complete: {time.time()-t0:.1f}s elapsed", "ok"))

    # Auto-select and deploy
    if args.auto_select:
        log.info(obs_log("Running multi-objective scoring...", "info"))
        selector = CheckpointSelector(
            weights=_weights_map[args.select_weights],
            output_root=output_root,
        )
        deployed = selector.deploy_best(all_results, top_k=args.deploy_top_k)
        log.info(obs_log(f"Deployed: {[str(p) for p in deployed]}", "ok"))
        log.info("\n" + selector.ranking_table(all_results))

    # HTML report
    if not config.skip_report:
        log.info(obs_log("Generating HTML report...", "info"))
        report_path = ReportGenerator(output_root=output_root).generate(all_results)
        log.info(obs_log(f"Report saved: {report_path}", "ok"))

    tracker.finish()
    print(rule("Scan Complete"))
    log.info(obs_log("Pipeline finished.", "ok"))


if __name__ == "__main__":
    main()
