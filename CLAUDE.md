# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` covers what OBSERVER does, how to install it, and the full CLI surface — read it first. `docs/INTEGRATION.md` is the contract observer expects from user-supplied eval/record scripts. This file only documents the rules those two don't make explicit.

## Non-obvious conventions

- **Subprocess-only boundary.** The orchestrator never imports the user's RL stack, torch, or Isaac. Only files under `isaac/` may `import omni.*`, and even those degrade to mock mode when the import fails. If you're adding a torch/env/Isaac import outside `isaac/`, you're violating the design.
- **`docs/INTEGRATION.md` ↔ orchestrator CLIs must stay in sync.** The two subprocess command shapes live in `pipeline/orchestrator.py` (`_build_metrics_cmd`, `_build_record_cmd`). Any change to either must update `INTEGRATION.md` in the same commit — adapter authors rely on it.
- **Preserve metric-key fallbacks.** `auto_select.py` and `report_generator.py` accept aliases (e.g. `energy_J_mean` ∨ `energy_J_per_episode`, `object_pos_error_mm_mean` ∨ `object_pose_error_mm`) so older user scripts keep working. Don't collapse them when refactoring.
- **Optional deps go through `try/except` + `_*_AVAILABLE` flags** at module load (see `experiment_tracker.py`, `state_coverage.py`, `isaac/recorder.py`). Follow that pattern for new optional integrations rather than hard-importing.
- **Framework-specific knobs belong in `runtime.extra_eval_args` / `extra_record_args`**, forwarded verbatim to the subprocess. Don't add framework-specific fields to `RuntimeConfig` or the orchestrator.
- **`brand.py`** is the single source for ANSI styling and the banner. Don't sprinkle ANSI codes elsewhere.

## Things that surprise people

- No `__init__.py` anywhere — relies on PEP 420 namespace packages. `eval_runner.py` lives at the repo *root*, not under `observer/`, but imports use `observer.*`. Run via the installed `observer` console script (`pip install -e .`) or from the repo's parent directory.
- `--dry_run` is the closest thing to a test suite: it skips the metrics/record subprocesses and feeds `_dummy_metrics()` through the rest of the pipeline. Use it to exercise report/ranking/tracker code paths offline.
- One failed checkpoint never aborts a sweep — `_run_subprocess` errors are caught in `run_single` and recorded on `CheckpointResult.error_msg`.
