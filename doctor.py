"""
doctor.py
==========
OBSERVER pre-flight check.

Verifies the environment, dependencies, and config before any Isaac
subprocess is launched. Designed to be the first thing a new teammate
runs after `pip install -e .`.

Usage:
    observer doctor                       # uses default config
    observer doctor --config path.yaml    # custom config
    observer doctor --skip-runtime        # env-only check (no config load)
"""

from __future__ import annotations
import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from observer.brand import (
    BOLD, DIM, GREEN, RED, RESET, SIGIL, YELLOW, rule,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "eval_config.yaml"


def _ok(msg: str) -> None:   print(f"  {GREEN}✓{RESET} {msg}")
def _warn(msg: str) -> None: print(f"  {YELLOW}!{RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {RED}✗{RESET} {msg}")


# ── Environment checks ────────────────────────────────────────────────

def check_python() -> bool:
    v = sys.version_info
    if v < (3, 10):
        _fail(f"Python {v.major}.{v.minor} — observer requires >= 3.10")
        return False
    _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_core_deps() -> bool:
    ok = True
    for mod, hint in (("numpy", "numpy"),
                      ("yaml", "pyyaml"),
                      ("matplotlib", "matplotlib")):
        if importlib.util.find_spec(mod):
            _ok(f"import {mod}")
        else:
            _fail(f"import {mod} — pip install {hint}")
            ok = False
    return ok


def check_optional_deps() -> None:
    for mod, hint in (("wandb",       "pip install wandb"),
                      ("tensorboard", "pip install tensorboard"),
                      ("cv2",         "pip install opencv-python")):
        label = "opencv-python" if mod == "cv2" else mod
        if importlib.util.find_spec(mod):
            _ok(f"import {label} {DIM}(optional){RESET}")
        else:
            _warn(f"{label} not installed — {hint} {DIM}(optional){RESET}")


def check_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        _ok("ffmpeg in PATH")
        return True
    _fail("ffmpeg not found — apt install ffmpeg (required for video stage)")
    return False


def check_cuda() -> None:
    if not shutil.which("nvidia-smi"):
        _warn("nvidia-smi not found — CPU-only mode")
        return
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
        _ok("nvidia-smi works")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _warn(f"nvidia-smi failed: {e}")


# ── Config checks ─────────────────────────────────────────────────────

def check_config(config_path: Path):
    if not config_path.exists():
        _fail(f"config not found: {config_path}")
        return None
    try:
        from observer.configs.eval_config import EvalConfig
        cfg = EvalConfig.from_yaml(str(config_path))
        _ok(f"config loaded: {config_path}")
        return cfg
    except Exception as e:
        _fail(f"config load error: {type(e).__name__}: {e}")
        return None


def check_runtime(config) -> bool:
    rt = config.runtime
    ok = True

    if not rt.task:
        _fail("runtime.task is empty — fill it in eval_config.yaml")
        ok = False
    else:
        _ok(f"runtime.task = {rt.task!r}")

    if not rt.eval_module:
        _fail("runtime.eval_module is empty")
        ok = False
    else:
        if importlib.util.find_spec(rt.eval_module) is not None:
            _ok(f"runtime.eval_module importable: {rt.eval_module}")
        else:
            _fail(
                f"runtime.eval_module not importable: {rt.eval_module} "
                f"— is it pip-installed in this env?"
            )
            ok = False

    if config.skip_video:
        _ok(f"skip_video=true {DIM}(record stage disabled){RESET}")
        return ok

    # Record stage prerequisites
    if not rt.record_script:
        _warn("runtime.record_script empty — set skip_video=true to silence")
    else:
        rs = Path(rt.record_script)
        if rs.exists():
            _ok(f"runtime.record_script exists: {rs}")
        else:
            _fail(f"runtime.record_script not found: {rs}")
            ok = False

    isaac = Path(rt.resolve_isaac_lab_path())
    if "$" in str(isaac):
        _fail(f"isaac_lab_path has unresolved vars: {isaac} — set $ISAACLAB_PATH")
        ok = False
    elif isaac.exists():
        _ok(f"isaac_lab_path exists: {isaac}")
    else:
        _fail(f"isaac_lab_path not found: {isaac}")
        ok = False

    return ok


# ── Entry point ───────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="observer doctor",
        description="Pre-flight environment & config check.",
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help=f"path to eval_config.yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="check environment only; skip config and runtime")
    args = parser.parse_args(argv)

    print(rule("OBSERVER Doctor"))

    print(f"\n{BOLD}Environment{RESET}")
    py_ok = check_python()
    deps_ok = check_core_deps()
    check_optional_deps()
    ff_ok = check_ffmpeg()
    check_cuda()

    cfg_ok = True
    rt_ok = True
    if not args.skip_runtime:
        print(f"\n{BOLD}Configuration{RESET}")
        cfg = check_config(Path(args.config))
        if cfg is None:
            cfg_ok = False
            rt_ok = False
        else:
            rt_ok = check_runtime(cfg)

    print()
    print(rule())
    if py_ok and deps_ok and ff_ok and cfg_ok and rt_ok:
        print(f"  {SIGIL} {GREEN}All checks passed.{RESET} Ready to run.")
        return 0
    print(f"  {SIGIL} {RED}Issues detected.{RESET} Fix the {RED}✗{RESET} items above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
