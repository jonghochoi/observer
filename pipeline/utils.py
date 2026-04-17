"""Shared helpers used across observer/pipeline modules."""

from __future__ import annotations
import re


def extract_step(checkpoint_name: str) -> int:
    """Extract the largest integer embedded in a checkpoint filename.

    Sharpa saves checkpoints as e.g. ``ep_42_step_0005M_reward_1.23.pth``; the
    agent step count is the largest numeric run in the name.
    """
    nums = re.findall(r"\d+", checkpoint_name)
    return int(max(nums, key=int)) if nums else 0
