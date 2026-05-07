"""
observer/brand.py
=================
OBSERVER brand identity — ASCII art, sigils, and color constants.

Project story:
    In StarCraft, the Observer is a cloaked Protoss sensor drone.
    It drifts unseen across the battlefield, watching every unit,
    piercing through cloaking fields that would fool any other eye.
    It asks nothing. It judges nothing. It simply sees — and reports.

    This project carries the same mandate.

    Dozens of checkpoints emerge from training runs — each one a policy
    that learned something, or failed to. Left unexamined, they sit in
    the dark: numbers without meaning, .pth files without story.

    OBSERVER deploys silently.
    It scans every episode, every fingertip contact trace, every joint
    velocity spike. It finds the failures that success rate hides.
    It renders the full picture — ranked, charted, and ready to act on.

    The fog of war lifts. The battlefield becomes legible.

Usage:
    from observer.brand import print_banner, SIGIL, rule, log

    print_banner()
    print(f"{SIGIL} Evaluation complete")
"""

# ── ANSI color codes ──────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
PURPLE = "\033[95m"
RED    = "\033[91m"

# ── Inline sigil — use in log lines and CLI prompts ───────────────────
#
#   [OBS]  (bold purple)
#
SIGIL = f"{PURPLE}{BOLD}[OBS]{RESET}"

# ── Full startup banner ───────────────────────────────────────────────
#
#  Visual language:
#    · Angular frame     — Protoss crystalline geometry (◆ corners, ━ ┃ edges)
#    · Central sensor    — ◉ the cloaked Observer sigil
#    · Checkpoint scan   — training runs scrutinized, reported, ranked
#
#  Checkpoint convergence flow:
#
#    ●   ●   ●   ●   ●   ← checkpoints (training runs)
#    │   │   │   │   │
#    └───┴───┼───┴───┘   ← all scanned by Observer
#            │
#         OBSERVER       ← failure analysis · ranking · report
#
BANNER = f"""{PURPLE}{BOLD}
  {RESET}{PURPLE}◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆{BOLD}
  ┃                                            ┃
  ┃          {WHITE}{BOLD}◉   O B S E R V E R   ◉          {PURPLE}{BOLD} ┃
  ┃                                            ┃
  ┃  {RESET}{YELLOW}All policies watched. No failure hidden.{PURPLE}{BOLD}  ┃
  ┃                                            ┃
  {RESET}{PURPLE}◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆{BOLD}
{RESET}"""

# ── Checkpoint convergence flow diagram (standalone) ──────────────────
FLOW = (
    f"{DIM}{PURPLE}  ●  ●  ●  ●  ●{RESET}  "
    f"{DIM}← checkpoints (training runs){RESET}\n"
    f"{PURPLE}  │  │  │  │  │{RESET}\n"
    f"{PURPLE}  └──┴──┼──┴──┘{RESET}  "
    f"{DIM}← all scanned by Observer{RESET}\n"
    f"{PURPLE}        │{RESET}\n"
    f"  {WHITE}{BOLD}   OBSERVER{RESET}    "
    f"{DIM}← failure analysis · ranking · report{RESET}\n"
)

# ── Version ───────────────────────────────────────────────────────────
VERSION = "0.1.0"
VERSION_STRING = f"{PURPLE}{BOLD}OBSERVER{RESET} {DIM}v{VERSION}{RESET}"


# ── Public functions ──────────────────────────────────────────────────
def print_banner() -> None:
    """Print the full OBSERVER startup banner."""
    print(BANNER)


def print_flow() -> None:
    """Print the checkpoint convergence flow diagram."""
    print(FLOW)


def rule(title: str = "", width: int = 54) -> str:
    """Return a styled horizontal rule with an optional centered title."""
    if title:
        pad = (width - len(title) - 2) // 2
        line = f"{'─' * pad} {title} {'─' * (width - len(title) - 2 - pad)}"
    else:
        line = "─" * width
    return f"{PURPLE}{line}{RESET}"


def log(msg: str, level: str = "info") -> str:
    """
    Return a formatted log prefix line.

    Parameters
    ----------
    msg   : message text
    level : "info" | "ok" | "warn" | "error"
    """
    icons = {
        "info":  f"{PURPLE}[OBS]{RESET}",
        "ok":    f"{GREEN}[OBS]{RESET}",
        "warn":  f"{YELLOW}[OBS]{RESET}",
        "error": f"{RED}[OBS]{RESET}",
    }
    prefix = icons.get(level, icons["info"])
    return f"{prefix} {msg}"


if __name__ == "__main__":
    print_banner()
    print()
    print_flow()
    print()
    print(f"  Sigil   : {SIGIL}")
    print(f"  Version : {VERSION_STRING}")
    print()
    print(rule("Scan Complete"))
    print()
    print(log("3 checkpoints evaluated", "info"))
    print(log("Best checkpoint deployed → rank01__model_5000.pth", "ok"))
    print(log("2 high-risk pose zones detected", "warn"))
    print(log("Isaac subprocess failed (exit 139)", "error"))
