# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` covers what OBSERVER does, how to install it, and the full CLI surface — read it first.
`docs/20_INTEGRATION_CONTRACT.md` is the contract observer expects from user-supplied eval/record scripts.
This file only documents the rules those two don't make explicit.

## Non-obvious conventions

- **Subprocess-only boundary.** The orchestrator never imports the user's RL stack, torch, or Isaac. Only files under `isaac/` may `import omni.*`, and even those degrade to mock mode when the import fails. If you're adding a torch/env/Isaac import outside `isaac/`, you're violating the design.
- **`docs/20_INTEGRATION_CONTRACT.md` ↔ orchestrator CLIs must stay in sync.** The two subprocess command shapes live in `observer/pipeline/orchestrator.py` (`_build_metrics_cmd`, `_build_record_cmd`). Any change to either must update `20_INTEGRATION_CONTRACT.md` in the same commit — adapter authors rely on it.
- **Preserve metric-key fallbacks.** `auto_select.py` and `report_generator.py` accept aliases (e.g. `energy_J_mean` ∨ `energy_J_per_episode`, `object_pos_error_mm_mean` ∨ `object_pose_error_mm`) so older user scripts keep working. Don't collapse them when refactoring.
- **Optional deps go through `try/except` + `_*_AVAILABLE` flags** at module load (see `observer/pipeline/experiment_tracker.py`, `observer/pipeline/state_coverage.py`, `observer/isaac/recorder.py`). Follow that pattern for new optional integrations rather than hard-importing.
- **Framework-specific knobs belong in `runtime.extra_eval_args` / `extra_record_args`**, forwarded verbatim to the subprocess. Don't add framework-specific fields to `RuntimeConfig` or the orchestrator.
- **`brand.py`** is the single source for ANSI styling and the banner. Don't sprinkle ANSI codes elsewhere.

## Things that surprise people

- The Python code lives under a wrapper `observer/` subdirectory (`observer/observer/eval_runner.py`, `observer/observer/pipeline/...`, etc.) so the package is discovered via `PYTHONPATH=<repo_root>` symmetrically with peer libraries (e.g. nexus's `nexus/nexus/...`). Invoke via the `observer` console script after `pip install -e .`, or as `python -m observer.eval_runner`.
- `--dry_run` is the closest thing to a test suite: it skips the metrics/record subprocesses and feeds `_dummy_metrics()` through the rest of the pipeline. Use it to exercise report/ranking/tracker code paths offline.
- One failed checkpoint never aborts a sweep — `_run_subprocess` errors are caught in `run_single` and recorded on `CheckpointResult.error_msg`.

## Docs Markdown style

All files under `docs/` and `README.md` follow these rules (established in nexus, applied here for consistency):

| Level | Format | Rule |
|:---:|---|---|
| H1 | `# 🔧 Document Title` | One thematic emoji + plain title. One H1 per doc. |
| H2 | `## Section title` | **No emoji ever.** Sentence case. Plain text. |
| H3 | `### ── Subsection title` | Prefix `── ` (U+2500 × 2 + space). No emoji. Sentence case. |
| H4 | `#### ▸ Sub-subsection title` | Prefix `▸` (U+25B8, BLACK RIGHT-POINTING SMALL TRIANGLE). Visually distinct from H3. |

**H3 vs H4 prefix characters:**
- **H3** uses `──` (U+2500 × 2) — matches the source-code section-divider convention.
- **H4** uses `▸` (U+25B8) — visually distinct from H3 to make nesting depth immediately apparent.

**Body emoji** — allowed in: tables (sparingly), blockquote callouts (`> 📖`, `> ⚠️`, `> 💡`, 🇰🇷 flag). Never in H2+ headers.

**Em dash `—`** for label/explanation joins in prose (not ` - `). Hyphen-minus `-` stays for compound words and CLI flags only.

**GitHub anchor caveats:**
- Both `──` (H3) and `▸` (H4) are stripped by GitHub but the trailing space is kept → **leading hyphen** in anchor:
  - `### ── Method selection` → anchor `#-method-selection`
  - `#### ▸ On local machine` → anchor `#-on-local-machine`
- Em dash in H2: `## Step 0 — Verify` → anchor `#step-0--verify` (double hyphen from em dash).
- TOC links to H3/H4 must include the leading hyphen; H2 links do not.

**Table column headers** — strip decorative emoji. Emoji allowed in table *cells* where they convey meaning.

**When adding a doc file under `docs/`:**
- [ ] `README.md` → "Documentation" table — add the new entry.
- [ ] `CLAUDE.md` → "Where to read more" — update if relevant to code changes.
- [ ] If the doc applies to Korean users, consider adding a `docs/ko/` entry.

## Code formatting

Target (configured in `pyproject.toml` under `[tool.ruff]`):

```
line-length  = 100
indent-width = 4   # spaces, never tabs
skip-magic-trailing-comma = true
```

Formatting command: `ruff format .`

## Comment & docstring style

Every Python module starts with a docstring using the unicode banner format:

```python
"""
observer/observer/pipeline/metrics_collector.py
======================================
Aggregates per-step episode data into the metrics.json schema.
"""
```

The `=` line length must match the path line length exactly.

**Section dividers inside a file:**

```python
# ── Public interface ─────────────────────────────────────────────────
```

Use `─` (U+2500 LIGHT HORIZONTAL). Pad the rule so the comment ends near column 76.
Number top-level sections in larger CLI scripts: `# ── 1. Argument parsing ──...`

**Rules:**
1. Use `─` (U+2500) — **never** `-` (hyphen-minus) or `=` for dividers. `grep -nE "^# (-{4,}|={4,})" file.py` must return nothing.
2. Use em dash `—` (U+2014) for label/explanation joins in docstrings and "why" comments.
   `make_logger — factory function`, `# Dirty tree detected — git patch saved`
3. ASCII tree chars in docstrings: `└──` `├──` (U+2514, U+251C).
4. Short "why" comments stay ASCII: `# MLflow hard limit per log_batch() call`.
5. New files: copy the header of the closest sibling instead of inventing a new layout.

**Note:** existing `.py` files are not yet conformant. New files and significant edits should follow the rule; a sweeping retro-conversion is a separate task.

## Commit message style

Derived from the nexus convention and observer's own commit history. Not vanilla Conventional Commits — stricter in several ways.

**Subject line format:**

```
<type>(<scope>): <verb> <description>
<type>: <verb> <description>           # scope optional for repo-wide changes
```

**Hard rules:**

1. **Start with an imperative verb** (most important). First word after the colon must be a verb in imperative mood.
   - ✅ `add`, `fix`, `remove`, `rename`, `move`, `split`, `clean up`, `clarify`, `standardize`,
     `harden`, `consolidate`, `pin`, `unify`, `apply`, `support`, `generalize`, `skip`, `treat`
   - ❌ Past tense (`added`, `fixed`), gerunds (`adding`), noun-first (`new feature for X`)

2. **`<type>`:** one of `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `deps`.

3. **`<scope>`:** lowercase, matches the module or area changed:
   `pipeline`, `orchestrator`, `metrics_collector`, `failure_classifier`, `state_coverage`,
   `auto_select`, `experiment_tracker`, `report`, `isaac`, `tactile`, `doctor`, `brand`,
   `eval_runner`, `setup`, `CLAUDE.md`, `docs`, `adapters`
   Omit scope for repo-wide changes.

4. **Description:** lowercase first letter, ≤ 72 chars total subject line, **no trailing period**.
   State *what*, not *why* (body carries the *why*).

5. **No `(#NN)` in subject** — GitHub appends the PR number automatically on squash-merge;
   adding it manually creates duplicates.

**Real examples from this repo:**

```
✅ feat: initial release — OBSERVER v0.1.0
✅ docs: restructure guide docs into numbered sequence with integration contract
✅ feat(pipeline): unify section dividers and docstring banners across pipeline modules
✅ fix(failure_classifier): handle missing slip_count field without aborting episode
✅ docs(adapters): add sharpa-rl-lab integration example
✅ chore: pin numpy<2.0 to avoid downstream ABI breakage

❌ Added new metric for energy                  # past tense, no type
❌ feat: new failure mode.                       # noun-first, trailing period
❌ fix(pipeline): Fixes missing slip_count.      # 3rd-person, capitalized, period
❌ update classifier                             # no type, vague verb
```

**Body (optional for trivial commits; required for multi-area or complex changes):**

1. Blank line between subject and body.
2. Wrap at ~72 columns. URLs and code blocks may exceed.
3. **Lead with the *why*** — one short paragraph stating problem/motivation before listing the *what*.
4. Lists: `-` bullets for parallel changes, `1.` `2.` for sequenced items.
5. Per-file groupings for larger commits: `observer/pipeline/orchestrator.py:` on its own line, then indented bullets.
6. Unicode dividers for large commits (`feat`/`refactor` touching many files):
   ```
   ── 1. Anchor text mismatches ───────────────────────────────────────────
   ```
   Use `─` (U+2500) — never `-` or `=`. Pad to ~72 chars. Number sections when multiple.
7. **Em dash `—`, not ` - `** for label/explanation joins in body prose.
8. **Backticks** around paths, identifiers, CLI flags, shell commands.

**Audit checklist before committing:**
- [ ] First word after `<type>(<scope>):` is imperative verb
- [ ] Description is lowercase, ≤ 72 chars total, no trailing period
- [ ] No `(#NN)` PR-number suffix in subject
- [ ] Body (if present) leads with *why*, wraps at ~72, uses `─` dividers, em dash `—` for joins

## When adding new features

### ── New metric

- [ ] `observer/pipeline/metrics_collector.py` — add field to `EpisodeStats` and aggregation logic
- [ ] `observer/pipeline/auto_select.py` — add key to scoring; add alias fallbacks for backward compat
- [ ] `observer/pipeline/report_generator.py` — add rendering for the new metric
- [ ] `docs/20_INTEGRATION_CONTRACT.md` — update `metrics.json` schema table
- [ ] `docs/30_METRICS_REFERENCE.md` — add row to the metrics table + interpretation

### ── New failure mode

- [ ] `observer/pipeline/failure_classifier.py` — insert rule at the correct priority in the chain
- [ ] `docs/30_METRICS_REFERENCE.md` — add row to the taxonomy table; update priority numbers
- [ ] `docs/31_CHECKPOINT_RANKING.md` — consider whether ranking weights need to account for it

### ── New framework adapter

- [ ] `docs/adapters/<framework>.md` — use `docs/adapters/sharpa.md` as the template
- [ ] `docs/21_ADAPTER_GUIDE.md` — add the new adapter to the worked-example list
- [ ] `README.md` → "Documentation" table — add the adapter doc entry

### ── New runtime config field

- [ ] `observer/configs/eval_config.py` — add field to the dataclass
- [ ] `observer/configs/eval_config.yaml` — add the key (commented out if optional)
- [ ] `doctor.py` — add validation if the field is user-supplied
- [ ] `README.md` → "Quick start" — update example config stanza if user-facing
- [ ] `docs/20_INTEGRATION_CONTRACT.md` — update if the field is forwarded to a subprocess

### ── New CLI flag on `eval_runner.py`

- [ ] `eval_runner.py` — add to `parse_args()`
- [ ] `README.md` → "Quick start" — add usage example
- [ ] `docs/10_ARCHITECTURE.md` → output structure section if the flag changes output layout

### ── New optional dependency

- [ ] Use `try/except` + `_*_AVAILABLE` flag at module load — see `experiment_tracker.py` for the pattern
- [ ] `pyproject.toml` — add under `[project.optional-dependencies]`
- [ ] `doctor.py` — check and surface availability at startup if user-visible

### ── New subprocess CLI flag (eval/record contract)

- [ ] `observer/pipeline/orchestrator.py` — `_build_metrics_cmd` and/or `_build_record_cmd`
- [ ] `docs/20_INTEGRATION_CONTRACT.md` — **same commit** — adapter authors depend on this being current

## Cross-cutting conventions

Items where multiple files must stay in lockstep — changing one without the others is a bug:

**Metric-key fallbacks** — `observer/pipeline/auto_select.py` and `observer/pipeline/report_generator.py` both
implement alias lookups (e.g. `energy_J_mean` ∨ `energy_J_per_episode`). Adding or removing an
alias in one file must be reflected in the other in the **same commit**.

**Subprocess CLI contract** — `observer/pipeline/orchestrator.py` (`_build_metrics_cmd`, `_build_record_cmd`)
and `docs/20_INTEGRATION_CONTRACT.md` are the two authoritative sources for the subprocess command
shape. Change them together, never separately.

**ANSI styling** — `brand.py` is the single source for color codes and the console banner.
No raw `\x1b[` or `\033[` sequences anywhere else in the codebase.

**`--dry_run` coverage** — `_dummy_metrics()` in `eval_runner.py` must cover every pipeline path
that a real run would exercise. When adding a new pipeline stage, extend `_dummy_metrics()` so
`--dry_run` still succeeds end-to-end.

## Where to read more

| Document | Content |
|:---|:---|
| [`docs/00_PRINCIPLES.md`](docs/00_PRINCIPLES.md) | Why evaluation matters; four bias types; five core design decisions |
| [`docs/10_ARCHITECTURE.md`](docs/10_ARCHITECTURE.md) | Component map, repository file tree, output directory layout, dependencies |
| [`docs/20_INTEGRATION_CONTRACT.md`](docs/20_INTEGRATION_CONTRACT.md) | Subprocess CLI shapes, `metrics.json` / `episodes.json` schemas, env instrumentation |
| [`docs/21_ADAPTER_GUIDE.md`](docs/21_ADAPTER_GUIDE.md) | Step-by-step guide for writing a new framework adapter |
| [`docs/30_METRICS_REFERENCE.md`](docs/30_METRICS_REFERENCE.md) | 8-metric table, 6-class failure taxonomy, acting on the distribution |
| [`docs/31_CHECKPOINT_RANKING.md`](docs/31_CHECKPOINT_RANKING.md) | Scoring formula, presets, state-coverage heatmaps |
| [`docs/adapters/sharpa.md`](docs/adapters/sharpa.md) | Complete sharpa-rl-lab integration example (use as adapter template) |
| [`docs/ko/01_INTRO.md`](docs/ko/01_INTRO.md) | 🇰🇷 Self-contained Korean onboarding — setup, first run, output interpretation |
