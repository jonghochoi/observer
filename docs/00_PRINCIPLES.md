# 🧠 00 · Why Evaluation Matters — Design Principles

## Table of contents

- [TL;DR](#tldr)
- [The evaluation bottleneck](#the-evaluation-bottleneck)
- [What manual review actually gives you](#what-manual-review-actually-gives-you)
- [Why dexterous manipulation is especially unforgiving](#why-dexterous-manipulation-is-especially-unforgiving)
- [Core design decisions](#core-design-decisions)
- [Next steps](#next-steps)

---

## TL;DR

- Checkpoints accumulate fast, and manual comparison is riddled with bias and cognitive limits.
- `success_rate` alone cannot catch slip, overheating, or singularity hazards that can destroy hardware.
- Observer is designed around a subprocess-only boundary and is framework-agnostic by intent.

---

## The evaluation bottleneck

Dexterous manipulation research generates checkpoints faster than any human can meaningfully review them.
The failure point is not training — it is **analysis at scale**.

Scenarios that happen all the time:

---

🔬 **Scenario A — The Ablation Study**
> Reward shaping ablation: contact force weight × 4 conditions, slip penalty × 3 levels, curriculum schedule × 2 variants.
> 24 configurations. Each produces checkpoints at 1k, 3k, 5k, 10k steps.
> **96 checkpoints.** You have one afternoon.

🎲 **Scenario B — The Seed Sensitivity Check**
> Your policy looks great — but is it reproducible?
> You retrain with 10 different random seeds to characterize variance.
> Manually comparing convergence behavior across 10 runs is **cognitively impossible**.

🔁 **Scenario C — The Iterative Reward Design Loop**
> You tweak the reward function, retrain overnight, watch a few episodes in Isaac, feel uncertain,
> tweak again, repeat. After a week you have 30+ checkpoints from 8 reward variants.
> **Which reward actually worked, and why?** You've lost the thread.

🧪 **Scenario D — The Architecture Search**
> Policy network depth × 3, latent dimension × 4, with and without tactile observation.
> You train each configuration to convergence.
> The question is not just "what succeeded" — it is **"what failed, and where in pose space."**

---

## What manual review actually gives you

Every engineer who has sat through 20+ Isaac playback sessions knows this feeling:

| What you need to know | What manual review gives you |
|:---|:---|
| Is this policy statistically better? | *"It looked smoother today"* |
| Where in pose space does it fail? | *"It dropped the cube once"* |
| Is fingertip contact stable over time? | *"Seemed okay"* |
| Which of these 30 checkpoints is actually best? | *Whichever one you watched last* |
| Did the slip rate improve vs. last run? | *"Hard to say"* |

> **The three invisible biases of manual evaluation:**
>
> - **Selection bias** — you tend to watch the runs you already expect to be good
> - **Recency bias** — run #30 gets remembered more vividly than run #3
> - **Observer fatigue** — your 20th review session is never as sharp as your first

---

## Why dexterous manipulation is especially unforgiving

High-DOF dexterous hand control has failure modes that are **completely invisible to scalar success rate**:

```
success_rate = 0.91   ← looks great on paper
```

But underneath that number:

| Hidden problem | What success rate shows | Real-world consequence |
|:---|:---|:---|
| 3 slip events per episode | Nothing | Fingertip actuator wear within hours |
| Policy only works for roll ∈ [−20°, 20°] | Nothing | Brittle generalization — memorization, not learning |
| Joint velocity spikes near singularities | Nothing | Hardware safety hazard on real robot |
| Energy consumption 5× higher than baseline | Nothing | Battery drain, thermal overload |

> 💡 **The critical insight:** A policy that *looks* good in a 30-second viewport session
> may be actively dangerous when deployed on physical hardware.
> The only way to catch these issues is **systematic, quantitative analysis across many episodes.**

---

## Core design decisions

Non-obvious decisions in Observer's design:

### ── Subprocess-only boundary

Observer never instantiates the user's actor directly. It runs the eval script and record script as **subprocesses**, and only requires that they satisfy the contract (→ [`docs/20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md)).

Rationale:
- Completely avoids cross-framework compatibility issues
- No framework-specific imports in Observer code
- No version conflicts with the user's stack

### ── Framework-agnostic by design

Works with PPO / RSL-RL / CleanRL / any framework. The only requirement is that the eval script outputs `metrics.json` and `episodes.json` in the specified schema.

### ── Rule-based failure classification

`FailureModeClassifier` uses no training data. It is a priority-ordered rule chain — works from checkpoint zero. ML-based classifiers have a bootstrapping problem. Rule-based is immediately usable and decisions are fully explainable.

### ── Optional deps pattern

`tensorboard` and `opencv-python` are optional. When absent, the corresponding feature degrades gracefully. `observer.pipeline.experiment_tracker` detects availability at runtime.

### ── Metric-key fallbacks

If a field is missing from `metrics.json`, Observer skips the corresponding analysis and continues processing the rest. The only required fields are `checkpoint`, `num_episodes`, and `success_rate` (→ [`docs/20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) §1).

---

## Next steps

| Document | Content |
|:---|:---|
| [`10_ARCHITECTURE.md`](./10_ARCHITECTURE.md) | System structure, component map, output directory |
| [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) | Eval / record script contract |
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Collected metrics + failure mode taxonomy |
