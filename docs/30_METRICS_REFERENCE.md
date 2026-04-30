# 📊 30 · Metrics Reference

## 📑 Table of Contents

- [⚡ TL;DR](#-tldr)
- [📦 Collected Metrics](#-collected-metrics)
- [🔍 Failure Mode Taxonomy](#-failure-mode-taxonomy)
- [📋 Acting on the Failure Distribution](#-acting-on-the-failure-distribution)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🗺️ Next Steps](#️-next-steps)

---

## ⚡ TL;DR

- Observer collects **8 quantitative metrics** per episode.
- A **6-class priority rule chain** classifies failure type per episode — no training data required, works from checkpoint zero.
- Reading the failure distribution tells you **what to fix**, not just that something is broken.

---

## 📦 Collected Metrics

All per-episode aggregated metrics stored in `metrics.json`:

| 📏 Metric | 📐 Unit | 💬 Interpretation | 🚨 When it's bad |
|:---|:---:|:---|:---|
| `success_rate` | % | Episode success rate | Basic task failure |
| `contact_force_rms` | N | Fingertip RMS force — lower = more stable grasp | Grip too strong or unstable |
| `joint_velocity_rms` | rad/s | Jerk proxy — spikes indicate control instability | Near singularity, hardware risk |
| `slip_events_per_episode` | count | Tactile slip count — critical for hardware safety | Fingertip actuator wear |
| `mean_episode_length` | steps | Short episodes may indicate early failure | Grasp initialization failure |
| `object_pos_error_mm` | mm | Final position deviation from goal | Task-level failure |
| `object_rot_error_deg` | deg | Final rotation deviation from goal | Task-level failure |
| `energy_J` | J | Joint torque × velocity integral — efficiency proxy | Battery drain, thermal overload |

> 💡 `contact_force_rms`, `joint_velocity_rms`, and `energy_J` require `contact_forces` and `joint_velocities` to be present in `episodes.json`. Empty arrays result in those metrics being recorded as `null`.

---

## 🔍 Failure Mode Taxonomy

Episodes are classified by a **priority-ordered rule chain** — the first matching rule wins:

| Priority | 🏷️ Mode | 📋 Rule | ⚡ Hardware Implication |
|:---:|:---|:---|:---|
| 1 | `early_drop` | Episode length < 50 steps | Grasp initialization failure |
| 2 | `singularity_hit` | Max joint velocity > 5 rad/s | Actuator overload risk |
| 3 | `late_slip` | Slip event count ≥ 3 | Progressive grasp degradation |
| 4 | `contact_loss` | Tail-window mean force < 0.01 N | Loss of fingertip engagement |
| 5 | `repose_failure` | Final pose error > threshold | Task-level failure despite stable grasp |
| 6 | `timeout` | Max steps reached, no other rule matched | Policy too slow or stuck |

> 💡 **Advantage of rule-based classification:** immediately usable from checkpoint zero with no training data.
> Decisions are fully explainable — "classified as `late_slip` because rule 3 matched."

### Reading the failure distribution

Example `failure_distribution` field from `metrics.json`:

```json
{
  "failure_distribution": {
    "early_drop": 0.04,
    "singularity_hit": 0.02,
    "late_slip": 0.22,
    "contact_loss": 0.10,
    "repose_failure": 0.18,
    "timeout": 0.44
  },
  "dominant_failure_mode": "timeout"
}
```

---

## 📋 Acting on the Failure Distribution

| 🏷️ Dominant failure mode | 🔧 Recommended action |
|:---|:---|
| High `late_slip` share | Increase the **slip penalty** in the reward function |
| High `early_drop` share | Revisit grasp initialization / curriculum design |
| High `singularity_hit` share | Add joint velocity limits / singularity avoidance reward |
| High `contact_loss` share | Strengthen the contact force reward |
| High `repose_failure` share | Adjust the goal pose error reward |
| Overwhelming `timeout` share | Simplify the curriculum or improve exploration |

> 💡 **Example:** If 60% of failures are classified as `late_slip`,
> increase the **slip penalty** in the reward function —
> not tune the grasp initialization curriculum.
> The classifier tells you *what to fix*, not just *that something is broken*.

---

## 🛠️ Troubleshooting

**`failure_distribution` is empty or all episodes are `unknown`**

→ `episodes.json` is likely missing or empty.
```bash
cat eval_results/*/episodes.json | python -m json.tool | head -20
```
Check that `slip_count`, `length`, and `joint_velocities` fields are present.

**`contact_force_rms` or `energy_J` is null**

→ `contact_forces` or `joint_velocities` in `episodes.json` are empty arrays.
Check that the eval script is collecting per-step RMS values.

**`slip_events_per_episode` is always 0**

→ The environment does not detect tactile / contact signal transitions.
See `sharpa_wave_env.py` for an example: the `eval/slip_detected` key detects contact signal transitions.

---

## 🗺️ Next Steps

| Document | Content |
|:---|:---|
| [`31_CHECKPOINT_RANKING.md`](./31_CHECKPOINT_RANKING.md) | Multi-objective ranking + state coverage heatmap |
| [`20_INTEGRATION_CONTRACT.md`](./20_INTEGRATION_CONTRACT.md) | `episodes.json` schema details |
| [`10_ARCHITECTURE.md`](./10_ARCHITECTURE.md) | Where `FailureModeClassifier` lives in the codebase |
