# 🏆 31 · Checkpoint Ranking & State Coverage

## 📑 Table of Contents

- [⚡ TL;DR](#-tldr)
- [🏆 Multi-Objective Checkpoint Ranking](#-multi-objective-checkpoint-ranking)
- [🗺️ State Coverage Analysis](#️-state-coverage-analysis)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🗺️ Next Steps](#️-next-steps)

---

## ⚡ TL;DR

- Selecting checkpoints by `success_rate` alone can surface policies that are dangerous to hardware.
- `CheckpointSelector` computes a **weighted sum** of success rate, slip, energy, and pose error to rank checkpoints.
- `StateCoverageAnalyzer` visualizes **where** the policy is weak via a heatmap over the roll × pitch pose space.

---

## 🏆 Multi-Objective Checkpoint Ranking

### Why success_rate alone is not enough

```
success_rate = 0.91   ← looks great on paper
```

Hidden underneath: 3 slip events per episode, 5× energy consumption, joint velocity spikes.
Deploying such a policy on hardware rapidly wears down actuators.

### Scoring formula

$$\text{Score} = w_{sr} \cdot \text{SR}_{\text{norm}} - w_{\text{slip}} \cdot \text{slip}_{\text{norm}} - w_{\text{energy}} \cdot E_{\text{norm}} - w_{\text{pos}} \cdot \text{pos\_err}_{\text{norm}}$$

Each metric is normalized by the min–max range across the evaluation batch.

### Presets

| 🎛️ Preset | 🎯 When to use | Characteristics |
|:---|:---|:---|
| `balanced` | Everyday experiment comparison | All metrics weighted evenly |
| `hardware_safe` | Pre-deployment selection | Heavy penalties on slip and energy |
| `performance_first` | Ablation studies | Prioritizes raw task success rate |

### Running

```bash
# rank the full directory with hardware_safe preset + deploy top-2
python eval_runner.py --checkpoint_dir runs/ \
    --auto_select --select_weights hardware_safe --deploy_top_k 2
```

### Output

```
📁 eval_results/best/
  ├── 🥇 rank01__model_6000.pth  →  (symlink to original)
  ├── 🥈 rank02__model_4000.pth
  └── 📋 selection_meta.json
```

`selection_meta.json` records each checkpoint's score and individual metric values.

---

## 🗺️ State Coverage Analysis

> 💬 *"A policy with 90% success rate that only works for 30% of the initial pose space
> is not a good policy — it's a brittle one."

### How it works

`StateCoverageAnalyzer` bins episodes by initial object pose (roll × pitch) and visualizes
**where** in pose space the policy breaks down.

### Output files

| 🖼️ File | 📋 Description | 🎯 Use |
|:---|:---|:---|
| `success_heatmap.png` | 2D success rate over roll × pitch bins | Identify high-risk pose zones |
| `coverage_scatter.png` | Per-episode scatter colored by failure mode | Visualize failure clustering |
| `pose_histogram.png` | Roll / pitch / yaw sampling distribution | Verify curriculum coverage |
| `coverage_stats.json` | Worst zone coordinates + uniformity score | Input for next curriculum design |

### Reading the heatmap

🔴 **Red zones = next curriculum targets.**

If failures concentrate in roll ∈ [30°, 60°] → sample that region more heavily in the next training run.

```bash
# inspect coverage stats directly
cat eval_results/*/coverage/coverage_stats.json | python -m json.tool
```

### Requirements

`StateCoverageAnalyzer` requires `init_roll_deg` and `init_pitch_deg` from `episodes.json`.
The heatmap will not be generated if these fields are missing.

---

## 🛠️ Troubleshooting

**`coverage/` directory is empty or no heatmap generated**

→ `init_roll_deg` / `init_pitch_deg` are missing from `episodes.json`.
Check that the eval script populates the initial object pose in `EpisodeStats`.

**All checkpoints receive the same score**

→ Only one checkpoint is being compared — normalization is meaningless with a single value. Compare at least 3.
Or all metric values are identical — check whether the eval script is returning a fixed value.

**Certain metrics are `null` in `selection_meta.json`**

→ `contact_forces` / `joint_velocities` are empty arrays in the episodes.
`energy_J` and `joint_velocity_rms` calculations are skipped and excluded from ranking.

---

## 🗺️ Next Steps

| Document | Content |
|:---|:---|
| [`30_METRICS_REFERENCE.md`](./30_METRICS_REFERENCE.md) | Collected metrics + failure taxonomy |
| [`00_PRINCIPLES.md`](./00_PRINCIPLES.md) | Why multi-objective ranking is necessary |
| [`ko/01_INTRO.md`](./ko/01_INTRO.md) | Korean onboarding (includes preset selection guide) |
