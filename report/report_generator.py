"""
report_generator.py
===================
Generates a self-contained HTML evaluation report from CheckpointResult objects.

Report sections:
  1. Summary cards       — key aggregate stats at a glance
  2. Comparison table    — per-checkpoint metrics with dominant failure mode column
  3. Performance charts  — Bar charts via Chart.js CDN (success rate, slip, force, energy)
  4. Failure mode pies   — per-checkpoint doughnut charts of failure distribution
  5. Coverage images     — success heatmap, scatter plot, pose histogram
  6. Video player        — inline <video> tags for recorded mp4 files
  7. Failed checkpoints  — error list for any checkpoint that did not complete
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("observer.report.report_generator")

# Failure mode display strings and colors (mirrors failure_classifier.py)
_FAILURE_LABELS = {
    "success":         "Success",
    "early_drop":      "Early Drop",
    "singularity_hit": "Singularity Hit",
    "late_slip":       "Late Slip",
    "contact_loss":    "Contact Loss",
    "repose_failure":  "Repose Failure",
    "timeout":         "Timeout",
    "unknown":         "Unclassified",
}

_FAILURE_COLORS = {
    "success":         "#22c55e",
    "early_drop":      "#ef4444",
    "singularity_hit": "#f97316",
    "late_slip":       "#a855f7",
    "contact_loss":    "#06b6d4",
    "repose_failure":  "#f59e0b",
    "timeout":         "#6b7280",
    "unknown":         "#374151",
}


class ReportGenerator:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    def generate(self, results: list) -> Path:
        """
        Parameters
        ----------
        results : list[CheckpointResult]

        Returns
        -------
        Path
            Path to the generated HTML report.
        """
        report_path = self.output_root / "eval_report.html"
        valid  = [r for r in results if r.success and r.metrics]
        failed = [r for r in results if not r.success]
        valid_sorted = sorted(
            valid, key=lambda r: r.metrics.get("success_rate", 0), reverse=True
        )
        html = self._render(valid_sorted, failed)
        report_path.write_text(html, encoding="utf-8")
        log.info(f"Report generated: {report_path}")
        return report_path

    # ── Top-level HTML assembly ───────────────────────────────────────
    def _render(self, valid: list, failed: list) -> str:
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_all = len(valid) + len(failed)
        avg_sr = (
            sum(r.metrics.get("success_rate", 0) for r in valid) / len(valid)
            if valid else 0
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Isaac Lab Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
{self._css()}
</head>
<body>
<h1>🤖 Isaac Lab Evaluation Report</h1>
<p class="meta">
  Generated: {ts} &nbsp;|&nbsp;
  Checkpoints: {n_all} &nbsp;|&nbsp;
  Evaluated: {len(valid)} &nbsp;|&nbsp;
  Failed: {len(failed)}
</p>

{self._summary_cards(valid, avg_sr)}

<h2>📊 Checkpoint Performance Comparison</h2>
<div class="card scroll-x">
{self._table(valid)}
</div>

<h2>📈 Performance Metrics</h2>
<div class="g2">
  <div class="chart-card"><canvas id="cSR"></canvas></div>
  <div class="chart-card"><canvas id="cSlip"></canvas></div>
</div>
<div class="g2" style="margin-top:14px">
  <div class="chart-card"><canvas id="cForce"></canvas></div>
  <div class="chart-card"><canvas id="cEnergy"></canvas></div>
</div>

<h2>🔍 Failure Mode Distribution</h2>
<div class="pie-grid">
{self._failure_pies(valid)}
</div>

{self._coverage_section(valid)}

<h2>🎬 Recorded Videos</h2>
<div class="video-grid">
{self._videos(valid)}
</div>

{self._failed_section(failed)}

{self._scripts(valid)}
</body>
</html>"""

    # ── CSS ───────────────────────────────────────────────────────────
    def _css(self) -> str:
        return """<style>
:root{
  --bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;
  --text:#e2e8f0;--muted:#8892a4;--accent:#6c63ff;
  --ok:#22c55e;--warn:#f59e0b;--danger:#ef4444;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:'Segoe UI',system-ui,sans-serif;
  padding:28px 32px;line-height:1.6;
}
h1{font-size:1.85rem;font-weight:700}
h2{
  font-size:1.1rem;font-weight:600;color:var(--accent);
  border-bottom:1px solid var(--border);
  padding-bottom:7px;margin:34px 0 14px;
}
h3{font-size:.9rem;font-weight:600;margin-bottom:6px}
.meta{color:var(--muted);font-size:.83rem;margin:4px 0 28px}
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:18px;margin-bottom:18px;
}
.scroll-x{overflow-x:auto}

/* Summary cards */
.sg{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:12px;margin-bottom:28px;
}
.sc{
  background:var(--card);border:1px solid var(--border);
  border-radius:9px;padding:15px;text-align:center;
}
.sv{font-size:1.75rem;font-weight:700;color:var(--accent);line-height:1.1}
.sl{font-size:.76rem;color:var(--muted);margin-top:4px}

/* Table */
table{width:100%;border-collapse:collapse;font-size:.86rem}
th{
  background:#1e2235;padding:9px 11px;text-align:left;
  color:var(--muted);font-weight:600;font-size:.73rem;
  text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;
}
td{padding:9px 11px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:hover td{background:rgba(108,99,255,.06)}
.r1{color:#ffd700;font-weight:700}
.r2{color:#c0c0c0;font-weight:700}
.r3{color:#cd7f32;font-weight:700}
.badge{
  display:inline-block;padding:2px 7px;
  border-radius:11px;font-size:.72rem;font-weight:600;
}
.bok   {background:rgba(34,197,94,.15);color:var(--ok)}
.bwarn {background:rgba(245,158,11,.15);color:var(--warn)}
.bdanger{background:rgba(239,68,68,.15);color:var(--danger)}
.fbadge{font-size:.71rem;padding:2px 7px;border-radius:9px;font-weight:600}

/* Chart grids */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:14px;
}
canvas{max-height:250px}

/* Failure pie grid */
.pie-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(250px,1fr));
  gap:14px;
}
.pie-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:13px;
}
.pie-card h3{
  font-size:.82rem;color:var(--muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.pie-card canvas{max-height:170px}
.dtag{
  font-size:.73rem;margin-top:6px;padding:3px 8px;
  border-radius:8px;display:inline-block;font-weight:600;
}

/* Coverage */
.cov-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
  gap:14px;
}
.cov-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:13px;
}
.cov-card img{width:100%;border-radius:5px;margin-top:6px}

/* Video */
.video-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
  gap:16px;
}
.vc{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:14px;
}
video{width:100%;border-radius:5px;background:#000}

/* Failed checkpoints */
.fl{list-style:none}
.fl li{
  padding:7px 11px;
  background:rgba(239,68,68,.08);
  border-left:3px solid var(--danger);
  border-radius:4px;margin-bottom:5px;font-size:.86rem;
}

@media(max-width:860px){
  .g2{grid-template-columns:1fr}
  .pie-grid{grid-template-columns:1fr 1fr}
}
</style>"""

    # ── Summary cards ─────────────────────────────────────────────────
    def _summary_cards(self, valid: list, avg_sr: float) -> str:
        if not valid:
            return ""
        best    = valid[0]
        best_sr = best.metrics.get("success_rate", 0)

        all_dom = [
            r.metrics.get("dominant_failure_mode", "")
            for r in valid
            if r.metrics.get("dominant_failure_mode", "")
               not in ("", "unknown", "success")
        ]
        top_fail  = max(set(all_dom), key=all_dom.count) if all_dom else "—"
        top_label = _FAILURE_LABELS.get(top_fail, top_fail)

        return f"""<div class="sg">
  <div class="sc">
    <div class="sv">{len(valid)}</div>
    <div class="sl">Checkpoints Evaluated</div>
  </div>
  <div class="sc">
    <div class="sv">{avg_sr*100:.1f}%</div>
    <div class="sl">Mean Success Rate</div>
  </div>
  <div class="sc">
    <div class="sv">{best_sr*100:.1f}%</div>
    <div class="sl">Best Success Rate</div>
  </div>
  <div class="sc">
    <div class="sv" style="font-size:.95rem">{best.checkpoint.stem[-18:]}</div>
    <div class="sl">Best Checkpoint</div>
  </div>
  <div class="sc">
    <div class="sv" style="font-size:.9rem">{top_label}</div>
    <div class="sl">Dominant Failure Mode</div>
  </div>
</div>"""

    # ── Comparison table ──────────────────────────────────────────────
    def _table(self, valid: list) -> str:
        rnk  = {1: "r1", 2: "r2", 3: "r3"}
        rows = []
        for i, r in enumerate(valid, 1):
            m    = r.metrics
            sr   = m.get("success_rate", 0)
            slip = m.get("slip_events_per_episode", 0)
            eng  = m.get("energy_J_mean", m.get("energy_J_per_episode", 0))
            pos  = m.get("object_pos_error_mm_mean", m.get("object_pose_error_mm", 0))
            cf   = m.get("contact_force_rms_mean", m.get("contact_force_rms", 0))
            epl  = m.get("mean_episode_length", 0)
            src  = "b" + ("ok" if sr >= .8 else "warn" if sr >= .5 else "danger")

            dom = m.get("dominant_failure_mode", "")
            if dom and dom not in ("unknown", "success", ""):
                lbl = _FAILURE_LABELS.get(dom, dom)
                col = _FAILURE_COLORS.get(dom, "#6b7280")
                dfrag = (
                    f'<span class="fbadge" '
                    f'style="background:{col}22;color:{col}">{lbl}</span>'
                )
            else:
                dfrag = '<span style="color:#4a5568">—</span>'

            rows.append(
                f'<tr>'
                f'<td class="{rnk.get(i, "")}">{i}</td>'
                f'<td style="font-size:.8rem">{r.checkpoint.name}</td>'
                f'<td><span class="badge {src}">{sr*100:.1f}%</span></td>'
                f'<td>{cf:.4f}</td>'
                f'<td>{slip:.2f}</td>'
                f'<td>{pos:.2f}</td>'
                f'<td>{eng:.3f}</td>'
                f'<td>{epl:.0f}</td>'
                f'<td>{dfrag}</td>'
                f'</tr>'
            )
        hdr = (
            "<table><thead><tr>"
            "<th>Rank</th><th>Checkpoint</th><th>Success Rate</th>"
            "<th>Contact RMS (N)</th><th>Slip/ep</th>"
            "<th>Pos Error (mm)</th><th>Energy (J)</th>"
            "<th>Ep Length</th><th>Dominant Failure</th>"
            "</tr></thead><tbody>"
        )
        return hdr + "\n".join(rows) + "</tbody></table>"

    # ── Failure mode doughnut charts ──────────────────────────────────
    def _failure_pies(self, valid: list) -> str:
        cards = []
        for r in valid:
            dist = r.metrics.get("failure_distribution", {})
            if not dist:
                continue
            dom   = r.metrics.get("dominant_failure_mode", "unknown")
            dlbl  = _FAILURE_LABELS.get(dom, dom)
            dcol  = _FAILURE_COLORS.get(dom, "#6b7280")
            sr    = r.metrics.get("success_rate", 0)
            cards.append(
                f'<div class="pie-card">'
                f'<h3 title="{r.checkpoint.name}">{r.checkpoint.stem[-24:]}</h3>'
                f"<canvas data-pie='{json.dumps(dist)}'></canvas>"
                f'<div style="margin-top:7px;display:flex;'
                f'justify-content:space-between;align-items:center">'
                f'<span class="dtag" '
                f'style="background:{dcol}22;color:{dcol}">{dlbl}</span>'
                f'<span style="font-size:.78rem;color:var(--muted)">'
                f'SR {sr*100:.1f}%</span>'
                f'</div></div>'
            )
        if not cards:
            return (
                '<p style="color:var(--muted)">'
                'No failure distribution data available '
                '(episode-level recording required).'
                '</p>'
            )
        return "\n".join(cards)

    # ── Coverage image section ────────────────────────────────────────
    def _coverage_section(self, valid: list) -> str:
        plot_titles = {
            "success_heatmap":  "Success Rate Heatmap (Roll x Pitch)",
            "coverage_scatter": "Episode Scatter by Failure Mode",
            "pose_histogram":   "Initial Pose Distribution",
        }
        cards = []
        for r in valid:
            plots = getattr(r, "coverage_plots", []) or []
            for p in plots:
                if not Path(p).exists():
                    continue
                try:
                    src = str(Path(p).relative_to(self.output_root))
                except ValueError:
                    src = str(p)
                title = plot_titles.get(Path(p).stem, Path(p).stem)
                cards.append(
                    f'<div class="cov-card">'
                    f'<h3 style="color:var(--muted);font-size:.8rem">'
                    f'{r.checkpoint.stem[-20:]} — {title}</h3>'
                    f'<img src="{src}" alt="{title}" loading="lazy">'
                    f'</div>'
                )
        if not cards:
            return ""
        return (
            '<h2>🗺️ Initial Pose Coverage Analysis</h2>'
            f'<div class="cov-grid">{"".join(cards)}</div>'
        )

    # ── Video section ─────────────────────────────────────────────────
    def _videos(self, valid: list) -> str:
        cards = []
        for r in valid:
            files = []
            cv = getattr(r, "combined_video", None)
            if cv and Path(cv).exists():
                files = [(cv, "Grid View (All Angles)")]
            else:
                vps = getattr(r, "video_paths", []) or []
                files = [(p, Path(p).stem) for p in vps if Path(p).exists()][:3]

            sr  = r.metrics.get("success_rate", 0) if r.metrics else 0
            src_cls = "b" + ("ok" if sr >= .8 else "warn" if sr >= .5 else "danger")

            for vpath, label in files:
                try:
                    src = str(Path(vpath).relative_to(self.output_root))
                except ValueError:
                    src = str(vpath)
                cards.append(
                    f'<div class="vc">'
                    f'<h3>{r.checkpoint.name} '
                    f'<span style="color:var(--muted);font-weight:400">— {label}</span>'
                    f'<span class="badge {src_cls}" style="float:right">'
                    f'{sr*100:.1f}%</span></h3>'
                    f'<video controls preload="metadata">'
                    f'<source src="{src}" type="video/mp4">'
                    f'Your browser does not support HTML5 video.'
                    f'</video></div>'
                )
        if not cards:
            return '<p style="color:var(--muted);padding:8px">No videos available.</p>'
        return "\n".join(cards)

    # ── Failed checkpoint list ────────────────────────────────────────
    def _failed_section(self, failed: list) -> str:
        if not failed:
            return ""
        items = "".join(
            f'<li>❌ {r.checkpoint.name} — {r.error_msg or "Unknown error"}</li>'
            for r in failed
        )
        return (
            '<h2>⚠️ Failed Evaluations</h2>'
            f'<div class="card"><ul class="fl">{items}</ul></div>'
        )

    # ── JavaScript (Chart.js initialization) ──────────────────────────
    def _scripts(self, valid: list) -> str:
        labels = [r.checkpoint.stem[-22:] for r in valid]

        def _g(m, *keys):
            for k in keys:
                if k in m:
                    return m[k]
            return 0

        cd = {
            "labels": labels,
            "sr":     [_g(r.metrics, "success_rate") for r in valid],
            "slip":   [_g(r.metrics, "slip_events_per_episode") for r in valid],
            "force":  [_g(r.metrics, "contact_force_rms_mean", "contact_force_rms")
                       for r in valid],
            "energy": [_g(r.metrics, "energy_J_mean", "energy_J_per_episode")
                       for r in valid],
        }
        pie_colors = json.dumps(_FAILURE_COLORS)
        pie_labels = json.dumps(_FAILURE_LABELS)

        return f"""<script>
const CD = {json.dumps(cd)};
const DK = {{color:'#8892a4', grid:'#2a2d3e'}};
const PAL = [
  '#6c63ff','#22c55e','#f59e0b','#ef4444',
  '#06b6d4','#a855f7','#f97316','#14b8a6'
];

function bar(id, label, data, yOpts) {{
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, {{
    type: 'bar',
    data: {{
      labels: CD.labels,
      datasets: [{{
        label,
        data,
        backgroundColor: PAL.map(c => c + '99'),
        borderColor: PAL,
        borderWidth: 1.5,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: DK.color }} }} }},
      scales: {{
        x: {{ ticks: {{ color: DK.color, maxRotation: 40 }}, grid: {{ color: DK.grid }} }},
        y: {{ ticks: {{ color: DK.color }}, grid: {{ color: DK.grid }}, ...(yOpts || {{}}) }}
      }}
    }}
  }});
}}

bar('cSR',    'Success Rate (%)',     CD.sr.map(v => (v * 100).toFixed(1)), {{min:0, max:100}});
bar('cSlip',  'Slip Events / ep',     CD.slip);
bar('cForce', 'Contact Force RMS (N)',CD.force);
bar('cEnergy','Energy Consumption (J)',CD.energy);

// Failure mode doughnut charts
const FC = {pie_colors};
const FL = {pie_labels};

document.querySelectorAll('[data-pie]').forEach(canvas => {{
  const raw  = JSON.parse(canvas.dataset.pie);
  const keys = Object.keys(raw).filter(k => raw[k] > 0.005);
  new Chart(canvas, {{
    type: 'doughnut',
    data: {{
      labels:   keys.map(k => FL[k] || k),
      datasets: [{{
        data:            keys.map(k => (raw[k] * 100).toFixed(1)),
        backgroundColor: keys.map(k => FC[k] || '#374151'),
        borderWidth: 0,
      }}]
    }},
    options: {{
      responsive: true,
      cutout: '58%',
      plugins: {{
        legend: {{
          position: 'right',
          labels: {{ color: '#8892a4', font: {{ size: 10 }}, boxWidth: 11 }}
        }}
      }}
    }}
  }});
}});
</script>"""
