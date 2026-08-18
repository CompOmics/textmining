#!/usr/bin/env python3
"""Plot current benchmark metrics: one per agent + one overall."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'benchmark_data/reports/plots'

# ── Per-field data (new results) ──
bio_fields   = ['species', 'organ', 'cell_type', 'cell_line', 'disease']
bio_scores   = [0.897, 0.587, 0.500, 0.792, 0.634]
bio_match    = [0.925, 0.506, 0.410, 0.776, 0.617]

tech_fields  = ['instrument', 'cleavage_agent', 'label', 'fragmentation', 'collision_energy']
tech_scores  = [0.790, 0.841, 0.785, 0.739, 0.726]
tech_match   = [0.776, 0.925, 0.841, 0.710, 0.692]

exp_fields   = ['replicates', 'fractions', 'factor_value']
exp_scores   = [0.694, 0.444, 0.684]
exp_match    = [0.685, 0.462, 0.667]

# ── Agent-level aggregates ──
agent_names = ['Biological', 'Technical', 'Exp. Design']
agent_p     = [0.816, 0.965, 1.000]
agent_r     = [0.811, 0.919, 0.703]
agent_f1    = [0.813, 0.942, 0.826]

# ═══════════════════════════════════════════════════════
# Helper: per-agent bar chart
# ═══════════════════════════════════════════════════════
def plot_agent(fields, scores, match_rates, title, filename, color_main, color_sec):
    fig, ax = plt.subplots(figsize=(max(7, len(fields)*1.5), 5))
    x = np.arange(len(fields))
    w = 0.35
    b1 = ax.bar(x - w/2, scores, w, label='Avg Score', color=color_main, alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, match_rates, w, label='Match Rate', color=color_sec, alpha=0.75, edgecolor='white')
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_main)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=10, color=color_sec)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fields, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = f'{OUT}/{filename}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# 1) Biological
plot_agent(bio_fields, bio_scores, bio_match,
           'BiologicalAgent — Per-Field Metrics (107 PXDs)',
           'biological_metrics.png', '#2980b9', '#85c1e9')

# 2) Technical
plot_agent(tech_fields, tech_scores, tech_match,
           'TechnicalAgent — Per-Field Metrics (107 PXDs)',
           'technical_metrics.png', '#27ae60', '#82e0aa')

# 3) Experimental Design
plot_agent(exp_fields, exp_scores, exp_match,
           'ExperimentalDesignAgent — Per-Field Metrics (107 PXDs)',
           'experimental_metrics.png', '#8e44ad', '#c39bd3')

# ═══════════════════════════════════════════════════════
# 4) Overall + per-agent P/R/F1
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
labels = agent_names + ['Overall']
p_vals = agent_p + [0.927]
r_vals = agent_r + [0.811]
f1_vals = agent_f1 + [0.860]

x = np.arange(len(labels))
w = 0.25

b1 = ax.bar(x - w, p_vals, w, label='Precision', color='#3498db', alpha=0.85, edgecolor='white')
b2 = ax.bar(x,     r_vals, w, label='Recall',    color='#e67e22', alpha=0.85, edgecolor='white')
b3 = ax.bar(x + w, f1_vals, w, label='F1 Score',  color='#2ecc71', alpha=0.85, edgecolor='white')

for bars, col in [(b1, '#2c3e50'), (b2, '#2c3e50'), (b3, '#145a32')]:
    for bar in bars:
        h = bar.get_height()
        fw = 'bold' if bars is b3 else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight=fw, color=col)

ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Overall & Per-Agent Metrics — SDRF Benchmark\n107 PXDs · Llama-4-Scout',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Separator line before Overall
ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)

plt.tight_layout()
path = f'{OUT}/overall_agent_metrics.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

print('\n✅ All 4 plots generated!')
