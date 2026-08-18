#!/usr/bin/env python3
"""Plot before/after benchmark comparison after prompt improvements."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# DATA: Before (old prompts) vs After (new prompts with inference rules)
# ═══════════════════════════════════════════════════════════════════

# --- Agent-level metrics ---
agents = ['BiologicalAgent', 'TechnicalAgent', 'ExperimentalDesignAgent', 'Overall']
short_names = ['Biological', 'Technical', 'Experimental\nDesign', 'Overall\n(macro-avg)']

before = {
    'precision': [0.8081, 0.9574, 1.0000, 0.9219],
    'recall':    [0.8081, 0.7995, 0.7074, 0.7717],
    'f1':        [0.8081, 0.8714, 0.8286, 0.8361],
}
after = {
    'precision': [0.8158, 0.9653, 1.0000, 0.9270],
    'recall':    [0.8110, 0.9188, 0.7031, 0.8110],
    'f1':        [0.8134, 0.9415, 0.8256, 0.8602],
}

# --- Per-field metrics (before → after) ---
field_data = {
    # Biological fields
    'species':       {'before': 0.7255, 'after': 0.8970, 'agent': 'Bio'},
    'organ':         {'before': 0.6179, 'after': 0.5870, 'agent': 'Bio'},
    'cell_type':     {'before': 0.5635, 'after': 0.5004, 'agent': 'Bio'},
    'cell_line':     {'before': 0.7883, 'after': 0.7921, 'agent': 'Bio'},
    'disease':       {'before': 0.6273, 'after': 0.6341, 'agent': 'Bio'},
    # Technical fields
    'instrument':    {'before': 0.7948, 'after': 0.7896, 'agent': 'Tech'},
    'cleavage_agent':{'before': 0.8390, 'after': 0.8409, 'agent': 'Tech'},
    'label':         {'before': 0.4270, 'after': 0.7852, 'agent': 'Tech'},
    'fragmentation': {'before': 0.7387, 'after': 0.7394, 'agent': 'Tech'},
    'collision_e':   {'before': 0.7354, 'after': 0.7264, 'agent': 'Tech'},
    # Experimental fields
    'replicates':    {'before': 0.6968, 'after': 0.6943, 'agent': 'Exp'},
    'fractions':     {'before': 0.4536, 'after': 0.4439, 'agent': 'Exp'},
    'factor_value':  {'before': 0.6788, 'after': 0.6836, 'agent': 'Exp'},
}

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Agent-level P/R/F1 Before vs After
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Benchmark: Before vs After Prompt Improvements\n(107 PXDs, Llama-4-Scout vs SDRF Ground Truth)',
             fontsize=14, fontweight='bold')

metrics_names = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1 Score']
colors_before = ['#bdc3c7', '#e74c3c', '#95a5a6']
colors_after = ['#3498db', '#2ecc71', '#27ae60']

for idx, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
    ax = axes[idx]
    x = np.arange(len(agents))
    width = 0.35

    bars1 = ax.bar(x - width/2, before[metric], width, label='Before', color='#e74c3c', alpha=0.7, edgecolor='white')
    bars2 = ax.bar(x + width/2, after[metric], width, label='After', color='#2ecc71', alpha=0.85, edgecolor='white')

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#c0392b')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#27ae60', fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig('benchmark_data/reports/plots/before_after_agents.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: benchmark_data/reports/plots/before_after_agents.png')

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Per-field Avg Score Before vs After (horizontal bar)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

fields = list(field_data.keys())
before_scores = [field_data[f]['before'] for f in fields]
after_scores = [field_data[f]['after'] for f in fields]
deltas = [a - b for a, b in zip(after_scores, before_scores)]

# Sort by delta (biggest improvement first)
sorted_idx = np.argsort(deltas)
fields_sorted = [fields[i] for i in sorted_idx]
before_sorted = [before_scores[i] for i in sorted_idx]
after_sorted = [after_scores[i] for i in sorted_idx]
deltas_sorted = [deltas[i] for i in sorted_idx]
agents_sorted = [field_data[fields[i]]['agent'] for i in sorted_idx]

y = np.arange(len(fields_sorted))
height = 0.35

bars1 = ax.barh(y - height/2, before_sorted, height, label='Before', color='#e74c3c', alpha=0.6)
bars2 = ax.barh(y + height/2, after_sorted, height, label='After', color='#2ecc71', alpha=0.85)

# Delta annotations
for i, (b, a, d) in enumerate(zip(before_sorted, after_sorted, deltas_sorted)):
    color = '#27ae60' if d > 0.01 else '#e74c3c' if d < -0.01 else '#7f8c8d'
    sign = '+' if d > 0 else ''
    ax.text(max(a, b) + 0.02, i, f'{sign}{d:.3f}', va='center', fontsize=9,
            fontweight='bold', color=color)

# Agent labels on right
for i, agent in enumerate(agents_sorted):
    ax.text(1.12, i, agent, va='center', fontsize=8, color='#7f8c8d', style='italic',
            transform=ax.get_yaxis_transform())

ax.set_xlabel('Average Match Score', fontsize=12)
ax.set_title('Per-Field Score: Before vs After Prompt Improvements\n(Sorted by improvement Δ)',
             fontsize=13, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(fields_sorted, fontsize=10)
ax.set_xlim(0, 1.05)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig('benchmark_data/reports/plots/before_after_fields.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: benchmark_data/reports/plots/before_after_fields.png')

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Overall F1 comparison (single clean chart)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ['Precision', 'Recall', 'F1']
before_overall = [0.9219, 0.7717, 0.8361]
after_overall = [0.9270, 0.8110, 0.8602]

x = np.arange(len(metrics))
width = 0.3

bars1 = ax.bar(x - width/2, before_overall, width, label='Before (old prompts)',
               color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, after_overall, width, label='After (inference rules)',
               color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1.5)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}',
            ha='center', va='bottom', fontsize=14, color='#c0392b')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}',
            ha='center', va='bottom', fontsize=14, color='#27ae60', fontweight='bold')

ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Overall Benchmark: Before vs After\n107 PXDs · Llama-4-Scout vs SDRF',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

# Delta annotations
for i, (b, a) in enumerate(zip(before_overall, after_overall)):
    d = a - b
    ax.annotate(f'+{d:.3f}', xy=(i + width/2, a + 0.05),
                fontsize=10, color='#2980b9', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('benchmark_data/reports/plots/before_after_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: benchmark_data/reports/plots/before_after_overall.png')

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Match type distribution Before vs After
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
before_counts = {
    'EXACT': 64+60+25, 'NORMALIZED': 17+61+0, 'SEMANTIC': 131+163+114, 'NO_MATCH': 132+110+90
}
# After
after_counts = {
    'EXACT': 105+56+26, 'NORMALIZED': 23+62+0, 'SEMANTIC': 103+212+112, 'NO_MATCH': 113+64+91
}

labels = ['Exact', 'Normalized', 'Semantic', 'No Match']
colors = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c']

for ax_idx, (title, counts) in enumerate([('Before (Old Prompts)', before_counts),
                                            ('After (Inference Rules)', after_counts)]):
    ax = axes[ax_idx]
    vals = list(counts.values())
    total = sum(vals)
    wedges, texts, autotexts = ax.pie(vals, labels=labels, colors=colors,
                                       autopct=lambda p: f'{p:.1f}%\n({int(p*total/100)})',
                                       startangle=90, textprops={'fontsize': 9})
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f'{title}\n(n={total})', fontsize=12, fontweight='bold')

plt.suptitle('Match Type Distribution: Before vs After', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('benchmark_data/reports/plots/before_after_match_types.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: benchmark_data/reports/plots/before_after_match_types.png')

print('\n✅ All 4 comparison plots generated!')
