"""Minimal publication-style panels from the completed paired comparison."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'three_levels/job_966182'
OUT = RUN / 'figures'
TEAL, GREY = '#167D9A', '#A0ADB8'
PROPERTY_FALLBACK = 'Abstract + methods\n(PRIDE-property fallback)'
PRIDE_PROTOCOLS = 'Abstract + methods\n+ PRIDE properties/protocols'


def amount(ax, s):
    t, n = s['totals'], s['projects']
    x = np.arange(2)
    for offset, prefix, label, color in [(-.19, 'old', PROPERTY_FALLBACK, GREY),
                                         (.19, 'new', PRIDE_PROTOCOLS, TEAL)]:
        vals = [t[prefix + '_' + k] / n for k in ['fields', 'values']]
        bars = ax.bar(x + offset, vals, width=.35, color=color, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val+.45, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, ['Populated fields', 'Distinct values'])
    ax.set_ylabel('Mean count per dataset')
    ax.set_ylim(0, 46)
    ax.set_yticks([0, 10, 20, 30])
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=1,
              handlelength=1, labelspacing=.7)


def overlap(ax, s):
    t = s['totals']
    vals = [t[k] for k in ['shared_fields', 'expanded_only_fields', 'earlier_only_fields']]
    ax.barh(range(3), np.array(vals)/1000, color=[TEAL, '#68ADBD', GREY], height=.55)
    ax.set_yticks(range(3), ['Shared', PRIDE_PROTOCOLS + '\nonly',
                           PROPERTY_FALLBACK + '\nonly'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel('Dataset–field pairs (×1,000)')
    for i, v in enumerate(vals):
        ax.text(v/1000+.5, i, f'{v:,}', va='center', fontsize=8)


def agreement(ax, s):
    cats = ['equivalent_value_sets', 'hierarchically_related_value_sets',
            'semantic_candidate_value_sets', 'partial_overlap', 'no_accepted_overlap_review']
    vals = [100*s['field_value_categories'][c]/s['totals']['shared_fields'] for c in cats]
    assert sum(s['field_value_categories'][c] for c in cats) == s['totals']['shared_fields']
    ax.barh(range(5), vals, height=.6,
            color=[TEAL, '#68ADBD', '#8A83AD', GREY, '#CC9964'])
    ax.set_yticks(range(5), ['Equivalent', 'Hierarchy-related', 'Semantic candidate',
                           'Partial overlap', 'No accepted overlap'])
    ax.invert_yaxis()
    ax.set_xlabel('Shared fields (%)')
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60])
    for i, v in enumerate(vals):
        ax.text(v+1, i, f'{v:.1f}', va='center', fontsize=8)


def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=.06,
                    facecolor='white')
    plt.close(fig)


def main():
    assert (RUN / 'COMPLETE').exists()
    s = json.loads((RUN / 'summary.json').read_text())
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    panels = [('metadata_amount', amount), ('field_overlap', overlap),
              ('value_agreement', agreement)]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(4, 3.1), layout='constrained')
        fn(ax, s)
        save(fig, name)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), layout='constrained',
                              gridspec_kw={'width_ratios': [1, 1.05, 1.15]})
    for letter, ax, (_, fn) in zip('ABC', axes, panels):
        fn(ax, s)
        ax.set_title(letter, loc='left', fontweight='bold', fontsize=11, pad=12)
    save(fig, 'three_level_comparison')
    with (OUT / 'plotted_data.csv').open('w') as h:
        writer = csv.writer(h)
        writer.writerow(['section', 'measure', 'value'])
        writer.writerows(['totals', k, v] for k,v in s['totals'].items())
        writer.writerows(['field_value_categories', k, v] for k,v in s['field_value_categories'].items())
        writer.writerow(['denominator', 'datasets', s['projects']])
    print(f'Saved combined figure and three individual panels: {OUT}', flush=True)


if __name__ == '__main__':
    main()
