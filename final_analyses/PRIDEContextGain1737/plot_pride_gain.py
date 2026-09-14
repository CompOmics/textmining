"""Plot observed PRIDE-supported additions; no inference or scoring."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/job_965823'
OUT = ROOT / 'figures'


def main():
    summary = json.loads((RUN / 'summary.json').read_text())
    projects = {}
    with (RUN / 'field_changes.csv').open() as handle:
        for row in csv.DictReader(handle):
            projects.setdefault(row['cohort'], set()).add(row['pxd'])
    cohorts = ['abstract_pride_properties', 'abstract_methods_no_pride']
    totals = [summary['cohorts'][c]['new_support:added_pride'] for c in cohorts]
    counts = [len(projects[c]) for c in cohorts]
    means = [v / n for v, n in zip(totals, counts)]
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .7,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    fig.subplots_adjust(left=.21, right=.98, bottom=.23, top=.96)
    bars = ax.bar([0, 1], means, width=.52, color=['#167D9A', '#A0ADB8'])
    ax.set_xticks([0, 1], [f'Without methods\n(n = {counts[0]:,})',
                          f'With methods\n(n = {counts[1]:,})'])
    ax.set_xlabel('Methods availability in earlier input', labelpad=9)
    ax.set_ylabel('New PRIDE-supported fields\nper project', labelpad=8)
    ax.set_ylim(0, 6.7)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlim(-.6, 1.6)
    ax.tick_params(axis='y', length=3, width=.7)
    ax.tick_params(axis='x', length=0, pad=7)
    for bar, mean in zip(bars, means):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, mean + .12, f'{mean:.2f}', ha='center', va='bottom',
                fontsize=10)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'pride_gain_by_methods.{ext}', dpi=600, facecolor='white')
    with (OUT / 'pride_gain_by_methods.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['cohort', 'projects', 'pride_supported_new_fields', 'new_fields_per_project'])
        writer.writerows(zip(cohorts, counts, totals, means))
    print(f'Figures saved to {OUT}', flush=True)


if __name__ == '__main__':
    main()
