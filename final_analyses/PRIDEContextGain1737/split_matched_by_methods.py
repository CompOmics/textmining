"""Stratify completed matched-prompt comparison; no inference."""
import collections
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'matched_prompt_comparison/job_966224'
OUT=RUN/'by_methods'


def main():
    assert (RUN/'COMPLETE').exists()
    datasets=list(csv.DictReader((RUN/'results/dataset_amounts_overlap.csv').open()))
    fields=list(csv.DictReader((RUN/'results/field_value_agreement.csv').open()))
    cohorts=['abstract_only','with_methods']
    summaries=[]
    for cohort in cohorts:
        subset=[r for r in datasets if r['cohort']==cohort];n=len(subset)
        totals={k:sum(int(r[k]) for r in subset) for k in ['old_fields','new_fields',
            'old_values','new_values','expanded_only_fields','earlier_only_fields','shared_fields']}
        cats=collections.Counter(r['status'] for r in fields if r['cohort']==cohort
                                 and int(r['old_count']) and int(r['new_count']))
        assert sum(cats.values())==totals['shared_fields']
        domains={agent:sum(int(int(r['new_count'])>0)-int(int(r['old_count'])>0)
                    for r in fields if r['cohort']==cohort and r['agent']==agent)
                    for agent in ['BiologicalAgent','TechnicalAgent','ExperimentalDesignAgent']}
        summaries.append(dict(cohort=cohort,datasets=n,totals=totals,
            means={k:v/n for k,v in totals.items()},
            mean_net_field_gain=(totals['new_fields']-totals['old_fields'])/n,
            relative_field_gain_pct=100*(totals['new_fields']/totals['old_fields']-1),
            agreement_counts=dict(cats),
            agreement_pct={k:100*v/totals['shared_fields'] for k,v in cats.items()},
            net_field_gain_by_domain=domains))
    assert sum(s['datasets'] for s in summaries)==1737
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.5),layout='constrained')
    labels=[f'Without methods\n(n = {summaries[0]["datasets"]:,})',
            f'With methods\n(n = {summaries[1]["datasets"]:,})']
    x=np.arange(2)
    for shift,metric,label,color in [(-.19,'old_fields','Abstract + methods','#A0ADB8'),
        (.19,'new_fields','Abstract + methods\n+ PRIDE properties/protocols','#167D9A')]:
        values=[s['means'][metric] for s in summaries]
        bars=axes[0].bar(x+shift,values,width=.35,label=label,color=color)
        for bar,v in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,v+.4,f'{v:.1f}',ha='center',fontsize=8)
    axes[0].set_ylim(0,36);axes[0].set_yticks([0,10,20,30])
    axes[0].set_ylabel('Populated fields per dataset')
    axes[0].legend(frameon=False,fontsize=7.5,loc='upper left',handlelength=1)
    gains=[s['mean_net_field_gain'] for s in summaries]
    axes[1].bar(x,gains,width=.52,color=['#167D9A','#A0ADB8'])
    for i,v in enumerate(gains):axes[1].text(i,v+.2,f'{v:.2f}',ha='center',fontsize=9)
    axes[1].set_ylim(0,15);axes[1].set_yticks([0,5,10,15])
    axes[1].set_ylabel('Net additional fields per dataset')
    for ax,letter in zip(axes,'AB'):
        ax.set_xticks(x,labels);ax.set_xlabel('Methods availability in manuscript input')
        ax.set_title(letter,loc='left',fontweight='bold')
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'pride_gain_by_methods.{ext}',dpi=600,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
