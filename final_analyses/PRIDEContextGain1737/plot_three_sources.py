"""Add independently extracted PRIDE-only coverage; require complete audited runs."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
HAMLET=ROOT.parents[2]


def main():
    pride=ROOT/'pride_only_1737'; manuscript=ROOT/'manuscript_control_1737'
    assert (pride/'COMPLETE').exists() and (manuscript/'COMPLETE').exists()
    spec=importlib.util.spec_from_file_location('coverage_values',ROOT/'compare.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    manifest=list(csv.DictReader((pride/'input_manifest.csv').open()))
    sources={'manuscript':manuscript/'outputs','pride':pride/'outputs',
        'combined':HAMLET/'textmining/framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737'}
    for shard in range(8):
        audit=json.loads((pride/f'outputs/shard_{shard}/multivalue_audit.json').read_text())
        assert audit['problem_count']==0
    fields={agent:yaml.safe_load((pride/f'framework/docetl_pipeline/pipeline_{kind}.yaml').read_text())['operations'][0]['output']['schema']
            for kind,agent in [('biological','BiologicalAgent'),('technical','TechnicalAgent'),('experimental','ExperimentalDesignAgent')]}
    rows=[]
    for rec in manifest:
        r={'pxd':rec['pxd'],'cohort':'with_methods' if rec['has_methods']=='True' else 'abstract_only'}
        for label,path in sources.items():
            populated=0;values=0
            for agent,schema in fields.items():
                found=list((path/f"shard_{rec['shard']}"/agent).glob(rec['pxd']+'_*.json'))
                assert len(found)==1,(label,agent,rec['pxd'])
                data=json.loads(found[0].read_text())
                for field in schema:
                    assert field in data
                    v=mod.values(data[field]);populated+=bool(v);values+=len(v)
            r[label+'_fields']=populated;r[label+'_values']=values
        rows.append(r)
    assert len(rows)==1737
    out=ROOT/'three_source_comparison';out.mkdir(exist_ok=True)
    with (out/'dataset_coverage.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cohorts=['abstract_only','with_methods'];groups=[[r for r in rows if r['cohort']==c] for c in cohorts]
    labels=[f'Without methods\n(n = {len(groups[0]):,})',f'With methods\n(n = {len(groups[1]):,})']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
    for metric,ylabel in [('fields','Populated fields per dataset'),('values','Distinct values per dataset')]:
        fig,ax=plt.subplots(figsize=(5.2,3.8),layout='constrained')
        ymax=0
        for shift,source,name,color in [(-.25,'manuscript','Abstract + methods','#A0ADB8'),
            (0,'pride','PRIDE properties/protocols','#C79B60'),
            (.25,'combined','Abstract + methods + PRIDE properties/protocols','#167D9A')]:
            means=[sum(r[source+'_'+metric] for r in g)/len(g) for g in groups]
            ymax=max(ymax,*means)
            bars=ax.bar(np.arange(2)+shift,means,width=.23,label=name,color=color)
            for b,v in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.1f}',ha='center',fontsize=8)
        ax.set_ylim(0,ymax*1.5+1);ax.set_xticks([0,1],labels)
        ax.set_ylabel(ylabel);ax.set_xlabel('Methods availability in manuscript input')
        ax.legend(frameon=False,loc='upper left',fontsize=7.5,handlelength=1)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'three_source_{metric}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    (out/'COMPLETE').touch()
    print('Three-source figures complete:',out,flush=True)


if __name__=='__main__':main()
