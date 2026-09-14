"""Audit and plot the paired 1,736-dataset subset without altering source outputs."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
HAMLET=ROOT.parents[2]
OUT=ROOT/'three_source_comparison_excluding_PXD001017_v2'
EXCLUDED='PXD001017'


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    pride=ROOT/'pride_only_1737';manuscript=ROOT/'manuscript_control_1737'
    spec=importlib.util.spec_from_file_location('coverage_values',ROOT/'compare.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    all_rows=list(csv.DictReader((pride/'input_manifest.csv').open()))
    assert sum(r['pxd']==EXCLUDED for r in all_rows)==1
    manifest=[r for r in all_rows if r['pxd']!=EXCLUDED]
    assert len(manifest)==len({r['pxd'] for r in manifest})==1736
    sources={'manuscript':manuscript/'outputs','pride':pride/'outputs',
        'combined':HAMLET/'textmining/framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737'}
    fields={agent:yaml.safe_load((pride/f'framework/docetl_pipeline/pipeline_{kind}.yaml').read_text())['operations'][0]['output']['schema']
            for kind,agent in [('biological','BiologicalAgent'),('technical','TechnicalAgent'),('experimental','ExperimentalDesignAgent')]}
    audit_root=OUT/'audit_subset';audit_root.mkdir()
    (audit_root/'inputs').mkdir()
    for agent in fields:
        (audit_root/'outputs'/agent).mkdir(parents=True)
        (audit_root/'outputs/NormalizedAgent'/agent).mkdir(parents=True)
    rows=[];provenance=[]
    for rec in manifest:
        r={'pxd':rec['pxd'],'cohort':'with_methods' if rec['has_methods']=='True' else 'abstract_only'}
        inp=pride/f"inputs/shard_{rec['shard']}"/(rec['pxd']+'.txt')
        (audit_root/'inputs'/inp.name).symlink_to(inp)
        for label,path in sources.items():
            populated=0;values=0
            for agent,schema in fields.items():
                found=list((path/f"shard_{rec['shard']}"/agent).glob(rec['pxd']+'_*.json'))
                assert len(found)==1,(label,agent,rec['pxd'])
                raw=found[0];normalized=raw.parent.parent/'NormalizedAgent'/agent/raw.name
                assert normalized.is_file(),normalized
                data=json.loads(raw.read_text());norm=json.loads(normalized.read_text())
                for field in schema:
                    assert field in data and field in norm
                    # Combined production includes legacy [value,evidence] records;
                    # values() handles both schemas. Only the newly audited PRIDE
                    # condition requires matched multi-value cardinality here.
                    if label=='pride':
                        assert isinstance(data[field],list) and len(data[field])==len(norm[field])
                    v=mod.values(data[field]);populated+=bool(v);values+=len(v)
                if label=='pride':
                    (audit_root/'outputs'/agent/raw.name).symlink_to(raw)
                    (audit_root/'outputs/NormalizedAgent'/agent/raw.name).symlink_to(normalized)
                provenance.append(dict(pxd=rec['pxd'],condition=label,agent=agent,
                    raw=str(raw),raw_sha256=mod.sha(raw),normalized=str(normalized)))
            r[label+'_fields']=populated;r[label+'_values']=values
        rows.append(r)
    subprocess.run([sys.executable,str(pride/'audit_multivalue_outputs.py'),
        '--input',str(audit_root/'inputs'),'--output',str(audit_root/'outputs'),
        '--report',str(OUT/'pride_subset_audit.json')],check=True,stdout=subprocess.DEVNULL)
    audit=json.loads((OUT/'pride_subset_audit.json').read_text());assert audit['problem_count']==0
    for agent in fields:
        assert audit['stats']['raw_'+agent]==audit['stats']['normalized_'+agent]==1736
    with (OUT/'dataset_coverage.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    groups=[[r for r in rows if r['cohort']==c] for c in ['abstract_only','with_methods']]
    labels=[f'Without methods\n(n = {len(groups[0]):,})',f'With methods\n(n = {len(groups[1]):,})']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary={'datasets':1736,'excluded_from_all_conditions':[EXCLUDED],'cohorts':{}}
    for c,g in zip(['abstract_only','with_methods'],groups):
        summary['cohorts'][c]={'n':len(g),'means':{k:sum(r[k] for r in g)/len(g) for k in rows[0] if k not in ['pxd','cohort']}}
    for metric,ylabel in [('fields','Populated fields per dataset'),('values','Distinct values per dataset')]:
        fig,ax=plt.subplots(figsize=(5.2,3.8),layout='constrained');ymax=0
        for shift,source,name,color in [(-.25,'manuscript','Abstract + methods','#A0ADB8'),
            (0,'pride','PRIDE properties/protocols','#C79B60'),
            (.25,'combined','Abstract + methods + PRIDE properties/protocols','#167D9A')]:
            means=[sum(r[source+'_'+metric] for r in g)/len(g) for g in groups];ymax=max(ymax,*means)
            bars=ax.bar(np.arange(2)+shift,means,width=.23,label=name,color=color)
            for b,v in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.1f}',ha='center',fontsize=8)
        ax.set_ylim(0,ymax*1.5+1);ax.set_xticks([0,1],labels)
        ax.set_ylabel(ylabel);ax.set_xlabel('Methods availability in manuscript input')
        ax.legend(frameon=False,loc='upper left',fontsize=7.5,handlelength=1)
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'three_source_{metric}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'FIGURE_LEGEND.md').write_text('Three independently extracted input conditions, with PXD001017 excluded from ALL conditions because its PRIDE-only technical extraction did not complete. Paired subset: 1,736 datasets (1,135 without extracted methods; 601 with methods). Methods means available extracted methods; the without-methods manuscript condition contains abstracts only. Bars show mean populated fields or within-field distinct lexical values per dataset. Counts describe coverage, not accuracy; combined output need not equal the sum or union of source-specific output. No uncertainty intervals are implied. Same frozen production multi-value prompts; PRIDE contains properties and available sample/data-processing protocols. PRIDE-only cached-response recoveries were reprocessed through normalization and QC, and the selected subset passed the evidence/schema audit. Original 1,737-dataset outputs and completion state are unchanged.\n')
    (OUT/'COMPLETE').touch();print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
