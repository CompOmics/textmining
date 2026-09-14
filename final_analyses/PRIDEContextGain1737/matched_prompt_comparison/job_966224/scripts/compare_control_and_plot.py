"""Compare completed manuscript control to PRIDE protocols run and plot."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parent
HAMLET = ROOT.parents[2]
CONTROL = ROOT / 'manuscript_control_1737'
FULL = HAMLET / 'textmining/framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737'
DEST = ROOT / 'matched_prompt_comparison' / ('job_' + os.environ['SLURM_JOB_ID'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    assert (CONTROL/'COMPLETE').exists()
    recovery = json.loads(Path((CONTROL/'COMPLETE').read_text().strip()).read_text())
    assert recovery['status']=='complete' and all(a['problem_count']==0 for a in recovery['audits'])
    DEST.mkdir(parents=True,exist_ok=False)
    scripts = DEST/'scripts';scripts.mkdir()
    for filename in ['compare.py','compare_three_levels.py','plot_three_levels.py','compare_control_and_plot.py']:
        shutil.copy2(ROOT/filename,scripts/filename)
    extraction = module('control_extraction_values',scripts/'compare.py')
    manifest = list(csv.DictReader((CONTROL/'input_manifest.csv').open()))
    assert len(manifest)==1737 and len({r['pxd'] for r in manifest})==1737
    provenance=[];rows=[]
    agents={'biological':'BiologicalAgent','technical':'TechnicalAgent','experimental':'ExperimentalDesignAgent'}
    for kind,agent in agents.items():
        fields=yaml.safe_load((CONTROL/f'framework/docetl_pipeline/pipeline_{kind}.yaml').read_text())['operations'][0]['output']['schema']
        for rec in manifest:
            pxd,shard=rec['pxd'],rec['shard']
            a=list((CONTROL/f'outputs/shard_{shard}'/agent).glob(pxd+'_*.json'))
            b=list((FULL/f'shard_{shard}'/agent).glob(pxd+'_*.json'))
            assert len(a)==len(b)==1,(pxd,agent)
            aa=json.loads(a[0].read_text());bb=json.loads(b[0].read_text())
            provenance.append(dict(pxd=pxd,agent=agent,manuscript_path=str(a[0]),
                pride_path=str(b[0]),manuscript_sha256=extraction.sha(a[0]),pride_sha256=extraction.sha(b[0])))
            for field in fields:
                assert field in aa and field in bb,(pxd,field)
                rows.append(dict(pxd=pxd,cohort='with_methods' if rec['has_methods']=='True' else 'abstract_only',
                    agent=agent,field=field,old_values=json.dumps(sorted(extraction.values(aa[field]))),
                    new_values=json.dumps(sorted(extraction.values(bb[field])))))
    assert len(rows)==66006
    source=DEST/'field_changes.csv'
    with source.open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (DEST/'input_provenance.json').write_text(json.dumps(provenance,indent=2))
    compare=module('matched_control_comparison',scripts/'compare_three_levels.py')
    compare.ROOT=ROOT;compare.HAMLET=HAMLET;compare.SOURCE=source;compare.OUT=DEST/'results'
    compare.main()
    summary_path=compare.OUT/'summary.json'
    s=json.loads(summary_path.read_text())
    s['comparison']='Abstract + available methods versus abstract + available methods + PRIDE properties/protocols; production multi-value prompts matched'
    s['limitations']=[s['comparison']]+s['limitations'][1:]+[
        'Both sides use final production prompts, but are separate stochastic generations',
        'Full-context run includes timeout repairs; control includes nine exact-cache recoveries',
        'CPU thread allocation differs; raw values are compared before normalization']
    summary_path.write_text(json.dumps(s,indent=2))
    plot=module('matched_control_figures',scripts/'plot_three_levels.py')
    plot.RUN=compare.OUT;plot.OUT=DEST/'figures'
    plot.PROPERTY_FALLBACK='Abstract + methods'
    plot.PRIDE_PROTOCOLS='Abstract + methods\n+ PRIDE properties/protocols'
    plot.main()
    (DEST/'COMPLETE').write_text('Comparison and figures complete\n')
    print('COMPLETE:',DEST,flush=True)


if __name__=='__main__':main()
