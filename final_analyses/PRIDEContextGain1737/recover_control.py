"""Recover exact cached responses; stage QC/normalization; add missing files only."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickletools
import re
import sqlite3
import subprocess
import sys
import time
import jinja2
import yaml

ROOT = Path(__file__).resolve().parent
HAMLET = ROOT.parents[2]
CONTROL = ROOT / 'manuscript_control_1737'
FRAMEWORK = CONTROL / 'framework'
TAG = 'qwen3_8_27b_manuscript_only_multivalue_1737'
AGENTS = ['BiologicalAgent', 'TechnicalAgent', 'ExperimentalDesignAgent']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    started = time.time()
    stage = CONTROL / ('recovery_' + os.environ['SLURM_JOB_ID'])
    stage.mkdir(exist_ok=False)
    (stage / 'inputs').mkdir()
    existing = {str(p):sha(p) for p in (CONTROL/'outputs').rglob('*.json')}
    src = FRAMEWORK / 'docetl_pipeline/run_docetl.py'
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_pipeline')
    block = next(n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'multi_value')
    pipeline = yaml.safe_load((FRAMEWORK/'docetl_pipeline/pipeline_experimental.yaml').read_text())
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(src), 'exec'),
         {'pipeline_cfg':pipeline, 'multi_value':True, 're':re})
    op = pipeline['operations'][0]
    op.update(timeout=600, max_retries_per_timeout=0, bypass_cache=True)
    records = []; mappings = []; lookup = {}
    for shard in range(8):
        inp = CONTROL / f'inputs/shard_{shard}'
        out = CONTROL / f'outputs/shard_{shard}'
        expected = {p.stem for p in inp.glob('*.txt')}
        for agent in AGENTS[:2]:
            assert {p.name.split('_')[0] for p in (out/agent).glob('*.json')} == expected
        got = {p.name.split('_')[0] for p in (out/AGENTS[2]).glob('*.json')}
        assert got <= expected
        missing = expected - got
        if not missing: continue
        db = HAMLET / f'project_cache/docetl/{TAG}/966203_{shard}/.cache/docetl/llm/cache.db'
        wal = Path(str(db)+'-wal')
        assert not wal.exists() or wal.stat().st_size == 0
        con = sqlite3.connect(db.resolve().as_uri()+'?immutable=1', uri=True)
        for pxd in sorted(missing):
            source = inp / (pxd+'.txt'); text = source.read_text()
            prompt = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(op['prompt']).render(input={'id':pxd,'text':text})
            kd = dict(model='openai/Qwen/Qwen3.8-27B', op_type='map',
                messages=json.dumps([{'role':'user','content':prompt}],sort_keys=True),
                output_schema=json.dumps(op['output']['schema'],sort_keys=True), scratchpad=None,
                system_prompt=json.dumps({},sort_keys=True), op_config=json.dumps(op,sort_keys=True))
            key = hashlib.md5(json.dumps(kd,sort_keys=True).encode()).hexdigest()
            row = con.execute('select value from Cache where key=?',(key,)).fetchone()
            assert row is not None, pxd
            candidates = {}
            for _, value, _ in pickletools.genops(row[0]):
                if not isinstance(value,str) or not value.startswith('{'): continue
                try: d=json.loads(value)
                except ValueError: continue
                if isinstance(d,dict) and set(op['output']['schema']) <= set(d):
                    candidates[json.dumps(d,sort_keys=True)] = d
            assert len(candidates)==1, pxd
            d = next(iter(candidates.values()))
            for field in op['output']['schema']:
                assert isinstance(d[field],list) and d[field]
                assert all(isinstance(v,dict) and isinstance(v.get('value'),str) and isinstance(v.get('evidence'),str) for v in d[field])
            records.append({'id':pxd, **d}); lookup[pxd]=text
            (stage/'inputs'/source.name).write_bytes(source.read_bytes())
            mappings.append(dict(pxd=pxd, shard=shard, cache_db=str(db), request_key=key,
                                 source_sha256=sha(source), payload=d))
        con.close()
    assert len(records)==9, f'Expected exactly nine missing records; got {len(records)}'
    (stage/'cache_provenance.json').write_text(json.dumps(mappings,indent=2))
    print('Exactly recovered nine cached responses',flush=True)
    os.chdir(FRAMEWORK)
    sys.path.insert(0,str(FRAMEWORK))
    spec=importlib.util.spec_from_file_location('frozen_control_runner',src)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner._apply_env(runner._load_config(str(FRAMEWORK/'configs/qwen3_8_27b_vllm.yaml')))
    runner._require_complete_qc()
    runner._write_outputs(records,lookup,stage/'outputs',AGENTS[2],'_experimental',True,
                          model_tag=TAG,multi_value=True)
    runner._run_normalization(stage/'outputs',[runner.AGENTS['experimental']])
    subprocess.run([sys.executable,str(CONTROL/'audit_multivalue_outputs.py'),
        '--input',str(stage/'inputs'),'--output',str(stage/'outputs'),
        '--report',str(stage/'staged_audit.json')],check=True,stdout=subprocess.DEVNULL)
    for m in mappings:
        name=f"{m['pxd']}_experimental_{TAG}.json"
        for relative in [Path(AGENTS[2])/name,Path('NormalizedAgent')/AGENTS[2]/name]:
            source=stage/'outputs'/relative;target=CONTROL/f"outputs/shard_{m['shard']}"/relative
            assert source.exists() and not target.exists(), str(target)
            # Atomic no-clobber insertion; staged files remain as provenance.
            os.link(source,target)
    assert all(sha(Path(p))==digest for p,digest in existing.items()), 'Existing output changed'
    audits=[]
    for shard in range(8):
        inp=CONTROL/f'inputs/shard_{shard}';out=CONTROL/f'outputs/shard_{shard}'
        expected={p.stem for p in inp.glob('*.txt')}
        for agent in AGENTS:
            for folder in [out/agent,out/'NormalizedAgent'/agent]:
                files=list(folder.glob('*.json'))
                assert len(files)==len(expected) and {p.name.split('_')[0] for p in files}==expected
        report=stage/f'full_audit_shard_{shard}.json'
        subprocess.run([sys.executable,str(CONTROL/'audit_multivalue_outputs.py'),
            '--input',str(inp),'--output',str(out),'--report',str(report)],
            check=True,stdout=subprocess.DEVNULL)
        audits.append({'shard':shard,'problem_count':json.loads(report.read_text())['problem_count']})
        print('Full shard audit passed',shard,flush=True)
    report=dict(status='complete',recovered=9,raw_outputs=5211,normalized_outputs=5211,
                original_outputs_unchanged=True,audits=audits,seconds=time.time()-started,
                no_llm_inference=True,recovery_job=os.environ['SLURM_JOB_ID'])
    (stage/'recovery_report.json').write_text(json.dumps(report,indent=2))
    for shard in [0,3]:
        (CONTROL/f'outputs/shard_{shard}/RECOVERED_FROM_CACHE').write_text(str(stage)+'\n')
        (CONTROL/f'outputs/shard_{shard}/SUCCESS').touch()
    (CONTROL/'COMPLETE').write_text(str(stage/'recovery_report.json')+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
