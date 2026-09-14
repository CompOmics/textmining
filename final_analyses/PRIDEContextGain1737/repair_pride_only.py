"""Exact-cache recovery, minimal retry preparation, no-clobber merge and audit."""
import ast,csv,hashlib,importlib.util,json,os,pickletools,re,sqlite3,subprocess,sys
from pathlib import Path
import jinja2,yaml

ROOT=Path(__file__).resolve().parent
HAMLET=ROOT.parents[2]
RUN=ROOT/'pride_only_1737'
FRAME=RUN/'framework'
TAG='qwen3_8_27b_pride_only_multivalue_1737'
STAGE=Path(os.environ['PRIDE_REPAIR_STAGE']) if os.environ.get('PRIDE_REPAIR_STAGE') else RUN/('repair_'+os.environ['SLURM_JOB_ID'])
AGENTS={'biological':'BiologicalAgent','technical':'TechnicalAgent','experimental':'ExperimentalDesignAgent'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(inputs,outputs,report):
    subprocess.run([sys.executable,str(RUN/'audit_multivalue_outputs.py'),'--input',str(inputs),
        '--output',str(outputs),'--report',str(report)],check=True,stdout=subprocess.DEVNULL)

def insert(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(),target
    json.loads(source.read_text());os.link(source,target)

def merge(mappings,base):
    for m in mappings:
        name=f"{m['pxd']}_{m['kind']}_{TAG}.json"
        for rel in [Path(m['agent'])/name,Path('NormalizedAgent')/m['agent']/name]:
            insert(base/rel,RUN/f"outputs/shard_{m['shard']}"/rel)

def recover():
    STAGE.mkdir(exist_ok=False);(STAGE/'inputs').mkdir()
    existing={str(p):sha(p) for p in (RUN/'outputs').rglob('*.json')}
    (STAGE/'existing_hashes.json').write_text(json.dumps(existing))
    src=FRAME/'docetl_pipeline/run_docetl.py';tree=ast.parse(src.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_pipeline')
    block=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='multi_value')
    ops={}
    for kind in AGENTS:
        pipeline=yaml.safe_load((FRAME/f'docetl_pipeline/pipeline_{kind}.yaml').read_text())
        exec(compile(ast.Module(body=[block],type_ignores=[]),str(src),'exec'),{'pipeline_cfg':pipeline,'multi_value':True,'re':re})
        op=pipeline['operations'][0];op.update(timeout=600,max_retries_per_timeout=0,bypass_cache=True);ops[kind]=op
    recovered=[];retry=[];records={k:[] for k in AGENTS};lookup={}
    for shard in range(8):
        inp=RUN/f'inputs/shard_{shard}';ids={p.stem for p in inp.glob('*.txt')}
        db=HAMLET/f'project_cache/docetl/{TAG}/966238_{shard}/.cache/docetl/llm/cache.db'
        wal=Path(str(db)+'-wal');assert not wal.exists() or wal.stat().st_size==0
        con=sqlite3.connect(db.resolve().as_uri()+'?immutable=1',uri=True)
        for kind,agent in AGENTS.items():
            raw=RUN/f'outputs/shard_{shard}'/agent;norm=RUN/f'outputs/shard_{shard}/NormalizedAgent'/agent
            got={p.name.split('_')[0] for p in raw.glob('*.json')}
            assert got<=ids and {p.name.split('_')[0] for p in norm.glob('*.json')}==got
            op=ops[kind]
            for pxd in sorted(ids-got):
                source=inp/(pxd+'.txt');text=source.read_text()
                prompt=jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(op['prompt']).render(input={'id':pxd,'text':text})
                kd=dict(model='openai/Qwen/Qwen3.8-27B',op_type='map',messages=json.dumps([{'role':'user','content':prompt}],sort_keys=True),
                    output_schema=json.dumps(op['output']['schema'],sort_keys=True),scratchpad=None,system_prompt=json.dumps({},sort_keys=True),op_config=json.dumps(op,sort_keys=True))
                key=hashlib.md5(json.dumps(kd,sort_keys=True).encode()).hexdigest()
                row=con.execute('select value from Cache where key=?',(key,)).fetchone();candidates={}
                if row:
                    for _,value,_ in pickletools.genops(row[0]):
                        if not isinstance(value,str) or not value.startswith('{'):continue
                        try:d=json.loads(value)
                        except ValueError:continue
                        if isinstance(d,dict) and set(op['output']['schema'])<=set(d):candidates[json.dumps(d,sort_keys=True)]=d
                m=dict(pxd=pxd,shard=shard,kind=kind,agent=agent,cache_db=str(db),request_key=key,source_sha256=sha(source))
                valid=False
                if len(candidates)==1:
                    d=next(iter(candidates.values()))
                    valid=all(isinstance(d[f],list) and d[f] and all(isinstance(v,dict) and isinstance(v.get('value'),str) and isinstance(v.get('evidence'),str) for v in d[f]) for f in op['output']['schema'])
                if valid:
                    records[kind].append({'id':pxd,**d});lookup[pxd]=text;m['payload']=d;recovered.append(m)
                    (STAGE/'inputs'/source.name).write_bytes(source.read_bytes())
                else:
                    retry.append(m);dest=STAGE/'retry_inputs'/kind;dest.mkdir(parents=True,exist_ok=True)
                    (dest/source.name).write_bytes(source.read_bytes())
        con.close()
    (STAGE/'cache_provenance.json').write_text(json.dumps(recovered,indent=2))
    (STAGE/'retry_manifest.json').write_text(json.dumps(retry,indent=2))
    print('Recovered',len(recovered),'Retry needed',len(retry),flush=True)
    os.chdir(FRAME);sys.path.insert(0,str(FRAME))
    spec=importlib.util.spec_from_file_location('pride_frozen_runner',src);runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner._apply_env(runner._load_config(str(FRAME/'configs/qwen3_8_27b_vllm.yaml')))
    runner._require_complete_qc();agents=[]
    for kind,rows in records.items():
        if not rows:continue
        runner._write_outputs(rows,lookup,STAGE/'outputs',AGENTS[kind],'_'+kind,True,model_tag=TAG,multi_value=True)
        agents.append(runner.AGENTS[kind])
    if agents:
        runner._run_normalization(STAGE/'outputs',agents)
        audit(STAGE/'inputs',STAGE/'outputs',STAGE/'cache_audit.json')
        merge(recovered,STAGE/'outputs')
    print('Cached outputs normalized, audited and merged.',flush=True)

def finish():
    retry=json.loads((STAGE/'retry_manifest.json').read_text())
    for kind in AGENTS:
        selected=[r for r in retry if r['kind']==kind]
        if selected:
            audit(STAGE/'retry_inputs'/kind,STAGE/'retry_outputs'/kind,STAGE/f'retry_audit_{kind}.json')
            merge(selected,STAGE/'retry_outputs'/kind)
    existing=json.loads((STAGE/'existing_hashes.json').read_text())
    assert all(sha(Path(p))==v for p,v in existing.items()),'Original outputs changed'
    for shard in range(8):
        inp=RUN/f'inputs/shard_{shard}';out=RUN/f'outputs/shard_{shard}';expected={p.stem for p in inp.glob('*.txt')}
        for agent in AGENTS.values():
            for folder in [out/agent,out/'NormalizedAgent'/agent]:
                files=list(folder.glob('*.json'));assert len(files)==len(expected) and {p.name.split('_')[0] for p in files}==expected
        report=STAGE/f'full_audit_shard_{shard}.json';audit(inp,out,report)
        assert json.loads(report.read_text())['problem_count']==0
        target=out/'multivalue_audit.json'
        if not target.exists():insert(report,target)
        else:assert json.loads(target.read_text())['problem_count']==0
        print('Shard',shard,'full audit passed',flush=True)
    for shard in range(8):(RUN/f'outputs/shard_{shard}/SUCCESS').touch()
    report=dict(status='complete',cached_recoveries=len(json.loads((STAGE/'cache_provenance.json').read_text())),
        rerun_records=len(retry),raw_outputs=5211,normalized_outputs=5211,original_outputs_unchanged=True)
    (STAGE/'repair_report.json').write_text(json.dumps(report,indent=2))
    (RUN/'COMPLETE').write_text(str(STAGE/'repair_report.json')+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    {'recover':recover,'finish':finish}[sys.argv[1]]()
