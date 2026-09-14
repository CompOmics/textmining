"""Recover historical cached responses by exact request hash; compare raw metadata.

No inference, pickle execution, semantic matching or ontology renormalization.
"""
import collections,csv,hashlib,json,os,pickletools,re,sqlite3,unicodedata
from pathlib import Path
import jinja2,yaml
ROOT=Path(__file__).resolve().parent
HAMLET=ROOT.parents[2]
TEXT=HAMLET/'textmining'
OLD_INPUT=TEXT/'docetl_inputs_abstract_methods'
NEW_INPUT=TEXT/'docetl_inputs_abstract_methods_pride_protocols'
LATEST=TEXT/'framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737'
CACHE=HAMLET/'project_cache/docetl/qwen3_8_27b_abstract_methods_1737_retry3'
AGENTS={'biological':'BiologicalAgent','technical':'TechnicalAgent','experimental':'ExperimentalDesignAgent'}
NULL={'','unknown','none','null','not available','not applicable','n/a','not specified','not reported'}

def norm(s):return ' '.join(unicodedata.normalize('NFKC',str(s)).split())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_cache():
    cache={};conflicts=[];provenance=[]
    for p in sorted(CACHE.rglob('cache.db')):
        wal=Path(str(p)+'-wal')
        if wal.exists() and wal.stat().st_size:raise RuntimeError('Nonempty WAL present: make a consistent snapshot before recovery')
        c=sqlite3.connect(p.resolve().as_uri()+'?immutable=1',uri=True)
        n=0
        for key,value in c.execute('select key,value from Cache'):
            n+=1
            if key in cache and cache[key][0]!=value:conflicts.append(key)
            cache[key]=(value,str(p))
        c.close();provenance.append({'path':str(p),'entries':n,'sha256':sha(p)})
    if conflicts:raise RuntimeError(f'Conflicting cached responses for {len(conflicts)} identical keys')
    return cache,provenance

def decode(value,fields):
    # Inspect pickle string literals only. Never unpickle or instantiate saved classes.
    candidates=[]
    for op,s,pos in pickletools.genops(value):
        if not isinstance(s,str) or not s.startswith('{'):continue
        try:d=json.loads(s)
        except ValueError:continue
        if isinstance(d,dict) and set(fields)<=set(d):candidates.append(d)
    unique={json.dumps(d,sort_keys=True):d for d in candidates}
    if len(unique)!=1:raise ValueError(f'Expected one response payload, got {len(unique)}')
    return next(iter(unique.values()))

def request_key(op,pxd,text):
    prompt=jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(op['prompt']).render(input={'id':pxd,'text':text})
    config={**op,'timeout':600,'max_retries_per_timeout':0,'bypass_cache':True}
    key={'model':'openai/Qwen/Qwen3.8-27B','op_type':'map',
         'messages':json.dumps([{'role':'user','content':prompt}],sort_keys=True),
         'output_schema':json.dumps(op['output']['schema'],sort_keys=True),
         'scratchpad':None,'system_prompt':json.dumps({},sort_keys=True),'op_config':json.dumps(config,sort_keys=True)}
    return hashlib.md5(json.dumps(key,sort_keys=True).encode()).hexdigest()

def entries(v):
    if isinstance(v,list) and all(isinstance(x,dict) for x in v):
        return [(str(x.get('value','')),str(x.get('evidence',''))) for x in v]
    if isinstance(v,list) and len(v)==2 and all(isinstance(x,str) for x in v):return [(v[0],v[1])]
    if isinstance(v,dict) and 'value' in v:return [(str(v['value']),str(v.get('evidence','')))]
    if isinstance(v,str):return [(v,'')]
    return []

def values(v):
    # Split only explicit semicolons in BOTH schemas. Do not split commas: chemical
    # and anatomical names can contain them. Counts of additional values are lexical.
    return {norm(part).casefold() for value,evidence in entries(v) for part in value.split(';') if norm(part).casefold() not in NULL}

def blocks(text):
    pieces=re.split(r'^===\s*(.*?)\s*===\s*$',text,flags=re.M)
    return {pieces[i]:pieces[i+1] for i in range(1,len(pieces)-1,2)}

def support(v,old,new):
    sections=blocks(new);old=norm(old)
    categories=collections.Counter()
    for value,evidence in entries(v):
        if norm(value).casefold() in NULL:continue
        quotes=[re.sub(r'^inferred:\s*','',q.strip(),flags=re.I) for q in evidence.split(' || ')]
        for quote in quotes:
            quote=norm(quote)
            if len(quote)<8:categories['unverified']+=1;continue
            if quote in old:categories['earlier_context']+=1;continue
            matched=[name for name,body in sections.items() if 'PRIDE' in name and quote in norm(body)]
            if matched:
                categories['added_pride']+=1
                for name in matched:categories['section:'+name]+=1
            else:categories['unverified']+=1
    if not categories:return 'unverified',{}
    earlier=categories['earlier_context']>0;added=categories['added_pride']>0
    label='mixed' if earlier and added else 'added_pride' if added else 'earlier_context' if earlier else 'unverified'
    return label,dict(categories)

def main():
    dest=ROOT/'runs'/('job_'+os.environ.get('SLURM_JOB_ID','local'))
    if dest.exists():raise RuntimeError('Preserve existing audit; choose a separate version')
    dest.mkdir(parents=True);recovered=dest/'recovered_earlier_raw';recovered.mkdir(exist_ok=False)
    cache,cache_provenance=load_cache()
    old_manifest=list(csv.DictReader((OLD_INPUT/'manifest.tsv').open(),delimiter='\t'))
    ids=[r['pxd'] for r in old_manifest]
    if len(ids)!=1737 or len(set(ids))!=1737:raise RuntimeError('Unexpected input coverage')
    cohort={r['pxd']:('abstract_methods_no_pride' if r['has_pride_properties_fallback']=='False' else 'abstract_pride_properties') for r in old_manifest}
    latest={};ops={};prompt_hashes={}
    for kind,agent in AGENTS.items():
        p=TEXT/'framework/docetl_pipeline'/f'pipeline_{kind}.yaml';ops[kind]=yaml.safe_load(p.read_text())['operations'][0];prompt_hashes[str(p)]=sha(p)
        paths=list(LATEST.glob('shard_*/'+agent+'/PXD*.json'))
        mapping={p.name.split('_')[0]:p for p in paths}
        if len(mapping)!=len(paths) or set(mapping)!=set(ids):raise RuntimeError(f'Latest {agent} missing/duplicate/extra IDs')
        latest[kind]=mapping;(recovered/agent).mkdir()
    provenance=[];missing=[];rows=[];field_stats=collections.defaultdict(collections.Counter);by_cohort=collections.defaultdict(collections.Counter);per_project=collections.defaultdict(collections.Counter)
    for i,pxd in enumerate(ids):
        old_path=OLD_INPUT/f'{pxd}.txt';new_path=NEW_INPUT/f'{pxd}.txt';old=old_path.read_text();new=new_path.read_text()
        for kind,agent in AGENTS.items():
            op=ops[kind];key=request_key(op,pxd,old)
            if key not in cache:
                missing.append({'pxd':pxd,'agent':agent,'key':key});continue
            payload,cache_path=cache[key];before=decode(payload,op['output']['schema'])
            (recovered/agent/f'{pxd}_{kind}_recovered.json').write_text(json.dumps(before,indent=2))
            new_file=latest[kind][pxd];after=json.loads(new_file.read_text())
            provenance.append({'pxd':pxd,'agent':agent,'key':key,'cache_db':cache_path,'old_input_sha256':sha(old_path),'new_input_sha256':sha(new_path),'latest_output':str(new_file),'latest_sha256':sha(new_file)})
            for field in op['output']['schema']:
                a=values(before.get(field));b=values(after.get(field))
                if not a and not b:status='both_unknown'
                elif not a:status='newly_populated'
                elif not b:status='lost'
                elif a==b:status='same_lexical_values'
                elif a<b:status='additional_lexical_values'
                elif b<a:status='fewer_lexical_values'
                else:status='changed_lexical_values'
                origin,detail=support(after.get(field),old,new) if status=='newly_populated' else ('not_assessed',{})
                row={'pxd':pxd,'cohort':cohort[pxd],'agent':agent,'field':field,'status':status,'old_values':json.dumps(sorted(a)),'new_values':json.dumps(sorted(b)),'added_values':json.dumps(sorted(b-a)),'removed_values':json.dumps(sorted(a-b)),'new_field_support':origin,'support_detail':json.dumps(detail),'new_evidence':json.dumps(entries(after.get(field)),ensure_ascii=False)}
                rows.append(row)
                for counter in [field_stats[agent+'.'+field],by_cohort[cohort[pxd]],per_project[pxd]]:
                    counter.update({'total_fields':1,'old_populated':int(bool(a)),'new_populated':int(bool(b)),status:1})
                if status=='newly_populated':by_cohort[cohort[pxd]]['new_support:'+origin]+=1
        if (i+1)%200==0:print('Processed',i+1,flush=True)
    (dest/'missing_cache_keys.json').write_text(json.dumps(missing,indent=2))
    (dest/'provenance.json').write_text(json.dumps({'cache_databases':cache_provenance,'prompts':prompt_hashes,'mappings':provenance},indent=2))
    if missing:raise RuntimeError(f'{len(missing)} cache mappings missing: do not report full-coverage comparison')
    with (dest/'field_changes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'projects':len(ids),'recovered_responses':len(provenance),'earlier_definition':'abstract+methods for 601; abstract+PRIDE properties for 1136',
      'latest_outputs':str(LATEST),'comparison':'incremental context plus changed multi-value extraction workflow, not causal PRIDE-only effect',
      'normalization':'raw provider response vs raw final extraction; normalization not compared',
      'limitations':['Historical cache contains provider responses, not restored final QC/repair outputs','Latest prompts include multi-value instructions and postprocessing differs','Lexical differences are not necessarily semantic additions or contradictions','Newly populated does not imply scientifically correct','Evidence attribution uses literal normalized containment; unmatched citations remain unverified'],
      'cohorts':{k:dict(v) for k,v in by_cohort.items()},'field_stats':{k:dict(v) for k,v in field_stats.items()},'projects_with_new_fields':sum(v['newly_populated']>0 for v in per_project.values()),'projects_with_lost_fields':sum(v['lost']>0 for v in per_project.values())}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'project_changes.json').write_text(json.dumps({k:dict(v) for k,v in per_project.items()},indent=2))
    (dest/'COMPLETE').touch();print(json.dumps({k:v for k,v in summary.items() if k!='field_stats'},indent=2))
if __name__=='__main__':main()
