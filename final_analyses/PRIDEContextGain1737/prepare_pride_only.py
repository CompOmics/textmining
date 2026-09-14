"""Freeze the manuscript-control workflow and retain only PRIDE source blocks."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parent
CONTROL=ROOT/'manuscript_control_1737'
DEST=ROOT/'pride_only_1737'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert (CONTROL/'COMPLETE').exists()
    assert not DEST.exists(), 'Preserve existing run'
    DEST.mkdir()
    frozen=DEST/'framework';frozen.mkdir()
    for directory in ['docetl_pipeline','normalization','validation','core','agents','configs']:
        shutil.copytree(CONTROL/'framework'/directory,frozen/directory,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(CONTROL/'framework/config.yaml',frozen/'config.yaml')
    for directory in ['ontologies','ontology_cache']:
        (frozen/directory).symlink_to((CONTROL/'framework'/directory).resolve(),target_is_directory=True)
    (frozen/'logs').mkdir()
    shutil.copy2(CONTROL/'audit_multivalue_outputs.py',DEST/'audit_multivalue_outputs.py')
    worker=(CONTROL/'worker.sh').read_text()
    assert str(CONTROL) in worker
    (DEST/'worker.sh').write_text(worker.replace(str(CONTROL),str(DEST)))
    records=list(csv.DictReader((CONTROL/'input_manifest.csv').open()))
    for shard in range(8):(DEST/f'inputs/shard_{shard}').mkdir(parents=True)
    manifest=[]
    allowed={'PRIDE PROJECT PROPERTIES','PRIDE SAMPLE PROCESSING PROTOCOL','PRIDE DATA PROCESSING PROTOCOL'}
    for rec in records:
        source=Path(rec['source']);assert sha(source)==rec['source_sha256'],source
        text=source.read_text();headers=list(re.finditer(r'^===\s*(.*?)\s*===\s*$',text,flags=re.M))
        kept=[];names=[]
        for i,h in enumerate(headers):
            name=h.group(1)
            if name not in allowed:
                assert name in {'ABSTRACT','MATERIALS AND METHODS'},name
                continue
            end=headers[i+1].start() if i+1<len(headers) else len(text)
            block=text[h.start():end];kept.append(block);names.append(name)
        assert kept and any(re.sub(r'^===.*?===\s*','',b,flags=re.S).strip() for b in kept),rec['pxd']
        only=''.join(kept)
        assert not re.search(r'^===\s*(ABSTRACT|MATERIALS AND METHODS)\s*===',only,re.M)
        target=DEST/f"inputs/shard_{rec['shard']}"/(rec['pxd']+'.txt');target.write_text(only)
        manifest.append(dict(pxd=rec['pxd'],shard=rec['shard'],has_methods=rec['has_methods'],
            source=str(source),source_sha256=sha(source),pride_only_sha256=sha(target),
            characters=len(only),sections=';'.join(names)))
    assert len(manifest)==len({r['pxd'] for r in manifest})==1737
    with (DEST/'input_manifest.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    hashes={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*')
            if p.is_file() and not any(x in p.parts for x in ['ontologies','ontology_cache'])}
    old=json.loads((CONTROL/'provenance.json').read_text())['frozen_hashes']
    assert hashes==old,'Frozen pipeline differs from manuscript control'
    (DEST/'provenance.json').write_text(json.dumps(dict(projects=1737,source_control=str(CONTROL),
        frozen_hashes=hashes,model='Qwen/Qwen3.8-27B',dtype='bfloat16',multi_value=True,
        request_timeout=600,input='PRIDE properties and available protocols only',
        note='Same frozen production prompts, normalization and QC as manuscript-only control; no manuscript blocks retained.'),indent=2))
    (DEST/'PREPARED').touch()
    print('Prepared 1737 PRIDE-only inputs; frozen pipeline hashes match manuscript control.',flush=True)


if __name__=='__main__':main()
