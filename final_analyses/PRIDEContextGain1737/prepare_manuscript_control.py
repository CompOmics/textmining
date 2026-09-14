"""Freeze production code and remove only PRIDE input blocks for a paired control."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
HAMLET = ROOT.parents[2]
TEXT = HAMLET / 'textmining'
DEST = ROOT / 'manuscript_control_1737'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists(), 'Preserve existing control run'
    framework = TEXT / 'framework'
    tracked = ['framework/docetl_pipeline', 'framework/configs/qwen3_8_27b_vllm.yaml',
               'framework/normalization', 'framework/validation', 'framework/smoke_docetl_b300.sbatch']
    status = subprocess.check_output(['git', '-C', str(TEXT), 'status', '--porcelain', '--', *tracked], text=True)
    assert not status.strip(), 'Uncommitted production workflow changes require inspection'
    DEST.mkdir()
    frozen = DEST / 'framework'; frozen.mkdir()
    for directory in ['docetl_pipeline', 'normalization', 'validation', 'core', 'agents', 'configs']:
        shutil.copytree(framework / directory, frozen / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(framework / 'config.yaml', frozen / 'config.yaml')
    for directory in ['ontologies', 'ontology_cache']:
        (frozen / directory).symlink_to(framework / directory, target_is_directory=True)
    (frozen / 'logs').mkdir()
    shutil.copy2(TEXT / 'audit_multivalue_outputs.py', DEST / 'audit_multivalue_outputs.py')
    source_script = (framework / 'smoke_docetl_b300.sbatch').read_text()
    assert source_script.count('FRAMEWORK="$PROJECT/textmining/framework"') == 1
    source_script = source_script.replace('FRAMEWORK="$PROJECT/textmining/framework"', f'FRAMEWORK="{frozen}"')
    source_script = source_script.replace('"$PROJECT/textmining/audit_multivalue_outputs.py"',
                                           f'"{DEST}/audit_multivalue_outputs.py"')
    (DEST / 'worker.sh').write_text(source_script)
    shards = DEST / 'inputs'
    for i in range(8): (shards / f'shard_{i}').mkdir(parents=True)
    full = TEXT / 'docetl_inputs_abstract_methods_pride_protocols'
    prior = TEXT / 'production_inputs/abstract_methods_pride_protocols_1737_8shards'
    audit = json.loads((ROOT / 'runs/job_965823/provenance.json').read_text())
    expected = {r['pxd']:r['new_input_sha256'] for r in audit['mappings']}
    seen = set(); records = []
    for i in range(8):
        for pointer in sorted((prior / f'shard_{i}').glob('PXD*.txt')):
            pxd = pointer.stem
            assert pxd not in seen
            seen.add(pxd)
            source = full / (pxd + '.txt')
            assert sha(source) == expected[pxd], f'Full-context source changed: {pxd}'
            text = source.read_text()
            headers = list(re.finditer(r'^===\s*(.*?)\s*===\s*$', text, flags=re.M))
            assert headers and headers[0].group(1) == 'ABSTRACT'
            kept = []; removed = []; methods = False
            for j, h in enumerate(headers):
                name = h.group(1)
                block = text[h.start():headers[j+1].start() if j+1<len(headers) else len(text)]
                if name in {'ABSTRACT', 'MATERIALS AND METHODS'}:
                    kept.append(block)
                    methods |= name == 'MATERIALS AND METHODS'
                else:
                    assert name.startswith('PRIDE '), f'Unexpected section {name}'
                    removed.append(name)
            control = ''.join(kept)
            assert control.strip() and not re.search(r'^===\s*PRIDE', control, re.M)
            target = shards / f'shard_{i}' / source.name
            target.write_text(control)
            records.append(dict(pxd=pxd, shard=i, has_methods=methods,
                source=str(source), source_sha256=sha(source), control_sha256=sha(target),
                source_chars=len(text), control_chars=len(control), removed_sections=';'.join(removed)))
    assert seen == set(expected) and len(seen) == 1737
    with (DEST / 'input_manifest.csv').open('w') as h:
        writer = csv.DictWriter(h, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    hashes = {str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*')
              if p.is_file() and 'ontologies' not in p.parts and 'ontology_cache' not in p.parts}
    provenance = dict(projects=len(records), with_methods=sum(r['has_methods'] for r in records),
        abstract_only=sum(not r['has_methods'] for r in records),
        framework_commit=subprocess.check_output(['git','-C',str(TEXT),'rev-parse','HEAD'],text=True).strip(),
        frozen_hashes=hashes, reference_run='qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737',
        model='Qwen/Qwen3.8-27B', dtype='bfloat16', multi_value=True,
        note='Reuse production prompts and QC; strip PRIDE input only. 12 CPU cores per worker to fit one node; original requested 32. Historical repair outputs used repair config and must be identified in interpretation.')
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    (DEST / 'PREPARED').touch()
    print(json.dumps({k:v for k,v in provenance.items() if k!='frozen_hashes'},indent=2))


if __name__ == '__main__': main()
