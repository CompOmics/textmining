"""Paired descriptive analysis; frozen HAMLET helpers, no extraction rerun."""
import collections
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
HAMLET = ROOT.parents[2]
SOURCE = ROOT / 'runs/job_965823/field_changes.csv'
OUT = ROOT / 'three_levels' / ('job_' + os.environ.get('SLURM_JOB_ID', 'local'))
FIELD_ONTOLOGY = {
    'species': 'species', 'tissue': 'uberon', 'cell_type': 'cl',
    'disease_state': 'mondo', 'sample_source': 'bto', 'cell_line': 'clo',
    'instrument': 'psi-ms', 'mass_analyzer': 'psi-ms',
    'fragmentation_method': 'psi-ms', 'ionization_type': 'psi-ms',
    'cleavage_agent': 'psi-ms', 'acquisition_method': 'psi-ms',
    'labeling': 'psi-ms', 'ptm': 'psimod',
    'reduction_reagent': 'chebi', 'alkylation_reagent': 'chebi',
}
PREFIX = {'species': 'NCBITaxon:', 'uberon': 'UBERON:', 'cl': 'CL:',
          'mondo': 'MONDO:', 'bto': 'BTO:', 'clo': 'CLO:',
          'psi-ms': 'MS:', 'psimod': 'MOD:', 'chebi': 'CHEBI:'}
NUMERIC = {'age', 'BMI', 'biological_replicate', 'technical_replicate',
           'reduction_concentration', 'alkylation_concentration', 'collision_energy'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def numeric(field):
    return field.startswith('number_of_') or field in NUMERIC


def key(value):
    return re.sub(r'[™®©]', '', value).casefold().strip()


def scalar(value):
    # Unit conversions only for explicitly recognised concentration units.
    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*(mm|µm|μm|um|nm|m|%)?', value)
    if not m:
        return None
    unit = m[2] or ''
    factor = {'m': 1, 'mm': 1e-3, 'µm': 1e-6, 'μm': 1e-6,
              'um': 1e-6, 'nm': 1e-9}.get(unit, 1)
    return float(m[1]) * factor, 'molar' if unit in {'m', 'mm', 'µm', 'μm', 'um', 'nm'} else unit


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    frozen = OUT / 'frozen'; frozen.mkdir()
    helpers = {}
    for name, rel in [('matcher', 'benchmark/semantic_matcher.py'),
                      ('ontology', 'normalization/ontology.py')]:
        src = HAMLET / 'textmining/framework' / rel
        data = src.read_bytes()
        (frozen / (name + '.py')).write_bytes(data)
        helpers[name] = hashlib.sha256(data).hexdigest()
    (frozen / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    matcher_mod = load_module('paired_hamlet_matcher', frozen / 'matcher.py')
    ontology_mod = load_module('paired_hamlet_ontology', frozen / 'ontology.py')
    rows = list(csv.DictReader(SOURCE.open()))
    assert len(rows) == 1737 * 38
    for r in rows:
        r['a'] = json.loads(r['old_values']); r['b'] = json.loads(r['new_values'])
    assert len({r['pxd'] for r in rows}) == 1737
    assert len({(r['pxd'], r['agent'], r['field']) for r in rows}) == len(rows)
    projects = {}; domains = collections.defaultdict(collections.Counter)
    for r in rows:
        a, b = r['a'], r['b']
        p = projects.setdefault(r['pxd'], {'pxd': r['pxd'], 'cohort': r['cohort'],
            'old_fields': 0, 'new_fields': 0, 'shared_fields': 0,
            'earlier_only_fields': 0, 'expanded_only_fields': 0,
            'old_values': 0, 'new_values': 0})
        counts = dict(old_fields=int(bool(a)), new_fields=int(bool(b)),
                      shared_fields=int(bool(a and b)), earlier_only_fields=int(bool(a and not b)),
                      expanded_only_fields=int(bool(b and not a)), old_values=len(a), new_values=len(b))
        for k, v in counts.items(): p[k] += v
        domains[r['agent']].update(counts)
    for p in projects.values():
        union = p['shared_fields'] + p['earlier_only_fields'] + p['expanded_only_fields']
        p['field_jaccard'] = p['shared_fields'] / union if union else None
        p['field_delta'] = p['new_fields'] - p['old_fields']
        p['value_delta'] = p['new_values'] - p['old_values']
    write_csv(OUT / 'dataset_amounts_overlap.csv', list(projects.values()))
    print('Levels 1 and 2 saved', flush=True)

    # Exact primary names / unambiguous EXACT synonyms only. No nearest-neighbour
    # ontology assignment: ambiguous and related labels must not imply ancestry.
    graphs = {}; ontology_hashes = {}
    for ont in sorted(set(FIELD_ONTOLOGY.values())):
        path = HAMLET / 'textmining/framework/ontologies' / (ont + ('.owl' if ont == 'clo' else '.obo'))
        graphs[ont] = ontology_mod.OntologyLoader().load(str(path))
        ontology_hashes[ont] = hashlib.sha256(path.read_bytes()).hexdigest()
        print('Loaded ontology', ont, len(graphs[ont]), flush=True)
    matcher = matcher_mod.HierarchicalMatcher(semantic_threshold=.70)
    matcher.term_normalizer = SimpleNamespace(graphs=graphs)
    resolutions = {}

    def resolve(field, value):
        cache_key = (field, value)
        if cache_key in resolutions: return resolutions[cache_key]
        ont = FIELD_ONTOLOGY.get(field)
        if not ont: return None
        graph = graphs[ont]
        direct = graph.get_node(value.upper())
        candidates = [direct] if direct else graph.get_primary_candidates(value)
        candidates = [n for n in candidates if n.id.startswith(PREFIX[ont])]
        if not candidates:
            candidates = [n for n, scope in graph.get_synonym_matches(value)
                          if scope == 'EXACT' and n.id.startswith(PREFIX[ont])]
        ids = {n.id for n in candidates}
        result = next(iter(ids)) if len(ids) == 1 else None
        resolutions[cache_key] = result
        return result

    def deterministic(field, a, b):
        if a == b: return 'EXACT_LEXICAL', 1., None, None, None
        if key(a) == key(b): return 'NORMALIZED', 1., None, None, None
        if numeric(field):
            x, y = scalar(a), scalar(b)
            if x is not None and y is not None and x[1] == y[1]:
                if np.isclose(x[0], y[0], rtol=1e-9, atol=0):
                    return 'NUMERIC_EQUIVALENT', 1., None, None, None
                return 'NUMERIC_DIFFERENCE', 0., None, None, None
            return 'UNRESOLVED_NUMERIC', 0., None, None, None
        x, y = resolve(field, a), resolve(field, b)
        if x and y:
            if x == y: return 'ONTOLOGY_EQUIVALENT', 1., x, y, 0
            ont = FIELD_ONTOLOGY[field]
            hops = matcher._find_ancestor_distance(x, y, ont)
            if hops is not None: return 'EXPANDED_NARROWER', .9, x, y, hops
            hops = matcher._find_ancestor_distance(y, x, ont)
            if hops is not None: return 'EXPANDED_BROADER', .9, x, y, hops
        return None

    candidates = set()
    for r in rows:
        candidates.update((r['field'], a, b) for a in r['a'] for b in r['b'])
    pair_results = {}; pending = []
    for field, a, b in sorted(candidates):
        result = deterministic(field, a, b)
        if result is None: pending.append((field, a, b))
        else: pair_results[(field, a, b)] = result
    terms = sorted({v for _, a, b in pending for v in [a, b]})
    print('Unique pairs', len(candidates), 'semantic pairs', len(pending), 'embedding strings', len(terms), flush=True)
    if terms:
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
        semantic = matcher_mod.SemanticMatcher(threshold=.70, device='cpu')
        model = semantic.model
        for start in range(0, len(terms), 64):
            batch = terms[start:start+64]
            inputs = semantic._tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.inference_mode():
                emb = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :], dim=1).numpy()
            semantic._cache.update(zip(batch, emb))
            if start % 1024 == 0: print('Embedded', start, '/', len(terms), flush=True)
        for field, a, b in pending:
            score = semantic.cosine_similarity(a, b)
            pair_results[(field, a, b)] = ('SEMANTIC_CANDIDATE' if score >= .70 else 'UNRESOLVED',
                                          score, resolve(field, a), resolve(field, b), None)
    print('Pair scoring complete', flush=True)
    write_csv(OUT / 'unique_pair_scores.csv', [dict(field=f, earlier=a, expanded=b, category=v[0],
              similarity=v[1], earlier_id=v[2], expanded_id=v[3], hops=v[4])
              for (f, a, b), v in sorted(pair_results.items())])
    write_csv(OUT / 'ontology_resolutions.csv', [dict(field=f, value=v, ontology_id=i)
              for (f, v), i in sorted(resolutions.items())])
    strict = {'EXACT_LEXICAL', 'NORMALIZED', 'NUMERIC_EQUIVALENT', 'ONTOLOGY_EQUIVALENT'}
    related = strict | {'EXPANDED_NARROWER', 'EXPANDED_BROADER'}
    accepted = related | {'SEMANTIC_CANDIDATE'}

    def assign(a, b, field, allowed):
        if not a or not b: return []
        weight = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                kind, score, *_ = pair_results[(field, x, y)]
                if kind in allowed:
                    # Maximise cardinality first, then prefer stronger tiers.
                    rank = 3 if kind in strict else 2 if kind in related else 1
                    weight[i, j] = 10 * (min(len(a), len(b)) + 1) + rank + score / 10
        ii, jj = linear_sum_assignment(-weight)
        return [(int(i), int(j)) for i, j in zip(ii, jj) if weight[i, j] > 0]

    field_rows = []; alignments = []; category_counts = collections.Counter()
    for r in rows:
        a, b, field = r['a'], r['b'], r['field']
        if not a and not b: continue
        strict_pairs = assign(a, b, field, strict)
        related_pairs = assign(a, b, field, related)
        pairs = assign(a, b, field, accepted)
        used_a = {i for i, j in pairs}; used_b = {j for i, j in pairs}
        if not a: status = 'expanded_only_field'
        elif not b: status = 'earlier_only_field'
        elif len(strict_pairs) == len(a) == len(b): status = 'equivalent_value_sets'
        elif len(related_pairs) == len(a) == len(b): status = 'hierarchically_related_value_sets'
        elif len(pairs) == len(a) == len(b): status = 'semantic_candidate_value_sets'
        elif pairs: status = 'partial_overlap'
        else: status = 'no_accepted_overlap_review'
        category_counts[status] += 1
        fr = dict(pxd=r['pxd'], cohort=r['cohort'], agent=r['agent'], field=field,
                  old_count=len(a), new_count=len(b), status=status,
                  equivalent_pairs=len(strict_pairs), hierarchical_inclusive_pairs=len(related_pairs),
                  semantic_inclusive_pairs=len(pairs),
                  equivalent_jaccard=len(strict_pairs)/(len(a)+len(b)-len(strict_pairs)),
                  unmatched_earlier=[x for i,x in enumerate(a) if i not in used_a],
                  unmatched_expanded=[y for j,y in enumerate(b) if j not in used_b])
        field_rows.append(fr)
        p = projects[r['pxd']]
        for k in ['equivalent_pairs', 'hierarchical_inclusive_pairs', 'semantic_inclusive_pairs']:
            p[k] = p.get(k, 0) + fr[k]
        for i, j in pairs:
            v = pair_results[(field, a[i], b[j])]
            alignments.append(dict(pxd=r['pxd'], agent=r['agent'], field=field,
                earlier=a[i], expanded=b[j], category=v[0], similarity=v[1],
                earlier_id=v[2], expanded_id=v[3], hops=v[4]))
    for p in projects.values():
        for k in ['equivalent_pairs', 'hierarchical_inclusive_pairs', 'semantic_inclusive_pairs']:
            m = p.get(k, 0); union = p['old_values'] + p['new_values'] - m
            p[k + '_jaccard'] = m / union if union else None
    write_csv(OUT / 'dataset_amounts_overlap.csv', list(projects.values()))
    write_csv(OUT / 'field_value_agreement.csv', field_rows)
    write_csv(OUT / 'matched_value_pairs.csv', alignments)
    write_csv(OUT / 'review_no_overlap.csv', [r for r in field_rows if r['status'] == 'no_accepted_overlap_review'])
    totals = {k: sum(p[k] for p in projects.values()) for k in ['old_fields', 'new_fields',
        'shared_fields', 'earlier_only_fields', 'expanded_only_fields', 'old_values', 'new_values',
        'equivalent_pairs', 'hierarchical_inclusive_pairs', 'semantic_inclusive_pairs']}
    summary = dict(projects=len(projects), totals=totals, domains=dict(domains),
        field_value_categories=dict(category_counts),
        selected_pair_categories=dict(collections.Counter(r['category'] for r in alignments)),
        same_field_count_projects=sum(p['field_delta']==0 for p in projects.values()),
        same_value_count_projects=sum(p['value_delta']==0 for p in projects.values()),
        helper_sha256=helpers, ontology_sha256=ontology_hashes,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        semantic_threshold=.70,
        limitations=['Earlier versus expanded workflow, not independent manuscript versus PRIDE sources',
            'Casefolded values and semicolon splitting inherited from previous audit',
            'Counts include schema aliases; lexical values are not independent scientific facts',
            'Ontology equivalence restricted to unique primary names/EXACT synonyms; no fuzzy mapping',
            'Hierarchy uses HAMLET is_a traversal, maximum four hops; part_of not modelled',
            'Semantic candidates are not proven equivalence; unmatched values are not contradictions',
            'Scalar differences flag review, not errors; numeric definitions and scope may differ',
            'No precision/recall/F1: neither extraction is ground truth'])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'COMPLETE').touch()
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
