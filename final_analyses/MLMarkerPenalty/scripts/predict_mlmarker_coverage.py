"""MLMarker predictions over the reprocessing quant matrix with a coverage rule.

For every run the share of the model's 5,979 features that is detected
(non-zero) is computed. Runs at or above COVERAGE_MIN are scored with the
Random Forest's predict_proba, which is what Arnaud's run used. Runs below
it are scored through the package's SHAP path with penalty_factor = 1, which
subtracts the contribution of absent proteins and removes the Bone marrow
default that low-coverage samples otherwise receive.

Output columns follow Arnaud's run_meta_mlmarker_all.tsv (pxd, run,
source_parquet, tissue, confidence, tissue_2 .. confidence_5) plus
coverage and scoring_mode.

Sharded: --shard i --nshards n writes results/predictions_shards/shard_i.tsv.
Run with the scratch venv that has a NumPy-2 compatible shap.
"""
from __future__ import annotations

import argparse, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/tine/git/Publication/MLMarker")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from mlmarker.model import MLMarker
from mlmarker.utils import validate_sample

ROOT = "/home/tine/git/Publication/textmining"
QUANT_DEFAULT = f"{ROOT}/MLMarker/Reprocessing_database/agentic_metadata/data/sample_nsaf_matrix.parquet"
META_COLS = {"pxd", "run", "source", "sample_base", "num_fractions", "total_psms", "quant_type"}
NAME = f"{ROOT}/MLMarker/Reprocessing_database/notebooks/run_meta_name.tsv"
OUT_DEFAULT = f"{ROOT}/final_analyses/MLMarkerPenalty/results/predictions_shards"
COVERAGE_MIN = 0.10
TOPN = 5


def dedupe(q: pd.DataFrame) -> pd.DataFrame:
    acq = pd.read_csv(NAME, sep="\t", usecols=["pxd", "run", "acquisition"]).drop_duplicates(["pxd", "run"])
    q = q.merge(acq, on=["pxd", "run"], how="left")
    dup = q.duplicated(["pxd", "run"], keep=False)
    mode = q.source.str.lower().map(lambda x: "DIA" if "dia" in x and "dda" not in x else "DDA")
    return q[~dup | (mode == q.acquisition)].drop(columns="acquisition").drop_duplicates(["pxd", "run"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--quant", default=QUANT_DEFAULT); ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--no-dedupe", action="store_true", help="matrix already has unique (pxd, run) keys")
    a = ap.parse_args(); QUANT = a.quant; OUT = a.out
    m0 = MLMarker(penalty_factor=0)
    m1 = MLMarker(penalty_factor=1, explainer=m0.explainability.explainer)
    feats = m0.features
    pf = pq.ParquetFile(QUANT)
    names = pf.schema_arrow.names
    keycols = [c for c in ("pxd", "run", "source") if c in names]
    keys = pq.read_table(QUANT, columns=keycols).to_pandas()
    if not a.no_dedupe and "source" in keycols:
        keys = dedupe(keys)
    mine = keys.iloc[a.shard::a.nshards]
    full = pf.read(columns=keycols + [f for f in feats if f in names]).to_pandas()
    full = full.merge(mine[keycols], on=keycols, how="inner").drop_duplicates(["pxd", "run"]).reset_index(drop=True)
    X = validate_sample(feats, full[[f for f in feats if f in full.columns]])
    cov = (X.values > 0).sum(axis=1) / len(feats)
    cls = m0.model.classes_
    proba = m0.model.predict_proba(X)
    rows = []; t0 = time.time(); n_shap = 0
    for i in range(len(X)):
        r = {"pxd": full.pxd[i], "run": full.run[i], "source_parquet": QUANT.rsplit("/", 1)[-1], "coverage": round(float(cov[i]), 4)}
        if cov[i] >= COVERAGE_MIN:
            order = np.argsort(-proba[i])[:TOPN]; preds = [(cls[j], float(proba[i][j])) for j in order]; r["scoring_mode"] = "predict_proba"
        else:
            m1.load_sample(X.iloc[[i]]); preds = [(t, float(s)) for t, s in m1.predict_top_tissues_shap(n_preds=TOPN)]; r["scoring_mode"] = "shap_penalty1"; n_shap += 1
        for k, (t, s) in enumerate(preds, 1):
            suf = "" if k == 1 else f"_{k}"; r[f"tissue{suf}"] = t; r[f"confidence{suf}"] = round(s, 4)
        rows.append(r)
        if i % 200 == 0: print(f"shard {a.shard}: {i}/{len(X)} shap={n_shap} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/shard_{a.shard}.tsv", sep="\t", index=False)
    print(f"shard {a.shard} done: {len(rows)} rows, {n_shap} via shap, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
