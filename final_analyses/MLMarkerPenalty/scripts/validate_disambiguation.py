"""Ground truth for the file-level disambiguation, from SDRFs and a sample map.

Sources (per project, never HAMLET):
  PXD000561  results/fig5_concordance_figure/sdrf(1).tsv (PRIDE SDRF; source name = matrix run)
  PXD020192  MLMarker/Reprocessing_database/agentic_metadata/cache/sdrf/PXD020192.sdrf.tsv
  PXD010154  MLMarker/Reprocessing_database/agentic_metadata/cache/sdrf/PXD010154.sdrf.tsv
  PXD048734  results/fig5_concordance_figure/Type.xlsx (sample ID -> cancer type, i.e. tissue of origin)
  PXD006401  SDRF via run_labels.tsv (human tonsil files; the kidney files are mouse and excluded)
PXD010296 is dropped: its spleen files are rat, scored by a human model.

Writes results/fig5_concordance_figure/disambiguation_truth.csv and
disambiguation_validation.csv; the figure script picks the truth up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
RES = ROOT / "results"
CF = RES / "fig5_concordance_figure"
CACHE = REPO / "MLMarker/Reprocessing_database/agentic_metadata/cache/sdrf"
sys.path.insert(0, str(HERE))
from granularity_and_disease import to_class  # noqa: E402

EXTRA = {"frontal cortex": "Brain", "cerebellum": "Brain", "medulla oblongata": "Brain", "parietal": "Brain",
         "temporal": "Brain", "occipital": "Brain", "adrenalgland": "Adrenal gland", "adipose": "Adipose tissue",
         "pituitary/hypophysis": "Pituitary gland", "appendix/vermiform appendix": "Appendix",
         "endometrium/uterine endometrium": "Endometrium", "b cells": "B-cells", "monocytes": "Monocytes",
         "lymph node": "Lymph node", "smooth muscle": "Smooth muscle", "urinary bladder": "Urinary bladder",
         "salivary gland": "Salivary gland", "thyroid": "Thyroid", "rectum": "Rectum", "tonsil": "Tonsil",
         "occipital cortex": "Brain", "bladder": "Urinary bladder", "fat/adipose tissue": "Adipose tissue", "fallopian tube/oviduct": "Oviduct"}
DROP = {"PXD010296"}


def cls(term):
    if not isinstance(term, str):
        return None
    t = term.strip().lower()
    return EXTRA.get(t) or to_class(t)


def sdrf_truth(path, pxd):
    sd = pd.read_csv(path, sep="\t", dtype=str)
    sd.columns = [c.lower() for c in sd.columns]
    part = [c for c in sd.columns if "organism part" in c][0]
    t = sd.groupby("source name")[part].agg(lambda v: "; ".join(sorted(set(v)))).reset_index()
    t.columns = ["run", "truth_raw"]
    t["pxd"] = pxd
    return t


def main():
    truth = [sdrf_truth(CF / "sdrf(1).tsv", "PXD000561"),
             sdrf_truth(CACHE / "PXD020192.sdrf.tsv", "PXD020192"),
             sdrf_truth(CACHE / "PXD010154.sdrf.tsv", "PXD010154")]
    x = pd.read_excel(CF / "Type.xlsx")
    truth.append(pd.DataFrame({"pxd": "PXD048734", "run": x["ID"].astype(str), "truth_raw": x["Cancer Type"].str.lower()}))
    lab = pd.read_csv(ROOT / "results/run_labels.tsv", sep="\t", low_memory=False)
    l6 = lab[(lab.pxd == "PXD006401") & (lab.tissue_source == "sdrf") & (lab.organism == "homo sapiens")]
    truth.append(pd.DataFrame({"pxd": "PXD006401", "run": l6.run, "truth_raw": l6.tissue}))
    truth = pd.concat(truth, ignore_index=True)
    truth["truth_class"] = truth.truth_raw.map(lambda s: cls(s) if ";" not in str(s) else None)
    truth["source"] = truth.pxd.map({"PXD000561": "PRIDE SDRF", "PXD020192": "SDRF", "PXD010154": "SDRF",
                                     "PXD048734": "sample map (Type.xlsx)", "PXD006401": "SDRF"})

    m = pd.read_csv(RES / "granularity/A_multi_tissue_samples.csv")
    m = m[~m.pxd.isin(DROP)].merge(truth, on=["pxd", "run"], how="left")
    m["in_training"] = m.pxd.isin(set(lab[lab.in_old_atlas.fillna(False).astype(bool)].pxd))
    m["cands"] = m.cand_classes.map(eval)
    v = m[m.truth_raw.notna() & m.usable].copy()
    v["truth_in_vocab"] = v.truth_class.notna()
    v["truth_in_list"] = [t in c if t else False for t, c in zip(v.truth_class, v.cands)]
    v["unrestricted_ok"] = v.pred_tissue == v.truth_class
    v["chosen_ok"] = v.assigned == v.truth_class
    truth.merge(m[["pxd", "run", "in_training"]], on=["pxd", "run"], how="left").to_csv(CF / "disambiguation_truth.csv", index=False)

    rows = []
    def summarize(name, s):
        w = s[s.truth_in_list]
        return {"subset": name, "projects": s.pxd.nunique(), "files": len(s),
                "truth in MLMarker vocabulary": round(s.truth_in_vocab.mean(), 3),
                "truth in HAMLET list": round(s.truth_in_list.mean(), 3),
                "chance 1/k": round((1 / s.k).mean(), 3),
                "MLMarker alone == truth (all files)": round(s.unrestricted_ok.mean(), 3),
                "MLMarker alone == truth | truth in list": round(w.unrestricted_ok.mean(), 3) if len(w) else np.nan,
                "HAMLET list + MLMarker choice == truth | truth in list": round(w.chosen_ok.mean(), 3) if len(w) else np.nan,
                "choice == truth | truth in list, conf >= 0.3": round(w[w.confidence >= 0.3].chosen_ok.mean(), 3) if (w.confidence >= 0.3).any() else np.nan,
                "files with conf >= 0.3 (of those)": int((w.confidence >= 0.3).sum()),
                "MLMarker alone == truth | in vocab, missed by HAMLET list": round(s[s.truth_in_vocab & ~s.truth_in_list].unrestricted_ok.mean(), 3) if (s.truth_in_vocab & ~s.truth_in_list).any() else np.nan,
                "files in vocab, missed by HAMLET list": int((s.truth_in_vocab & ~s.truth_in_list).sum())}
    for name, s in [("all", v), ("not in MLMarker training", v[~v.in_training]), ("in MLMarker training", v[v.in_training])]:
        rows.append(summarize(name, s))
    for pxd, s in v.groupby("pxd"):
        r = summarize(pxd + (" (training)" if s.in_training.iloc[0] else ""), s); r["k"] = int(s.k.iloc[0]); rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(CF / "disambiguation_validation.csv", index=False)
    v[["pxd", "run", "in_training", "k", "truth_raw", "truth_class", "truth_in_list", "pred_tissue", "confidence", "assigned",
       "unrestricted_ok", "chosen_ok"]].to_csv(CF / "disambiguation_validation_files.csv", index=False)
    # what HAMLET's list misses and what MLMarker cannot name
    oov = v[~v.truth_in_vocab].truth_raw.value_counts()
    missing = v[v.truth_in_vocab & ~v.truth_in_list].groupby("pxd").truth_class.value_counts()
    pd.set_option("display.width", 220)
    print(out.to_string(index=False)); print("\ntruth outside MLMarker's classes:", oov.to_dict()); print("\nin vocabulary but not in HAMLET's list:"); print(missing.to_string())


if __name__ == "__main__":
    main()
