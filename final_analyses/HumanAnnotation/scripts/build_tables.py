"""Raw brat annotations -> the six tables behind manuscript Figure 1.

Every table here feeds a panel. Nothing is computed that no panel uses.

Outputs (results/):
  pair_agreement_value_string.csv    pooled value kappa, exact string identity     -> panel a, lower
  pair_agreement_value_sapbert.csv   pooled value kappa, SapBERT-cluster identity  -> panel a, upper
  label_agreement_presence.csv       per-label presence kappa, every pair          -> panel c
  label_mention_rate.csv             per-label mention rate in the consensus       -> panel c
  value_match_by_label.csv           exact / SapBERT-only / no match, per label    -> panel d
  value_movement_by_category.csv     string vs SapBERT value kappa, per category   -> panel e
  agreement_decomposition.csv        2x2 decomposition of presence agreement       -> panel b
  figure1_stats.json                 the numbers the panels annotate themselves with

Usage: python final_analyses/HumanAnnotation/scripts/build_tables.py
"""
import itertools
import json
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, wilcoxon

import common as m

RESULTS = m.RESULTS

# Human-versus-human means every pair of human annotators, and that includes
# the single expert. SingleHuman is Ian, who annotated all 27 manuscripts,
# where A1-A8 each covered an overlapping batch; his MultiHuman/Ian/ copy is
# dropped upstream as byte-identical, so he appears under this name alone.
# Filtering to MultiHuman-vs-MultiHuman silently drops a whole rater.
HUMAN_PAIRS = {
    "MultiHuman annotator vs. MultiHuman annotator",
    "SingleHuman vs. MultiHuman annotator",
}



def rater_permutation_test(pair_values, raters, model_raters, metric_name=""):
    """Exact permutation test for model-model vs human-human, permuting which
    RATERS are models rather than which pairs are model pairs.

    A rank test over annotator pairs would assume the pairs are independent,
    and they are not: each rater appears in up to eleven of them, so one
    unusual annotator moves many pairs at once. Permuting rater group
    membership respects that structure. With 12 raters and 3 models there are
    only C(12,3) = 220 assignments, so the test is enumerated exactly rather
    than sampled, and the smallest attainable two-sided p is 1/220 = 0.0045.
    """
    k = len(model_raters)

    def group_means(model_set):
        mm, hh = [], []
        for (a, b), v in pair_values.items():
            am, bm = a in model_set, b in model_set
            if am and bm:
                mm.append(v)
            elif not am and not bm:
                hh.append(v)
        if not mm or not hh:
            return None
        return st.fmean(mm) - st.fmean(hh)

    observed = group_means(set(model_raters))
    if observed is None:
        return None
    diffs = [d for d in (group_means(set(c))
                         for c in itertools.combinations(raters, k)) if d is not None]
    n_extreme = sum(1 for d in diffs if abs(d) >= abs(observed) - 1e-12)
    return {"metric": metric_name, "observed_diff": observed,
            "n_permutations": len(diffs), "p_two_sided": n_extreme / len(diffs)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = m.load_filtered_data()
    annotator_docs, annotators, labels = d["annotator_docs"], d["annotators"], d["labels"]
    stats = {}

    print(f"Kept {len(labels)}/{len(d['all_labels'])} labels "
          f"(mentioned in >= {m.MIN_MENTIONS}/{d['n_docs_total']} papers AND tagged by "
          f">= {m.MIN_UNIQUE_ANNOTATORS} unique MultiHuman annotators).")
    print(f"  dropped, too few mentions:   {d['dropped_low_mentions']}")
    print(f"  dropped, single annotator:   {d['dropped_single_annotator']}")
    stats["n_labels"] = len(labels)
    stats["n_docs"] = d["n_docs_total"]
    stats["n_identities"] = len(annotators)

    pd.DataFrame([{"anonymized_id": a, "real_name": r} for a, r in d["name_mapping"].items()]).to_csv(
        m.ANN_ROOT / "annotator_name_mapping.csv", index=False)

    presence = m.build_presence(annotator_docs, labels)

    # -- panel c, y axis: per-label presence kappa ----------------------------
    rows = []
    for a, b in itertools.combinations(annotators, 2):
        cat = m.pair_category(a, b)
        for lab in labels:
            kappa, n_shared = m.per_label_kappa(presence, a, b, lab)
            rows.append({"label": lab, "category": m.LABEL_CATEGORY[lab],
                         "annotator_a": a, "annotator_b": b, "pair_category": cat,
                         "n_shared_docs": n_shared, "kappa": kappa})
    label_pairs = pd.DataFrame(rows)
    label_pairs.to_csv(RESULTS / "label_agreement_presence.csv", index=False)

    # -- panel c, x axis: how often the consensus tags each label -------------
    harmonized = presence["HarmonizedHuman"]
    n_docs = len(harmonized)
    mention_df = pd.DataFrame([
        {"label": lab, "category": m.LABEL_CATEGORY[lab], "n_docs": n_docs,
         "mention_rate": sum(dp[lab] for dp in harmonized.values()) / n_docs}
        for lab in labels])
    mention_df.to_csv(RESULTS / "label_mention_rate.csv", index=False)

    # -- panel b: is model-model agreement just shared absence? --------------
    # The harmonised consensus is derived from the annotators, so pairing it
    # with them measures reconciliation rather than independent agreement.
    raters = [r for r in annotators if r != "HarmonizedHuman"]
    dec = [r for r in (m.decompose_pair(presence, labels, a, b)
                       for a, b in itertools.combinations(raters, 2)) if r]
    dec_df = pd.DataFrame(dec)
    dec_df.to_csv(RESULTS / "agreement_decomposition.csv", index=False)

    prev = {r: m.prevalence(presence, labels, r) for r in raters}
    pd.DataFrame({"rater": list(prev), "prevalence": list(prev.values())}).to_csv(RESULTS / "rater_prevalence.csv", index=False)
    humans = [v for k, v in prev.items() if k not in m.MODEL_DIRS]
    models = [v for k, v in prev.items() if k in m.MODEL_DIRS]
    stats["prevalence"] = {
        "human_min": min(humans), "human_max": max(humans),
        "human_fold": max(humans) / min(humans),
        "model_min": min(models), "model_max": max(models),
        "model_fold": max(models) / min(models),
    }
    stats["decomposition"] = {
        g: {c: float(sub[c].mean()) for c in
            ("kappa", "raw_agreement", "chance_agreement", "positive_agreement", "both_absent_share")}
        | {"n_pairs": int(len(sub))}
        for g, sub in dec_df.groupby("group")}

    print("\ngroup           n   kappa     raw  chance  pos.agr  both-absent")
    for g in ("human-human", "model-human", "model-model"):
        s = stats["decomposition"].get(g)
        if s:
            print(f"{g:14s} {s['n_pairs']:3d} {s['kappa']:7.3f} {s['raw_agreement']:7.3f} "
                  f"{s['chance_agreement']:7.3f} {s['positive_agreement']:8.3f} "
                  f"{s['both_absent_share']:12.3f}")
    p = stats["prevalence"]
    print(f"prevalence: humans {p['human_min']:.3f}-{p['human_max']:.3f} "
          f"({p['human_fold']:.1f}-fold), models {p['model_min']:.3f}-{p['model_max']:.3f} "
          f"({p['model_fold']:.2f}-fold)")

    # -- significance tests for panel b --------------------------------------
    # Three questions, and they need different tests.
    stats["tests"] = {}

    # (1) Is agreement above chance at all? Paired within each pair, so a
    # signed-rank test on (raw - chance). The pairs are not independent, which
    # makes the p-value anticonservative; the robust statement is the count,
    # since the difference is positive in every single pair.
    raw_vs_chance = {}
    for g, sub in dec_df.groupby("group"):
        d = (sub["raw_agreement"] - sub["chance_agreement"]).dropna()
        entry = {"n": int(len(d)), "n_positive": int((d > 0).sum()),
                 "mean_excess": float(d.mean())}
        if len(d) >= 6:
            stat, pv = wilcoxon(d, alternative="greater")
            entry["wilcoxon_p"] = float(pv)
        else:
            # n=3 cannot reach 0.05 in a signed-rank test, so report the exact
            # sign-test floor instead of a p-value that only looks like one
            entry["sign_test_p"] = float(0.5 ** len(d))
        raw_vs_chance[g] = entry
    stats["tests"]["raw_vs_chance"] = raw_vs_chance

    # (2) Does kappa differ across the three groups at all? Kruskal-Wallis,
    # with the same independence caveat.
    for metric in ("kappa", "positive_agreement", "chance_agreement", "both_absent_share"):
        groups = [sub[metric].dropna().values for _, sub in dec_df.groupby("group")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2 and sum(len(g) for g in groups) > len(groups):
            stat, pv = kruskal(*groups)
            stats["tests"].setdefault("kruskal", {})[metric] = {
                "H": float(stat), "p": float(pv),
                "n_per_group": [int(len(g)) for g in groups]}

    # (3) Model-model against human-human specifically, by the rater-level
    # permutation test, which is the one that survives the dependence.
    model_raters = [r for r in raters if r in m.MODEL_DIRS]
    stats["tests"]["permutation"] = {}
    for metric in ("kappa", "positive_agreement", "chance_agreement",
                   "both_absent_share", "raw_agreement"):
        pv = {(r["annotator_a"], r["annotator_b"]): r[metric] for _, r in dec_df.iterrows()}
        res = rater_permutation_test(pv, raters, model_raters, metric)
        if res:
            stats["tests"]["permutation"][metric] = res

    print("\nmodel-model vs human-human, exact rater permutation test:")
    for metric, res in stats["tests"]["permutation"].items():
        print(f"  {metric:20s} diff {res['observed_diff']:+.3f}  "
              f"p = {res['p_two_sided']:.4f}  ({res['n_permutations']} permutations)")
    print("raw agreement above chance:")
    for g, e in raw_vs_chance.items():
        extra = (f"Wilcoxon p={e['wilcoxon_p']:.2g}" if "wilcoxon_p" in e
                 else f"sign-test floor p={e['sign_test_p']:.2g}")
        print(f"  {g:14s} +{e['mean_excess']:.3f} in {e['n_positive']}/{e['n']} pairs, {extra}")

    # -- panels a (upper), d and e all need SapBERT --------------------------
    print("\nLoading SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)...")
    scorer = m.SapBertScorer()

    # -- panel d: exact vs conceptual match, per label -----------------------
    value_df = m.value_agreement(annotator_docs, presence, labels, scorer)
    summary = value_df.groupby("label").agg(
        n=("exact_match", "size"), exact=("exact_match", "sum"), semantic=("semantic_match", "sum"),
    ).reset_index()
    summary["category"] = summary["label"].map(m.LABEL_CATEGORY)
    summary["exact_pct"] = 100 * summary["exact"] / summary["n"]
    summary["semantic_pct"] = 100 * summary["semantic"] / summary["n"]
    summary["no_match_pct"] = 100 - summary["exact_pct"] - summary["semantic_pct"]
    summary.to_csv(RESULTS / "value_match_by_label.csv", index=False)
    stats["value_match"] = {
        "n_instances": int(len(value_df)),
        "exact_pct": 100 * float(value_df["exact_match"].mean()),
        "semantic_pct": 100 * float(value_df["semantic_match"].mean()),
    }
    stats["value_match"]["no_match_pct"] = (
        100 - stats["value_match"]["exact_pct"] - stats["value_match"]["semantic_pct"])

    string_cats = m.value_categories(annotator_docs, labels, "string")
    sapbert_cats = m.value_categories(annotator_docs, labels, "sapbert", scorer=scorer)

    # -- panel a: pooled value kappa, both scoring schemes -------------------
    # The two triangles of panel a. Same items, same pooling, one difference:
    # whether two values count as the same category only when the strings are
    # identical, or also when SapBERT puts them within cosine 0.70. That makes
    # the triangles directly comparable, which presence kappa never was.
    string_pairs = value_pairs = None
    for scheme, cats, fname in (("string", string_cats, "pair_agreement_value_string.csv"),
                                ("sapbert", sapbert_cats, "pair_agreement_value_sapbert.csv")):
        rows = []
        for a, b in itertools.combinations(annotators, 2):
            kappa, n_shared = m.pooled_multicat_kappa(cats, labels, a, b)
            rows.append({"annotator_a": a, "annotator_b": b,
                         "pair_category": m.pair_category(a, b),
                         "n_shared": n_shared, "kappa": kappa})
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / fname, index=False)
        if scheme == "string":
            string_pairs = df
        else:
            value_pairs = df

    # -- panel e: what forgiving wording does, by metadata category ----------
    # Pooled per category rather than per label. Forty-two separate label-level
    # shifts is a list, not a result; the question is whether forgiving wording
    # helps biological, technical and experimental-design fields differently,
    # and that needs three numbers.
    by_cat = defaultdict(list)
    for lab in labels:
        by_cat[m.LABEL_CATEGORY[lab]].append(lab)
    rows = []
    for cat in m.CAT_ORDER:
        cat_labels = by_cat.get(cat, [])
        if not cat_labels:
            continue
        for a, b in itertools.combinations(annotators, 2):
            k_str, n_str = m.pooled_multicat_kappa(string_cats, cat_labels, a, b)
            k_sap, n_sap = m.pooled_multicat_kappa(sapbert_cats, cat_labels, a, b)
            if not n_str or np.isnan(k_str) or np.isnan(k_sap):
                continue
            rows.append({"category": cat, "n_labels": len(cat_labels),
                         "annotator_a": a, "annotator_b": b,
                         "pair_category": m.pair_category(a, b),
                         "n_items": n_str, "string_kappa": k_str,
                         "sapbert_kappa": k_sap, "shift": k_sap - k_str})
    movement = pd.DataFrame(rows)
    movement.to_csv(RESULTS / "value_movement_by_category.csv", index=False)

    human_mv = movement[movement["pair_category"].isin(HUMAN_PAIRS)]
    stats["movement"] = {
        cat: {"n_labels": int(sub["n_labels"].iloc[0]),
              "n_pairs": int(len(sub)),
              "string_kappa": float(sub["string_kappa"].mean()),
              "sapbert_kappa": float(sub["sapbert_kappa"].mean()),
              "shift": float(sub["shift"].mean())}
        for cat, sub in human_mv.groupby("category")}
    stats["movement_pooled"] = {
        "string_kappa": float(human_mv["string_kappa"].mean()),
        "sapbert_kappa": float(human_mv["sapbert_kappa"].mean()),
    }

    # -- panel c annotation: does rarity explain difficulty? -----------------
    human_label_pairs = label_pairs[label_pairs["pair_category"].isin(HUMAN_PAIRS)]
    mean_kappa = human_label_pairs.groupby("label")["kappa"].mean()
    rho, pval = spearmanr(mean_kappa.reindex(mention_df["label"]), mention_df["mention_rate"])
    stats["rarity_vs_difficulty"] = {"rho": float(rho), "p": float(pval),
                                     "n_labels": int(len(mention_df))}

    # presence kappa, for panel b and for the manuscript text. Read off the
    # decomposition rows so there is a single source for it.
    stats["presence_kappa"] = {
        g.replace("-", "_"): {"mean": float(sub["kappa"].mean()),
                              "min": float(sub["kappa"].min()),
                              "max": float(sub["kappa"].max()),
                              "n": int(len(sub))}
        for g, sub in dec_df.groupby("group")}

    # the two triangles of panel a, summarised per group
    def group_summary(df):
        sel = {
            "human_human": df["pair_category"].isin(HUMAN_PAIRS),
            "model_human": df["pair_category"] == "model vs. any human annotator",
            "model_model": df["pair_category"] == "model vs. model",
        }
        out = {}
        for g, mask in sel.items():
            k = df.loc[mask, "kappa"].dropna()
            out[g] = {"mean": float(k.mean()), "min": float(k.min()),
                      "max": float(k.max()), "n": int(len(k))}
        return out

    stats["value_kappa_string"] = group_summary(string_pairs)
    stats["value_kappa"] = group_summary(value_pairs)

    # SapBERT-cluster identity can only merge value categories, so it cannot
    # lower raw agreement. Kappa's chance correction can still move the wrong
    # way when merging changes the marginals, so report how often it does
    # rather than claiming it never happens.
    cmp = string_pairs.merge(value_pairs, on=["annotator_a", "annotator_b"],
                             suffixes=("_string", "_sapbert"))
    cmp["gain"] = cmp["kappa_sapbert"] - cmp["kappa_string"]
    stats["value_kappa_gain"] = {
        "n_pairs": int(cmp["gain"].notna().sum()),
        "n_up": int((cmp["gain"] > 1e-9).sum()),
        "n_down": int((cmp["gain"] < -1e-9).sum()),
        "mean_gain": float(cmp["gain"].mean()),
        "min_gain": float(cmp["gain"].min()),
        "max_gain": float(cmp["gain"].max()),
    }
    g = stats["value_kappa_gain"]
    print(f"\nvalue kappa, string -> SapBERT over {g['n_pairs']} pairs: "
          f"mean {g['mean_gain']:+.3f}, up in {g['n_up']}, down in {g['n_down']} "
          f"(range {g['min_gain']:+.3f} to {g['max_gain']:+.3f})")
    # SapBERT-cluster identity can only merge value categories, so it cannot
    # lower value agreement except through kappa's chance correction. Report
    # how often it does, since "it always goes up" would be overclaiming.
    down = int((movement["shift"] < -1e-9).sum())
    stats["movement_direction"] = {
        "n_category_pairs": int(len(movement)), "n_down": down,
        "worst_down": float(movement["shift"].min()),
    }

    with open(RESULTS / "figure1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nrarity vs difficulty: rho={rho:.3f}, p={pval:.3g} over {len(mention_df)} labels")
    print(f"value match: exact={stats['value_match']['exact_pct']:.1f}%  "
          f"SapBERT-only={stats['value_match']['semantic_pct']:.1f}%  "
          f"none={stats['value_match']['no_match_pct']:.1f}%")
    print("forgiving wording, human pairs, pooled per category:")
    for cat, s in stats["movement"].items():
        print(f"  {cat:20s} {s['string_kappa']:.3f} -> {s['sapbert_kappa']:.3f} "
              f"({s['shift']:+.3f}) over {s['n_labels']} labels")
    print(f"\nDone. {len(annotators)} identities, {len(labels)} labels, {n_docs} papers.")


if __name__ == "__main__":
    main()
