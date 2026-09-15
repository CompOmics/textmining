"""Figure 1: how well do expert annotators agree on proteomics metadata, and
how do three models compare?

Reads only the tables written by build_tables.py, so the figure can never
disagree with the analysis. Nothing is recomputed here.

Panels:
  a  pairwise value-level agreement, every identity. Lower triangle exact
     string identity, upper triangle SapBERT-cluster identity
  b  decomposition of presence agreement: model-model agreement is not an
     artefact of both models declining to annotate
  c  agreement against how often a field is stated at all
  d  value-level agreement: exact, recovered by SapBERT, residual mismatch
  e  what forgiving wording does, pooled per metadata category

Usage: python final_analyses/HumanAnnotation/scripts/build_figure1.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent          # final_analyses/HumanAnnotation/scripts
ANALYSIS = HERE.parent                          # final_analyses/HumanAnnotation
REPO = ANALYSIS.parents[1]                      # repository root
RESULTS = ANALYSIS / "results"
FIGDIR = ANALYSIS / "figures/manuscript_figure1"
OUT = FIGDIR / "figure1.png"
PANELS = FIGDIR / "panels"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "final_analyses"))
from plot_style import clean_axes  # noqa: E402
from figure_style import (AGENT, NEUTRAL, LIN_KIND, style, panel_title, legend_outside,  # noqa: E402
                          save_composite, save_panels)

style()
MUTED = "#6b7a76"
GRID = "#e4e9e7"

# categories use the agent palette, so biological / technical / experimental
# design look the same as in every other figure
CAT_COLOR = {
    "biological": AGENT["BiologicalAgent"],
    "technical": AGENT["TechnicalAgent"],
    "experimental_design": AGENT["ExperimentalDesignAgent"],
}
CAT_LABEL = {"biological": "biological", "technical": "technical",
             "experimental_design": "exp. design"}
CAT_ORDER = ["biological", "technical", "experimental_design"]

# the three groups of the decomposition panel, in the order they are argued
GROUP_ORDER = ["human-human", "model-human", "model-model"]
GROUP_COLOR = {"human-human": NEUTRAL, "model-human": "#7a9cc6", "model-model": "#b8866b"}

# human-versus-human is every pair of human annotators, the single expert
# included. See the note in build_tables.py.
HUMAN_PAIRS = {
    "MultiHuman annotator vs. MultiHuman annotator",
    "SingleHuman vs. MultiHuman annotator",
}
MODEL_ORDER = ["Qwen", "Gemma", "GLM"]


def load(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def stats():
    with open(RESULTS / "figure1_stats.json") as f:
        return json.load(f)


def fnum(s):
    """Float, or nan for the blank cells kappa leaves where a pair shares no papers."""
    s = (s or "").strip()
    return float(s) if s else float("nan")


# -----------------------------------------------------------------------------
def panel_a(ax, cbar_lower=None, cbar_upper=None):
    """One matrix, two questions.

    Both triangles are value-level kappa over the same items with the same
    pooling. The only difference is what counts as the same value: exact
    string identity in the lower triangle, SapBERT-cluster identity in the
    upper one, where two values within cosine 0.70 share a category. One
    matrix rather than two, since it is symmetric.

    That makes the two triangles a genuine before and after. Merging
    categories can only raise raw agreement, so the upper triangle should
    exceed the lower one, and it does for every pair but a handful where
    kappa's chance correction moves against the merge. An earlier version of
    this panel put presence kappa in the lower triangle, which is a different
    and easier question and so read as SapBERT lowering agreement.

    Greens and Blues are used deliberately: both are ColorBrewer single-hue
    sequential ramps with near-identical lightness profiles, so equal kappas
    look equally dark across the diagonal. Greens against Purples does not
    have that property and creates a drop that is not in the data.
    """
    pres = {(r["annotator_a"], r["annotator_b"]): (fnum(r["kappa"]), fnum(r["n_shared"]))
            for r in load("pair_agreement_value_string.csv")}
    val = {(r["annotator_a"], r["annotator_b"]): fnum(r["kappa"])
           for r in load("pair_agreement_value_sapbert.csv")}

    present = {x for pair in pres for x in pair}
    multi = sorted((x for x in present if x.startswith("Annotator")),
                   key=lambda n: int(n.removeprefix("Annotator")))
    humans = (["SingleHuman"] if "SingleHuman" in present else []) + multi
    consensus = ["HarmonizedHuman"] if "HarmonizedHuman" in present else []
    models = [x for x in MODEL_ORDER if x in present]
    order = humans + consensus + models
    n = len(order)
    idx = {k: i for i, k in enumerate(order)}

    # SingleHuman is a distinct role, one expert over all 27 manuscripts,
    # where A1-A8 each covered an overlapping batch. Numbering it among them
    # would hide that, so it keeps its own label.
    short = {"SingleHuman": "single\nexpert", "HarmonizedHuman": "consensus"}
    short.update({h: f"A{i + 1}" for i, h in enumerate(multi)})
    short.update({x: x for x in models})

    P = np.full((n, n), np.nan)
    V = np.full((n, n), np.nan)
    N = np.zeros((n, n))
    for (a, b), (k, nd) in pres.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            P[i, j] = P[j, i] = k
            N[i, j] = N[j, i] = 0 if np.isnan(nd) else nd
    for (a, b), k in val.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            V[i, j] = V[j, i] = k

    lower = np.ma.array(P, mask=np.triu(np.ones_like(P, dtype=bool)) | np.isnan(P))
    upper = np.ma.array(V, mask=np.tril(np.ones_like(V, dtype=bool)) | np.isnan(V))

    cm_s = plt.get_cmap("Greens").copy()
    cm_b = plt.get_cmap("Blues").copy()
    for cm in (cm_s, cm_b):
        cm.set_bad(alpha=0)
    im_p = ax.imshow(lower, cmap=cm_s, vmin=0, vmax=1)
    im_v = ax.imshow(upper, cmap=cm_b, vmin=0, vmax=1)
    ax.set_aspect("equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = P[i, j] if i > j else V[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.0,
                        color="0.62", style="italic")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                        color="white" if v > 0.58 else "0.2")
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#f4f6f5",
                                   edgecolor="white", lw=0.6, zorder=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([short[k] for k in order], fontsize=6.6, rotation=45, ha="right")
    ax.set_yticklabels([short[k] for k in order], fontsize=6.6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # separate the human block, the consensus, and the models
    sep = len(humans) + len(consensus) - 0.5
    ax.axhline(sep, color="0.35", lw=1.0)
    ax.axvline(sep, color="0.35", lw=1.0)
    if "SingleHuman" in idx:
        ax.axhline(0.5, color="0.6", lw=0.8, ls=(0, (3, 2)))
        ax.axvline(0.5, color="0.6", lw=0.8, ls=(0, (3, 2)))

    for im, cax, lab in ((im_p, cbar_lower, "lower: string identity"),
                         (im_v, cbar_upper, "upper: SapBERT clusters")):
        if cax is None:
            continue
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(lab, fontsize=6.6)
        cb.ax.tick_params(labelsize=6)

    panel_title(ax, "a")
    return im_p, im_v


# -----------------------------------------------------------------------------
def panel_b(ax):
    """Is the high model-model agreement just both models saying "absent"?

    The objection is legitimate: presence agreement is scored over every
    document-field slot, most of which are empty, so two sparse annotators
    would agree trivially often. Three quantities answer it without relying on
    kappa. Chance agreement would be HIGHER for the model pairs if shared
    sparsity were inflating them, and it is the lowest. The both-absent share
    measures the worry directly and is comparable across groups. Positive
    specific agreement discards the both-absent cell entirely, so it cannot be
    inflated by shared silence at all, and it widens the gap rather than
    closing it.

    Boxes rather than bars. The three groups have 35, 27 and 3 members, and a
    bar showing only the mean hides that the model-model claim rests on three
    pairs. Every pair is drawn as a point on top of its box, so the n is
    visible instead of asserted.

    The rows do not share a denominator, and the panel says so. Raw agreement,
    chance agreement and both-absent share are shares of all slots. Positive
    specific agreement is a share of only the slots at least one rater marked.
    Cohen's kappa is not a share at all: it is the fraction of the room above
    chance that was used, (raw - chance) / (1 - chance). Otherwise the obvious
    reading of kappa 0.43 sitting "below" chance agreement 0.56 is that
    agreement fell short of chance. It did not: raw agreement is 0.75 against a
    chance floor of 0.56, leaving 0.44 of headroom of which 0.19 was used.

    The p values are the exact rater-level permutation test from
    build_tables.py, model-model against human-human. They permute which
    RATERS are models, not which pairs are model pairs, because each rater sits
    in up to eleven pairs and a pair-level rank test would treat those as
    independent observations.
    """
    rows = load("agreement_decomposition.csv")
    by_group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for metric in ("kappa", "raw_agreement", "chance_agreement",
                       "both_absent_share", "positive_agreement"):
            by_group[r["group"]][metric].append(fnum(r[metric]))

    metrics = [
        ("raw_agreement", "raw agreement"),
        ("chance_agreement", "chance agreement"),
        ("both_absent_share", "both-absent share"),
        ("positive_agreement", "positive specific\nagreement"),
        ("kappa", "Cohen's kappa"),
    ]
    SPLIT = 3          # rows above this index are shares of all slots
    groups = [g for g in GROUP_ORDER if g in by_group]
    h = 0.78 / len(groups)
    base = np.arange(len(metrics))
    rng = np.random.default_rng(0)
    tests = stats()["tests"]

    for gi, g in enumerate(groups):
        off = (gi - (len(groups) - 1) / 2) * h
        color = GROUP_COLOR[g]
        vals = [np.asarray([v for v in by_group[g][k] if np.isfinite(v)])
                for k, _ in metrics]
        bp = ax.boxplot(vals, positions=base + off, widths=h * 0.80,
                        vert=False, patch_artist=True, manage_ticks=False,
                        showfliers=False, whis=(5, 95), zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.32, edgecolor=color, linewidth=0.9)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=color, linewidth=0.9)
        for med in bp["medians"]:
            med.set(color=color, linewidth=1.8)
        for y0, v in zip(base + off, vals):
            ax.scatter(v, y0 + rng.uniform(-h * 0.22, h * 0.22, size=len(v)),
                       s=4.5, color=color, alpha=0.75, linewidth=0, zorder=3)

    # model-model against human-human, per metric
    for yi, (key, _) in enumerate(metrics):
        res = tests.get("permutation", {}).get(key)
        if not res:
            continue
        pv = res["p_two_sided"]
        mark = "n.s." if pv > 0.05 else ("p < 0.005" if pv <= 1 / 220 + 1e-9
                                         else f"p = {pv:.3f}")
        ax.annotate(mark, xy=(1.0, yi), xytext=(-2, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.0,
                    color="0.30" if pv <= 0.05 else "0.55",
                    fontweight="bold" if pv <= 0.05 else "normal")

    ax.set_yticks(base)
    ax.set_yticklabels([lab for _, lab in metrics], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("value per annotator pair", fontsize=8.5)
    clean_axes(ax, grid_axis="x")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLOR[g], alpha=0.55,
                             edgecolor=GROUP_COLOR[g],
                             label=f"{g} (n={len(by_group[g]['kappa'])})")
               for g in groups]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=1,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.1)

    ax.axhline(SPLIT - 0.5, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=4)
    panel_title(ax, "b")


# -----------------------------------------------------------------------------
def panel_c(ax):
    """Agreement against how often a field is stated at all, human pairs."""
    ment = {r["label"]: (float(r["mention_rate"]) * 100, r["category"])
            for r in load("label_mention_rate.csv")}
    per = defaultdict(list)
    for r in load("label_agreement_presence.csv"):
        if r["pair_category"] in HUMAN_PAIRS and r["kappa"].strip():
            per[r["label"]].append(float(r["kappa"]))

    pts = [(ment[k][0], float(np.mean(v)), ment[k][1], k)
           for k, v in per.items() if k in ment]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    mean_y = float(np.mean(ys))
    sd_y = float(np.std(ys))
    ax.axhspan(mean_y - sd_y, mean_y + sd_y, color=GRID, alpha=0.75, zorder=0)
    ax.axhline(mean_y, color="0.55", lw=1.0, ls=(0, (4, 3)), zorder=1)

    for cat in CAT_ORDER:
        sel = [p for p in pts if p[2] == cat]
        if not sel:
            continue
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=46,
                   color=CAT_COLOR[cat], edgecolor="white", linewidth=0.8,
                   alpha=0.95, zorder=3, label=CAT_LABEL[cat])

    # name only the corners: the fields a reader would go looking for
    notable = (sorted(pts, key=lambda p: p[1])[:3]          # worst agreement
               + sorted(pts, key=lambda p: -p[1])[:3])      # best agreement
    seen, texts = set(), []
    for x, y, cat, label in notable:
        if label in seen:
            continue
        seen.add(label)
        texts.append(ax.text(x, y, label, fontsize=6.2, color=CAT_COLOR[cat],
                             fontweight="bold", zorder=4))
    adjust_text(texts, ax=ax, expand=(1.25, 1.5),
                arrowprops=dict(arrowstyle="-", color="#9aa6a2", lw=0.45))

    ax.set_xlabel("field stated in text (% of 27 papers)", fontsize=8.5)
    ax.set_ylabel("Cohen's kappa (presence)", fontsize=8.5)
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.04, 1)
    clean_axes(ax, grid_axis="both")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.3)

    panel_title(ax, "c")


# -----------------------------------------------------------------------------
def panel_d(ax):
    """Where both annotators tagged a field, did they write the same thing?"""
    rows = load("value_match_by_label.csv")
    rows.sort(key=lambda r: -(float(r["exact_pct"]) + float(r["semantic_pct"])))
    labels = [r["label"] for r in rows]
    exact = np.array([float(r["exact_pct"]) for r in rows])
    sem = np.array([float(r["semantic_pct"]) for r in rows])
    none = np.array([float(r["no_match_pct"]) for r in rows])
    x = np.arange(len(rows))

    ax.bar(x, exact, color=LIN_KIND["same term"], width=0.78, zorder=2, label="exact string")
    ax.bar(x, sem, bottom=exact, color=LIN_KIND["related (Lin 0.5-0.99)"], width=0.78, zorder=2,
           label="SapBERT only (cos ≥ 0.70)")
    ax.bar(x, none, bottom=exact + sem, color="#d9d9d9", width=0.78, zorder=2, label="residual mismatch")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=90)
    for tick, r in zip(ax.get_xticklabels(), rows):
        tick.set_color(CAT_COLOR.get(r["category"], "0.3"))
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("both-tagged instances (%)", fontsize=8.5)
    clean_axes(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.1)
    panel_title(ax, "d")


# -----------------------------------------------------------------------------
def panel_e(ax):
    """What forgiving wording buys, pooled per metadata category.

    Per label this is forty-two small shifts, which is a list and not a
    result. Pooled into the three categories it answers the question that
    matters: whether the wording problem is concentrated somewhere.
    """
    rows = [r for r in load("value_movement_by_category.csv")
            if r["pair_category"] in HUMAN_PAIRS]
    by_cat = defaultdict(lambda: {"s": [], "b": [], "n_labels": 0})
    for r in rows:
        d = by_cat[r["category"]]
        d["s"].append(float(r["string_kappa"]))
        d["b"].append(float(r["sapbert_kappa"]))
        d["n_labels"] = int(r["n_labels"])

    cats = [c for c in CAT_ORDER if c in by_cat]
    y = np.arange(len(cats))
    rng = np.random.default_rng(0)

    for yi, cat in zip(y, cats):
        d = by_cat[cat]
        color = CAT_COLOR[cat]
        s_m, b_m = float(np.mean(d["s"])), float(np.mean(d["b"]))
        # the individual annotator pairs behind each mean, so the spread is visible
        for arr, filled in ((d["s"], False), (d["b"], True)):
            jit = yi + rng.uniform(-0.13, 0.13, size=len(arr))
            ax.scatter(arr, jit, s=7, zorder=2, linewidth=0.5,
                       color=color if filled else "none",
                       edgecolor=color, alpha=0.28)
        ax.annotate("", xy=(b_m, yi), xytext=(s_m, yi), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                    shrinkA=0, shrinkB=0))
        ax.scatter([s_m], [yi], s=64, facecolor="white", edgecolor=color,
                   linewidth=1.8, zorder=4)
        ax.scatter([b_m], [yi], s=72, color=color, edgecolor="white",
                   linewidth=0.9, zorder=5)
        ax.text(b_m + 0.035, yi, f"{b_m - s_m:+.2f}", va="center", fontsize=7.2,
                fontweight="bold", color=color)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{CAT_LABEL[c]}\n({by_cat[c]['n_labels']} fields)" for c in cats],
                       fontsize=7.4)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("Cohen's kappa (value category), human pairs", fontsize=8.5)
    clean_axes(ax, grid_axis="x")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="0.35", markersize=7, label="string identity"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35",
                   markersize=7, label="SapBERT clusters"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    panel_title(ax, "e")


# -----------------------------------------------------------------------------
def main():
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], height_ratios=[1.0, 0.9],
                          hspace=0.28, wspace=0.55)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a,
            cbar_lower=ax_a.inset_axes([1.03, 0.05, 0.03, 0.38]),
            cbar_upper=ax_a.inset_axes([1.03, 0.55, 0.03, 0.38]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]))
    panel_d(fig.add_subplot(gs[1, 0:2]))
    panel_e(fig.add_subplot(gs[1, 2]))

    save_composite(fig, FIGDIR, "figure1")
    plt.close(fig)
    save_panels(plt, FIGDIR, "figure1", {
        "a": lambda ax: panel_a(ax, cbar_lower=ax.inset_axes([1.03, 0.05, 0.03, 0.38]),
                                cbar_upper=ax.inset_axes([1.03, 0.55, 0.03, 0.38])),
        "b": panel_b, "c": panel_c, "d": panel_d, "e": panel_e,
    }, {"a": (7.5, 7.5), "b": (6.5, 4.6), "c": (6, 4.6), "d": (11, 4.2), "e": (6, 3.6)})
    print(f"Saved: {FIGDIR / 'figure1.png'}")


if __name__ == "__main__":
    main()
