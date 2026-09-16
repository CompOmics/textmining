"""One palette and one style for every manuscript figure.

Import from any figure script:
    sys.path.insert(0, str(REPO / "final_analyses")); from figure_style import *

Rules: agents (biological / technical / experimental design) always use
AGENT; models always MODEL; input sources always SOURCE; matcher tiers TIER;
outcome bars OUTCOME. Muted tones throughout; no saturated primaries.
"""
from __future__ import annotations

from pathlib import Path

# ---- palettes ----------------------------------------------------------------
AGENT = {"BiologicalAgent": "#4f81b9", "TechnicalAgent": "#6aa56a", "ExperimentalDesignAgent": "#d99457"}
AGENT_LABEL = {"BiologicalAgent": "Biological", "TechnicalAgent": "Technical", "ExperimentalDesignAgent": "Experimental design"}
AGENTS = list(AGENT)

# models: purple / teal / brown, so they never read as the agents' blue / green / orange
MODEL = {"Qwen3.8-27B": "#8f7fbf", "Gemma 4 31B": "#5fa8a0", "GLM-4.7": "#b8866b"}
MODELS = list(MODEL)

# six-model benchmark (single run each); distinct from MODEL (staircase) and AGENT
SIX_MODEL = {"Llama-4 Scout": "#7a9cc6", "GPT-5.4": "#d9a679", "Claude Opus 4.5": "#8fb48f",
             "Gemini 3.5 Flash": "#cf8f8f", "Gemma 4 31B": "#8f7fbf", "Qwen 3.7 Max": "#b8866b"}
SOURCE = {"manuscript": "#7a9cc6", "pride": "#d9a679", "combined": "#8fb48f"}
SOURCE_LABEL = {"manuscript": "abstract + M&M", "pride": "PRIDE descriptor", "combined": "abstract + M&M + PRIDE"}

COHORT_LABEL = {"abstract_only": "closed-access manuscript", "with_methods": "open-access manuscript"}

# matcher tiers, strict to lenient, then failures
TIER = {"exact / normalised": "#1f6b45", "ontology / hierarchy": "#6aa56a", "SapBERT tier (cosine ≥ 0.70)": "#a9d3ae",
        "below tier, cosine ≥ 0.50": "#e6d9a0", "rejected": "#cf6f5f", "missing": "#c8c8c8"}
OUTCOME = {"accepted": "#8fb48f", "wrong": "#cf6f5f", "missing": "#c8c8c8"}
LIN_KIND = {"same term": "#1f6b45", "related (Lin 0.5-0.99)": "#7a9cc6", "distant (Lin 0.01-0.5)": "#d99457",
            "unrelated (Lin <= 0.01)": "#cf6f5f", "missing": "#c8c8c8"}
AGREEMENT = {"equivalent_value_sets": ("equivalent", "#1f6b45"), "hierarchically_related_value_sets": ("hierarchy-related", "#6aa56a"),
             "semantic_candidate_value_sets": ("SapBERT candidate", "#a9d3ae"), "partial_overlap": ("partial overlap", "#e6d9a0"),
             "no_accepted_overlap_review": ("no accepted overlap", "#cf6f5f")}
NEUTRAL = "#9a9a9a"
HEAT_CMAP = "Blues"


# ---- style and saving --------------------------------------------------------
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                         "grid.linestyle": "--", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
                         "legend.handlelength": 1.4, "svg.fonttype": "none"})
    return plt


def panel_title(ax, letter, text=""):
    ax.set_title(f"{letter}  {text}" if text else letter, loc="left")


def legend_outside(ax, where="below", ncol=3, **kw):
    """Legend that never overlaps the axes: below the x-label or right of the axes."""
    if where == "below":
        return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, **kw)
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, **kw)


def metric_group_legend(ax, groups: dict, metrics=(("F1", "-"), ("Lin IC", "--")), **kw):
    """Legend in two parts: the metric as grey line styles, then the colour
    key (model or agent). Placed to the right of the axes."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=NEUTRAL, ls=ls, lw=1.8, label=lab) for lab, ls in metrics]
    handles.append(Line2D([], [], color="none", label=""))
    handles += [Line2D([], [], color=col, ls="-", lw=3, label=lab) for lab, col in groups.items()]
    return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)


def save_composite(fig, out_dir: Path, name: str):
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=220, bbox_inches="tight")


def save_panels(plt, out_dir: Path, name: str, panel_fns: dict, sizes: dict | None = None):
    """Every panel drawn again on its own figure and saved as
    <out_dir>/panels/<name>_<letter>.{png,svg}; stale panels of that name removed."""
    sizes = sizes or {}
    pdir = out_dir / "panels"; pdir.mkdir(exist_ok=True)
    for f in pdir.glob(f"{name}_*"):
        f.unlink()
    for letter, fn in panel_fns.items():
        fig, ax = plt.subplots(figsize=sizes.get(letter, (5.5, 3.6)))
        fn(ax)
        for ext in ("png", "svg"):
            fig.savefig(pdir / f"{name}_{letter}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
