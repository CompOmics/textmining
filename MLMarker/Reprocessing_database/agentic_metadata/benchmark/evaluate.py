"""Evaluation module for scoring predictions against ground truth labels.

Scoring approach (per field value):
  1. Case-insensitive string match -> score = 1
  2. If no string match, SapBERT cosine similarity -> score = 1 if >= threshold, else 0

Aggregation:
  - Per field: precision, recall, F1 over all PXDs
  - Overall: macro-averaged F1 across fields

Directory structure expected:
  labels_dir/       -> PXD000561.json, PXD001468.json, ...
  predictions_dir/  -> PXD000561.json, PXD001468.json, ...
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .sapBERT import SapBERTEmbedder


# Global singleton
_embedder: Optional[SapBERTEmbedder] = None


def get_embedder() -> SapBERTEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SapBERTEmbedder()
    return _embedder


# 1. Sample Group Matching
def match_sample_groups(
    pred_groups: list[dict], gt_groups: list[dict]
) -> list[tuple[dict, dict]]:
    """Match predicted sample groups to ground truth groups using SapBERT
    similarity on group names. Greedy best-match pairing."""
    if len(pred_groups) == 1 and len(gt_groups) == 1:
        return [(pred_groups[0], gt_groups[0])]

    if not pred_groups or not gt_groups:
        return []

    pred_names = [g.get("name", "") for g in pred_groups]
    gt_names = [g.get("name", "") for g in gt_groups]

    sim_matrix = get_embedder().cosine_similarity_matrix(pred_names, gt_names)

    used_pred = set()
    used_gt = set()
    pairs = []

    flat = []
    for i in range(len(pred_groups)):
        for j in range(len(gt_groups)):
            flat.append((sim_matrix[i, j], i, j))
    flat.sort(reverse=True)

    for sim, i, j in flat:
        if i in used_pred or j in used_gt:
            continue
        pairs.append((pred_groups[i], gt_groups[j]))
        used_pred.add(i)
        used_gt.add(j)

    return pairs


# 2. Value Matching
def string_match(pred: str, gt: str) -> bool:
    return pred.strip().lower() == gt.strip().lower()


def score_value_pair(
    pred: str,
    gt: str,
    sim_matrix: Optional[np.ndarray] = None,
    pred_idx: Optional[int] = None,
    gt_idx: Optional[int] = None,
    sapbert_threshold: float = 0.75,
) -> tuple[float, str]:
    if string_match(pred, gt):
        return 1.0, "string"

    if sim_matrix is not None and pred_idx is not None and gt_idx is not None:
        sim = float(sim_matrix[pred_idx, gt_idx])
    else:
        vecs = get_embedder().embed([pred, gt])
        sim = float(np.dot(vecs[0], vecs[1]))

    if sim >= sapbert_threshold:
        return 1.0, "sapbert"

    return 0.0, "none"


# 3. Field-Level Scoring
@dataclass
class FieldScore:
    field_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_predicted: int = 0
    n_ground_truth: int = 0
    n_true_positive: int = 0
    matched_pairs: list = field(default_factory=list)


def extract_values(data) -> list[str]:
    """Extract values, preferring ontology_name over raw value when available."""
    if isinstance(data, list):
        return [
            str(item.get("ontology_name") or item["value"])
            for item in data
            if isinstance(item, dict) and item.get("value") is not None
        ]
    elif isinstance(data, dict):
        vals = []
        v = data.get("value")
        if v is not None:
            vals.append(str(v))
        m = data.get("method_ontology_name") or data.get("method")
        if m is not None:
            vals.append(str(m))
        return vals
    return []


def score_field(
    pred_values: list[str],
    gt_values: list[str],
    field_name: str,
    sapbert_threshold: float = 0.75,
) -> FieldScore:
    result = FieldScore(
        field_name=field_name,
        n_predicted=len(pred_values),
        n_ground_truth=len(gt_values),
    )

    if not pred_values and not gt_values:
        result.precision = 1.0
        result.recall = 1.0
        result.f1 = 1.0
        return result

    if not pred_values or not gt_values:
        return result

    sim_matrix = get_embedder().cosine_similarity_matrix(pred_values, gt_values)

    n_pred, n_gt = len(pred_values), len(gt_values)
    score_matrix = np.zeros((n_pred, n_gt))
    method_matrix = [["none"] * n_gt for _ in range(n_pred)]

    for i, pv in enumerate(pred_values):
        for j, gv in enumerate(gt_values):
            score, method = score_value_pair(
                pv, gv,
                sim_matrix=sim_matrix, pred_idx=i, gt_idx=j,
                sapbert_threshold=sapbert_threshold,
            )
            score_matrix[i, j] = score
            method_matrix[i][j] = method

    flat = []
    for i in range(n_pred):
        for j in range(n_gt):
            flat.append((score_matrix[i, j], i, j))
    flat.sort(reverse=True)

    used_pred = set()
    used_gt = set()

    for score, i, j in flat:
        if i in used_pred or j in used_gt:
            continue
        if score > 0:
            result.matched_pairs.append(
                (pred_values[i], gt_values[j], score, method_matrix[i][j])
            )
            used_pred.add(i)
            used_gt.add(j)

    tp = len(result.matched_pairs)
    result.n_true_positive = tp
    result.precision = tp / n_pred if n_pred > 0 else 0.0
    result.recall = tp / n_gt if n_gt > 0 else 0.0
    if result.precision + result.recall > 0:
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)

    return result


# 4. PXD-Level Scoring
ALL_FIELDS = [
    "organism", "tissue", "disease", "cell_part", "cell_line",
    "instrument", "fragmentation", "enzymes", "modifications",
    "collision_energy", "gradient_time_min", "lc_column", "acquisition",
    "labeling", "fractionation", "enrichment", "ionization",
    "treatment_type", "treatment_name", "treatment_class",
]


@dataclass
class PXDScore:
    pxd_id: str
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    n_groups_predicted: int = 0
    n_groups_ground_truth: int = 0
    n_groups_matched: int = 0


def score_pxd(
    prediction: dict,
    ground_truth: dict,
    pxd_id: str = "unknown",
    sapbert_threshold: float = 0.75,
) -> PXDScore:
    result = PXDScore(pxd_id=pxd_id)

    pred_groups = prediction.get("sample_groups", [])
    gt_groups = ground_truth.get("sample_groups", [])
    result.n_groups_predicted = len(pred_groups)
    result.n_groups_ground_truth = len(gt_groups)

    pairs = match_sample_groups(pred_groups, gt_groups)
    result.n_groups_matched = len(pairs)

    field_preds: dict[str, list[str]] = {f: [] for f in ALL_FIELDS}
    field_gts: dict[str, list[str]] = {f: [] for f in ALL_FIELDS}

    for pred_group, gt_group in pairs:
        for f in ALL_FIELDS:
            field_preds[f].extend(extract_values(pred_group.get(f, [] if f not in ("fractionation", "enrichment") else {})))
            field_gts[f].extend(extract_values(gt_group.get(f, [] if f not in ("fractionation", "enrichment") else {})))

    for f in ALL_FIELDS:
        pred_dedup = list({v.lower(): v for v in field_preds[f]}.values())
        gt_dedup = list({v.lower(): v for v in field_gts[f]}.values())
        result.field_scores[f] = score_field(
            pred_dedup, gt_dedup, f, sapbert_threshold=sapbert_threshold
        )

    return result


# 5. Aggregate Scoring
@dataclass
class BenchmarkResult:
    pxd_scores: list[PXDScore] = field(default_factory=list)
    field_summary: dict = field(default_factory=dict)
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0


def aggregate_scores(pxd_scores: list[PXDScore]) -> BenchmarkResult:
    result = BenchmarkResult(pxd_scores=pxd_scores)

    field_precisions: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    field_recalls: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    field_f1s: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}

    for pxd in pxd_scores:
        for f in ALL_FIELDS:
            if f in pxd.field_scores:
                fs = pxd.field_scores[f]
                if fs.n_ground_truth > 0 or fs.n_predicted > 0:
                    field_precisions[f].append(fs.precision)
                    field_recalls[f].append(fs.recall)
                    field_f1s[f].append(fs.f1)

    for f in ALL_FIELDS:
        if field_f1s[f]:
            result.field_summary[f] = {
                "precision": float(np.mean(field_precisions[f])),
                "recall": float(np.mean(field_recalls[f])),
                "f1": float(np.mean(field_f1s[f])),
                "n_pxds": len(field_f1s[f]),
            }

    f1s = [v["f1"] for v in result.field_summary.values() if v["n_pxds"] > 0]
    precs = [v["precision"] for v in result.field_summary.values() if v["n_pxds"] > 0]
    recs = [v["recall"] for v in result.field_summary.values() if v["n_pxds"] > 0]

    result.macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    result.macro_precision = float(np.mean(precs)) if precs else 0.0
    result.macro_recall = float(np.mean(recs)) if recs else 0.0

    return result


# 6. Reporting
def evaluate_benchmark(
    labels_dir: Path,
    predictions_dir: Path,
    sapbert_threshold: float = 0.75,
) -> BenchmarkResult:
    """Evaluate all PXDs: match label files to prediction files by PXD name."""
    pxd_scores = []

    for gt_file in sorted(labels_dir.glob("PXD*.json")):
        pxd_id = gt_file.stem
        pred_file = predictions_dir / f"{pxd_id}.json"

        if not pred_file.exists():
            continue

        ground_truth = json.loads(gt_file.read_text(encoding="utf-8"))
        prediction = json.loads(pred_file.read_text(encoding="utf-8"))

        pxd_scores.append(score_pxd(prediction, ground_truth, pxd_id=pxd_id, sapbert_threshold=sapbert_threshold))

    return aggregate_scores(pxd_scores)


def print_results(result: BenchmarkResult):
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS - {len(result.pxd_scores)} PXDs evaluated")
    print(f"{'=' * 70}")

    print(f"\n{'Field':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'#PXDs':>6}")
    print("-" * 58)

    for f in ALL_FIELDS:
        if f in result.field_summary:
            s = result.field_summary[f]
            print(f"{f:<25} {s['precision']:>8.3f} {s['recall']:>8.3f} {s['f1']:>8.3f} {s['n_pxds']:>6}")

    print("-" * 58)
    print(f"{'MACRO AVERAGE':<25} {result.macro_precision:>8.3f} {result.macro_recall:>8.3f} {result.macro_f1:>8.3f}")

    print(f"\n{'PXD':<15} {'Groups GT/Pred':>15} {'Matched':>8} {'Avg F1':>8}")
    print("-" * 48)
    for pxd in result.pxd_scores:
        f1s = [
            fs.f1 for fs in pxd.field_scores.values()
            if fs.n_ground_truth > 0 or fs.n_predicted > 0
        ]
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        print(
            f"{pxd.pxd_id:<15} {pxd.n_groups_ground_truth:>6}/{pxd.n_groups_predicted:<6} "
            f"{pxd.n_groups_matched:>8} {avg_f1:>8.3f}"
        )


def save_results(result: BenchmarkResult, output_path: Path):
    data = {
        "macro_f1": result.macro_f1,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "field_summary": result.field_summary,
        "pxd_details": [],
    }

    for pxd in result.pxd_scores:
        pxd_data = {
            "pxd_id": pxd.pxd_id,
            "n_groups_predicted": pxd.n_groups_predicted,
            "n_groups_ground_truth": pxd.n_groups_ground_truth,
            "n_groups_matched": pxd.n_groups_matched,
            "fields": {},
        }
        for f, fs in pxd.field_scores.items():
            pxd_data["fields"][f] = {
                "precision": fs.precision,
                "recall": fs.recall,
                "f1": fs.f1,
                "n_predicted": fs.n_predicted,
                "n_ground_truth": fs.n_ground_truth,
                "n_true_positive": fs.n_true_positive,
                "matched_pairs": [
                    {"pred": p, "gt": g, "score": s, "method": m}
                    for p, g, s, m in fs.matched_pairs
                ],
            }
        data["pxd_details"].append(pxd_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
