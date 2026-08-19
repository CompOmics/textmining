"""
Side-by-side comparison of prediction vs ground truth for a single PXD.

Usage:
    python -m benchmark.compare <prediction.json> <label.json>
    python -m benchmark.compare benchmark/results/qwen3.5/PXD001468.json benchmark/test_set/labels/PXD001468.json
"""

import json
import sys
from pathlib import Path

from benchmark.evaluate import extract_values, match_sample_groups, ALL_FIELDS

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _get_raw_and_normalized(data, field_name):
    """Extract (raw_value, normalized_value) pairs from a field."""
    items = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("value") is not None:
                raw = str(item["value"])
                norm = str(item.get("ontology_name") or raw)
                items.append((raw, norm))
    elif isinstance(data, dict):
        v = data.get("value")
        if v is not None:
            items.append((str(v), str(v)))
        m = data.get("method")
        if m is not None:
            norm_m = str(data.get("method_ontology_name") or m)
            items.append((str(m), norm_m))
    return items


def compare_pxd(pred_path: str, label_path: str):
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    label = json.loads(Path(label_path).read_text(encoding="utf-8"))

    pred_groups = pred.get("sample_groups", [])
    label_groups = label.get("sample_groups", [])

    pxd_id = Path(pred_path).stem
    print(f"\n{BOLD}{'=' * 70}")
    print(f"  {pxd_id}  —  {len(pred_groups)} predicted vs {len(label_groups)} ground truth groups")
    print(f"{'=' * 70}{RESET}\n")

    pairs = match_sample_groups(pred_groups, label_groups)

    for pred_group, label_group in pairs:
        pred_name = pred_group.get("name", "?") if pred_group else "—"
        label_name = label_group.get("name", "?") if label_group else "—"

        print(f"{BOLD}  Pred: {pred_name}")
        print(f"  GT:   {label_name}{RESET}")
        print(f"  {'-' * 66}")

        if not pred_group or not label_group:
            if not pred_group:
                print(f"  {RED}No matching prediction for this ground truth group{RESET}")
            else:
                print(f"  {RED}No matching ground truth for this prediction group{RESET}")
            print()
            continue

        for field_name in ALL_FIELDS:
            pred_data = pred_group.get(field_name, [])
            label_data = label_group.get(field_name, [])

            pred_items = _get_raw_and_normalized(pred_data, field_name)
            label_items = _get_raw_and_normalized(label_data, field_name)

            # Use normalized values for comparison
            pred_vals = {norm.lower(): (raw, norm) for raw, norm in pred_items}
            label_vals = {norm.lower(): (raw, norm) for raw, norm in label_items}

            matched = set(pred_vals.keys()) & set(label_vals.keys())
            pred_only = set(pred_vals.keys()) - set(label_vals.keys())
            label_only = set(label_vals.keys()) - set(pred_vals.keys())

            if not pred_items and not label_items:
                continue

            # Determine status
            if not pred_only and not label_only and matched:
                status = f"{GREEN}MATCH{RESET}"
            elif matched and (pred_only or label_only):
                status = f"{YELLOW}PARTIAL{RESET}"
            elif not matched and (pred_only or label_only):
                status = f"{RED}MISMATCH{RESET}"
            else:
                status = f"{DIM}EMPTY{RESET}"

            print(f"\n  {BOLD}{field_name:<22s}{RESET} {status}")

            for v in sorted(matched):
                raw_p, norm_p = pred_vals[v]
                raw_l, norm_l = label_vals[v]
                if raw_p.lower() == norm_p.lower():
                    print(f"    {GREEN}  = {norm_p}{RESET}")
                else:
                    print(f"    {GREEN}  = {norm_p}  {DIM}(pred raw: {raw_p}){RESET}")

            for v in sorted(pred_only):
                raw_p, norm_p = pred_vals[v]
                if raw_p.lower() == norm_p.lower():
                    print(f"    {RED}  + pred: {norm_p}{RESET}")
                else:
                    print(f"    {RED}  + pred: {norm_p}  {DIM}(raw: {raw_p}){RESET}")

            for v in sorted(label_only):
                raw_l, norm_l = label_vals[v]
                if raw_l.lower() == norm_l.lower():
                    print(f"    {RED}  - gt:   {norm_l}{RESET}")
                else:
                    print(f"    {RED}  - gt:   {norm_l}  {DIM}(raw: {raw_l}){RESET}")

        print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchmark.compare <PXD_ID> [--results <dir>]")
        print("       python -m benchmark.compare PXD001468")
        print("       python -m benchmark.compare PXD001468 --results benchmark/results/qwen3.5-35B-A3B-instruct")
        sys.exit(1)

    pxd_id = sys.argv[1]
    results_dir = "benchmark/results/qwen3.5-35B-A3B-instruct"
    if "--results" in sys.argv:
        results_dir = sys.argv[sys.argv.index("--results") + 1]

    pred_path = Path(results_dir) / f"{pxd_id}.json"
    label_path = Path("benchmark/test_set/labels") / f"{pxd_id}.json"

    if not pred_path.exists():
        print(f"Prediction not found: {pred_path}")
        sys.exit(1)
    if not label_path.exists():
        print(f"Label not found: {label_path}")
        sys.exit(1)

    compare_pxd(str(pred_path), str(label_path))
