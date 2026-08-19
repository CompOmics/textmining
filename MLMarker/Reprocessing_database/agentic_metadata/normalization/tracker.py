"""Tracks all normalization mappings for quality reporting."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class NormalizationRecord:
    field: str
    raw_value: str
    normalized_to: str
    similarity: float
    status: str  # "mapped", "unmapped", "exact_match", "post_mapped"


class NormalizationTracker:
    """Collects normalization records and writes a summary report.

    Thread-safe: list.append() is atomic in CPython (GIL).
    """

    def __init__(self):
        self.records: List[NormalizationRecord] = []

    def record(self, field: str, raw_value: str, normalized_to: str,
               similarity: float, status: str):
        self.records.append(NormalizationRecord(
            field=field, raw_value=raw_value,
            normalized_to=normalized_to, similarity=similarity,
            status=status,
        ))

    def write_report(self, output_path: Path):
        """Aggregate records and write normalization_map.tsv.

        Columns: field, raw_value, normalized_to, similarity, count, status

        Rows are grouped by (field, raw_value, normalized_to, status) with counts.
        Sorted by field, then count descending within each field.
        """
        # Group by (field, raw_value, normalized_to, status), count occurrences
        counts: Counter = Counter()
        sim_lookup: Dict[Tuple[str, str, str], float] = {}

        for r in self.records:
            key = (r.field, r.raw_value, r.normalized_to, r.status)
            counts[key] += 1
            sim_lookup[(r.field, r.raw_value, r.normalized_to)] = r.similarity

        rows = []
        for (fld, raw, norm, status), count in counts.items():
            sim = sim_lookup[(fld, raw, norm)]
            rows.append({
                "field": fld,
                "raw_value": raw,
                "normalized_to": norm,
                "similarity": f"{sim:.3f}",
                "count": count,
                "status": status,
            })

        # Sort: by field, then mapped before unmapped, then count descending
        status_order = {"mapped": 0, "exact_match": 0, "post_mapped": 0, "unmapped": 1}
        rows.sort(key=lambda r: (r["field"], status_order.get(r["status"], 2), -r["count"]))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["field", "raw_value", "normalized_to", "similarity", "count", "status"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
