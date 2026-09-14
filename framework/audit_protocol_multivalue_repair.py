#!/usr/bin/env python3
"""Run final completeness and multi-value audits after targeted repair."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HAMLET = Path(__file__).resolve().parents[2]
INPUT_ROOT = HAMLET / "textmining/production_inputs/abstract_methods_pride_protocols_1737_8shards"
OUTPUT_ROOT = HAMLET / "textmining/framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737"
AUDITOR = HAMLET / "textmining/audit_multivalue_outputs.py"
AGENTS = ("BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent")


def main() -> None:
    totals = {agent: {"raw": 0, "normalized": 0} for agent in AGENTS}
    shard_reports = []
    for shard in range(8):
        input_dir = INPUT_ROOT / f"shard_{shard}"
        output_dir = OUTPUT_ROOT / f"shard_{shard}"
        expected = len(list(input_dir.glob("PXD*.txt")))
        counts = {}
        for agent in AGENTS:
            raw = len(list((output_dir / agent).glob("PXD*.json")))
            normalized = len(list((output_dir / "NormalizedAgent" / agent).glob("PXD*.json")))
            counts[agent] = {"raw": raw, "normalized": normalized}
            totals[agent]["raw"] += raw
            totals[agent]["normalized"] += normalized
            if raw != expected or normalized != expected:
                raise RuntimeError(
                    f"shard_{shard} {agent} incomplete: expected={expected}, raw={raw}, normalized={normalized}"
                )
        report_path = output_dir / "multivalue_audit.json"
        subprocess.run(
            [sys.executable, str(AUDITOR), "--input", str(input_dir), "--output", str(output_dir),
             "--report", str(report_path)],
            check=True,
        )
        (output_dir / "SUCCESS").touch()
        shard_reports.append({"shard": shard, "expected": expected, "counts": counts, "audit": str(report_path)})

    if any(values[mode] != 1737 for values in totals.values() for mode in ("raw", "normalized")):
        raise RuntimeError(f"unexpected global totals: {totals}")
    report = {
        "status": "complete",
        "papers": 1737,
        "agent_totals": totals,
        "shards": shard_reports,
    }
    status_dir = OUTPUT_ROOT / "repair_status_protocols_multivalue"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "final_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT_ROOT / "SUCCESS").touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

