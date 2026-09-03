#!/usr/bin/env python3
"""Rebuild S0 derived adapters from preserved inference output without rerunning LLMs."""

import argparse
import json
from pathlib import Path

from brat_to_fields import convert


parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
s0 = args.run / "S0"
records = json.loads((s0 / "runtime/input.json").read_text())
results = json.loads((s0 / "runtime/docetl_output.json").read_text())
texts = {record["id"]: record["text"] for record in records}
out = s0 / "field_adapter"; out.mkdir(exist_ok=True)
totals = {"exact": 0, "unique_realign": 0, "ambiguous_surface": 0, "surface_absent": 0}
invalid = 0
for result in results:
    adapted = convert(texts[result["id"]], result.get("ann_output", ""))
    invalid += adapted["invalid_count"]
    for key, value in adapted["span_status_counts"].items(): totals[key] += value
    (out / f"{result['id']}.json").write_text(json.dumps(adapted, indent=2) + "\n")
manifest_path = s0 / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["invalid_brat_spans"] = invalid
manifest["span_status_counts"] = totals
manifest["adapter_rebuilt_from_preserved_raw"] = True
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"run": str(args.run), "invalid_brat_spans": invalid, "span_status_counts": totals}))
