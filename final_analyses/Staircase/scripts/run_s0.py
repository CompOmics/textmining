#!/usr/bin/env python3
"""Run the frozen S0 BRAT prompt and preserve all executable artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import yaml

STAIRCASE = Path(__file__).resolve().parents[1]
FRAMEWORK = STAIRCASE.parents[1] / "framework"
sys.path.insert(0, str(STAIRCASE / "scripts"))
from brat_to_fields import convert


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional output-token safety cap; omitted for historical behaviour.",
    )
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = args.output / "runtime"
    runtime.mkdir()
    started = time.time()

    manuscripts = sorted(args.input.glob("PXD*/manuscript.txt"))
    if not manuscripts:
        raise SystemExit("no manuscripts")
    records = [{"id": p.parent.name, "text": p.read_text(encoding="utf-8", errors="replace")} for p in manuscripts]
    cfg = yaml.safe_load(args.config.read_text())
    llm = cfg["llm"]
    if args.base_url:
        llm["base_url"] = args.base_url
    os.environ["OPENAI_API_KEY"] = llm.get("api_key", "local-vllm")
    os.environ["OPENAI_BASE_URL"] = llm["base_url"]

    pipeline = yaml.safe_load(args.pipeline.read_text())
    pipeline["default_model"] = "openai/" + llm["model"]
    pipeline["datasets"]["manuscripts"]["path"] = str(runtime / "input.json")
    pipeline["pipeline"]["output"]["path"] = str(runtime / "docetl_output.json")
    for op in pipeline["operations"]:
        op["bypass_cache"] = True
        op["timeout"] = args.timeout
        op["max_retries_per_timeout"] = 0
        op.setdefault("litellm_completion_kwargs", {})["temperature"] = 0
        if args.max_tokens is not None:
            op["litellm_completion_kwargs"]["max_tokens"] = args.max_tokens
    (runtime / "input.json").write_text(json.dumps(records, indent=2))
    (runtime / "executed_pipeline.yaml").write_text(yaml.safe_dump(pipeline, sort_keys=False, width=1000))

    from docetl.runner import DSLRunner
    max_threads = cfg.get("concurrency", {}).get("max_workers", 1)
    DSLRunner.from_yaml(
        str(runtime / "executed_pipeline.yaml"), max_threads=max_threads
    ).load_run_save()
    results = json.loads((runtime / "docetl_output.json").read_text())
    if len(results) != len(records):
        raise SystemExit(f"incomplete S0: expected {len(records)}, got {len(results)}")

    text_by_id = {r["id"]: r["text"] for r in records}
    ann_dir, field_dir = args.output / "ann_files", args.output / "field_adapter"
    ann_dir.mkdir(); field_dir.mkdir()
    invalid_total = 0
    span_status_totals = {"exact": 0, "unique_realign": 0, "ambiguous_surface": 0, "surface_absent": 0}
    for result in results:
        pxd = result["id"]
        ann = result.get("ann_output")
        if not isinstance(ann, str):
            raise SystemExit(f"{pxd}: ann_output is not a string")
        if ann.strip() == "Not found":
            raise SystemExit(
                f"{pxd}: DocETL inserted the 'Not found' placeholder; "
                "the model tool call omitted ann_output"
            )
        (ann_dir / f"{pxd}.ann").write_text(ann + ("\n" if ann and not ann.endswith("\n") else ""))
        adapted = convert(text_by_id[pxd], ann)
        invalid_total += adapted["invalid_count"]
        for status, count in adapted["span_status_counts"].items():
            span_status_totals[status] += count
        (field_dir / f"{pxd}.json").write_text(json.dumps(adapted, indent=2) + "\n")

    manifest = {
        "arm": "S0", "model": llm["model"], "model_tag": args.model_tag,
        "temperature": 0, "cache_bypassed": True, "thinking_disabled_server_side": True,
        "max_tokens": args.max_tokens,
        "input_count": len(records), "input_manifest_sha256": sha(args.input / "manifest.json"),
        "pipeline_sha256": sha(args.pipeline), "config_sha256": sha(args.config),
        "python": platform.python_version(), "started_epoch": started,
        "finished_epoch": time.time(), "invalid_brat_spans": invalid_total,
        "span_status_counts": span_status_totals,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
