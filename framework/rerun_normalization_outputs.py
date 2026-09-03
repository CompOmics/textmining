#!/usr/bin/env python3
"""Re-normalize existing raw agent outputs into a separate output tree."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from agents.normalization_agent import NormalizationAgent
from core.field_mappings import FIELD_TO_ENTITY_TYPE
from normalization.normalizer import TermNormalizer


AGENTS = ("BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent")


def load_json_files(directory: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        with path.open() as handle:
            results[path.name] = json.load(handle)
    return results


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def load_required_ontologies(agent: NormalizationAgent) -> None:
    """Load only ontologies used by the extraction schema plus Unimod."""
    normalizer = TermNormalizer(agent.config)
    ontology_ids = {"unimod"}
    for entity_type in set(FIELD_TO_ENTITY_TYPE.values()):
        ontology_id = agent.config.entity_ontology_map.get(entity_type)
        if ontology_id:
            ontology_ids.add(ontology_id)

    for ontology_id in sorted(ontology_ids):
        ontology_path = agent.config.get_ontology_path(ontology_id)
        if ontology_path and ontology_path.exists():
            normalizer.load_ontology(ontology_id, str(ontology_path))

    agent.normalizer = normalizer
    agent._loaded = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=int, choices=range(8))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_shard = source_root / f"shard_{args.shard}"
    output_shard = output_root / f"shard_{args.shard}"

    counts = {
        agent_name: len(list((source_shard / agent_name).glob("*.json")))
        for agent_name in AGENTS
    }
    if not source_shard.is_dir() or any(count == 0 for count in counts.values()):
        raise SystemExit(
            f"Incomplete source shard {source_shard}: {counts}"
        )
    if len(set(counts.values())) != 1:
        raise SystemExit(f"Agent file-count mismatch in {source_shard}: {counts}")

    if args.dry_run:
        print(json.dumps({
            "source_shard": str(source_shard),
            "output_shard": str(output_shard),
            "input_counts": counts,
        }, indent=2))
        return

    if output_shard.exists():
        raise SystemExit(
            f"Refusing to overwrite existing destination shard: {output_shard}"
        )

    started = time.time()
    normalization_agent = NormalizationAgent(auto_load=False)
    load_required_ontologies(normalization_agent)
    loaded_at = time.time()

    output_counts: dict[str, int] = {}
    agent_seconds: dict[str, float] = {}
    for agent_name in AGENTS:
        agent_started = time.time()
        raw = load_json_files(source_shard / agent_name)
        normalized = normalization_agent.normalize_batch(raw)
        destination = output_shard / "NormalizedAgent" / agent_name
        for filename, payload in normalized.items():
            write_json_atomic(destination / filename, payload)
        output_counts[agent_name] = len(normalized)
        agent_seconds[agent_name] = round(time.time() - agent_started, 3)

    if output_counts != counts:
        raise SystemExit(
            f"Output count mismatch: input={counts}, output={output_counts}"
        )

    finished = time.time()
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "shard": args.shard,
        "input_counts": counts,
        "output_counts": output_counts,
        "ontology_load_seconds": round(loaded_at - started, 3),
        "agent_seconds": agent_seconds,
        "total_seconds": round(finished - started, 3),
        "source_outputs_modified": False,
        "llm_inference_run": False,
    }
    write_json_atomic(output_shard / "normalization_summary.json", summary)
    (output_shard / "SUCCESS").touch()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
