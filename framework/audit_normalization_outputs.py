#!/usr/bin/env python3
"""Audit normalized extraction outputs without modifying source annotations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agents.normalization_agent import NormalizationAgent
from core.field_mappings import FIELD_TO_ENTITY_TYPE
from normalization.config import NormalizationConfig
from normalization.normalizer import NormalizationResult, TermNormalizer
from normalization.ontology import OntologyLoader


def entries(value: Any) -> Iterable[Tuple[Optional[int], Dict[str, Any]]]:
    if isinstance(value, dict):
        yield None, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                yield index, item


def normal_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def is_normal_state(field: str, value: Any, ontology_id: str) -> bool:
    return (
        field in {"disease", "disease_state"}
        and ontology_id == "PATO:0000461"
        and normal_key(value) in {"normal", "healthy", "control"}
    )


def base_row(
    path: Path,
    root: Path,
    field: str,
    index: Optional[int],
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    parts = path.parts
    agent = parts[parts.index("NormalizedAgent") + 1]
    return {
        "pxd": path.name.split("_")[0],
        "relative_file": str(path.relative_to(root)),
        "agent": agent,
        "field": field,
        "value_index": "" if index is None else index,
        "value": entry.get("value"),
        "evidence": entry.get("evidence"),
        "current_ontology_id": entry.get("ontology_id"),
        "current_ontology_name": entry.get("ontology_name"),
        "current_similarity": entry.get("similarity"),
        "proposed_ontology_id": None,
        "proposed_ontology_name": None,
        "proposed_method": None,
        "candidate_ids": [],
        "issue_type": None,
        "recommended_action": None,
    }


def proposed_row(
    row: Dict[str, Any], result: NormalizationResult
) -> Dict[str, Any]:
    updated = dict(row)
    updated.update({
        "proposed_ontology_id": result.ontology_id,
        "proposed_ontology_name": result.ontology_name,
        "proposed_method": result.normalization_method,
        "candidate_ids": [candidate[0] for candidate in result.candidates],
        "issue_type": "deterministic_mapping_correction",
        "recommended_action": "replace_mapping_only",
    })
    return updated


def review_row(
    row: Dict[str, Any],
    issue_type: str,
    action: str,
    candidate_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    updated = dict(row)
    updated.update({
        "candidate_ids": candidate_ids or [],
        "issue_type": issue_type,
        "recommended_action": action,
    })
    return updated


def main() -> None:
    logging.getLogger("normalization.normalizer").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ontologies",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(input_root.glob("shard_*/NormalizedAgent/*/*.json"))
    if not files:
        raise SystemExit(f"No normalized JSON files found under {input_root}")

    config = NormalizationConfig(
        ontology_dir=str(args.ontology_dir.resolve()),
        cache_dir=str(Path(__file__).resolve().parent / "ontology_cache"),
    )
    used_ontologies = {"unimod"}
    for path in files:
        record = json.loads(path.read_text())
        for field in record:
            entity_type = FIELD_TO_ENTITY_TYPE.get(field)
            ontology_id = config.entity_ontology_map.get(entity_type or "")
            if ontology_id:
                used_ontologies.add(ontology_id)

    normalizer = TermNormalizer(config)
    loader = OntologyLoader()
    for ontology_id in sorted(used_ontologies):
        ontology_path = config.get_ontology_path(ontology_id)
        if ontology_path and ontology_path.exists():
            normalizer.graphs[ontology_id] = loader.load(str(ontology_path))

    findings: List[Dict[str, Any]] = []
    scanned_values = 0
    records_by_path: Dict[Path, Dict[str, Any]] = {}

    for path in files:
        record = json.loads(path.read_text())
        records_by_path[path] = record
        for field, value in record.items():
            entity_type = FIELD_TO_ENTITY_TYPE.get(field)
            ontology_id = config.entity_ontology_map.get(entity_type or "")
            graph = normalizer.graphs.get(ontology_id or "")
            if not entity_type or not ontology_id or graph is None:
                continue
            for index, entry in entries(value):
                if "value" not in entry:
                    continue
                scanned_values += 1
                row = base_row(path, input_root, field, index, entry)
                current_id = entry.get("ontology_id")
                term = str(entry.get("value", ""))

                result = normalizer._ptm_unimod_result(term, entity_type)
                if result is None:
                    result = normalizer._exact_lexical_result(
                        graph, term, ontology_id, entity_type
                    )
                if result is not None:
                    if result.is_normalized and result.ontology_id != current_id:
                        findings.append(proposed_row(row, result))
                    elif not result.is_normalized:
                        findings.append(review_row(
                            row,
                            result.normalization_method
                            or "ambiguous_ontology_match",
                            "manual_or_contextual_disambiguation",
                            [candidate[0] for candidate in result.candidates],
                        ))

                # Validate against the field contract rather than only the
                # graph namespace. PTM fields intentionally allow both
                # PSI-MOD and Unimod identifiers.
                prefixes = NormalizationAgent._ENTITY_ALLOWED_PREFIXES.get(
                    entity_type
                )
                if (
                    current_id and prefixes
                    and not str(current_id).startswith(prefixes)
                    and not is_normal_state(field, entry.get("value"), current_id)
                ):
                    findings.append(review_row(
                        row,
                        "cross_namespace_mapping",
                        "review_field_assignment_or_mapping",
                    ))

    # Evaluate existing anatomy mappings against known species constraints.
    uberon = normalizer.graphs.get("uberon")
    if uberon is not None:
        for path, record in records_by_path.items():
            species_ids = {
                entry.get("ontology_id")
                for field in ("species", "organism")
                for _index, entry in entries(record.get(field))
                if str(entry.get("ontology_id", "")).startswith("NCBITaxon:")
            }
            if not species_ids:
                continue
            ancestors_by_species = {
                species_id: NormalizationAgent._taxon_ancestors(
                    species_id, normalizer.graphs
                )
                for species_id in species_ids
            }
            if any(
                ancestors is None
                for ancestors in ancestors_by_species.values()
            ):
                continue
            for field in ("tissue", "organ"):
                for index, entry in entries(record.get(field)):
                    node = uberon.get_node(str(entry.get("ontology_id", "")))
                    if node is None or not node.taxon_ids:
                        continue
                    compatible = any(
                        constraint
                        in ancestors_by_species[species_id]
                        for species_id in species_ids
                        for constraint in node.taxon_ids
                    )
                    if not compatible:
                        findings.append(review_row(
                            base_row(
                                path, input_root, field, index, entry
                            ),
                            "taxon_incompatible_mapping",
                            "replace_or_review_mapping",
                            node.taxon_ids,
                        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pxd", "relative_file", "agent", "field", "value_index", "value",
        "evidence", "current_ontology_id", "current_ontology_name",
        "current_similarity", "proposed_ontology_id",
        "proposed_ontology_name", "proposed_method", "candidate_ids",
        "issue_type", "recommended_action",
    ]
    with (output_dir / "findings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            csv_row = dict(finding)
            csv_row["candidate_ids"] = ";".join(finding["candidate_ids"])
            writer.writerow(csv_row)
    with (output_dir / "findings.jsonl").open("w") as handle:
        for finding in findings:
            handle.write(json.dumps(finding, ensure_ascii=False) + "\n")

    issue_counts = Counter(finding["issue_type"] for finding in findings)
    action_counts = Counter(
        finding["recommended_action"] for finding in findings
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "source_files_scanned": len(files),
        "normalized_values_scanned": scanned_values,
        "source_annotations_modified": False,
        "finding_rows": len(findings),
        "issue_counts": dict(sorted(issue_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "notes": [
            "Rows may overlap when one entry has more than one issue.",
            "Only deterministic lexical/Unimod changes are proposed automatically.",
            "Ambiguous, cross-namespace, and taxon findings remain review-only.",
            "No embedding or LLM inference was run.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
