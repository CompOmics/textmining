#!/usr/bin/env python3
"""
Annotation JSON -> Golden Set Converter
=======================================
Converts *_SDRFannotation_prompt_annotated.json files into the golden-set
JSON format used by the benchmark evaluator.

Each annotation JSON produces 3 golden JSONs (one per agent):
  - {PXD}_BiologicalAgent_golden.json
  - {PXD}_TechnicalAgent_golden.json
  - {PXD}_ExperimentalDesignAgent_golden.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path for core.field_mappings import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.field_mappings import ANNOTATION_TO_GOLDEN as FIELD_MAPPING


def convert_annotation(annotation_path: Path, pxd_id: str) -> dict:
    """Parse annotation JSON and return 3 golden-set dicts (one per agent)."""
    
    with open(annotation_path) as f:
        data = json.load(f)
    
    characteristics = data.get("characteristics", {})
    
    # Initialize agent dicts
    agents = {
        "BiologicalAgent":         {},
        "TechnicalAgent":          {},
        "ExperimentalDesignAgent": {},
    }
    
    for char_key, (golden_field, agent) in FIELD_MAPPING.items():
        values = characteristics.get(char_key, [])
        
        if values and isinstance(values, list):
            # Join multiple values with "; " to match golden format
            joined = "; ".join(str(v) for v in values if v)
            agents[agent][golden_field] = joined if joined else None
        else:
            agents[agent][golden_field] = None
    
    # Build the 3 golden dicts
    result = {}
    for agent_name, fields in agents.items():
        result[agent_name] = {
            "pxd_id": pxd_id,
            "agent_type": agent_name,
            "fields": fields,
            "_source": "annotation_json",
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotation JSONs to golden-set format"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing PXD subdirs with *_annotation.json files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for golden-set JSONs"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    skipped = 0
    
    for pxd_dir in sorted(input_dir.iterdir()):
        if not pxd_dir.is_dir() or not pxd_dir.name.startswith("PXD"):
            continue
        
        pxd_id = pxd_dir.name
        
        # Find annotation JSON
        annotation_files = list(pxd_dir.glob("*_annotation.json"))
        if not annotation_files:
            print(f"  SKIP {pxd_id}: no annotation JSON found")
            skipped += 1
            continue
        
        annotation_path = annotation_files[0]
        
        try:
            agent_goldens = convert_annotation(annotation_path, pxd_id)
            
            for agent_name, golden in agent_goldens.items():
                out_file = output_dir / f"{pxd_id}_{agent_name}_golden.json"
                with open(out_file, "w") as f:
                    json.dump(golden, f, indent=2)
            
            converted += 1
            print(f"  OK {pxd_id}")
        
        except Exception as e:
            print(f"  ERROR {pxd_id}: {e}")
            skipped += 1
    
    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
