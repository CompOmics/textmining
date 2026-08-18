#!/usr/bin/env python3
"""
Per-PXD Pipeline Runner
=======================
Runs the full extraction pipeline (all 3 agents + normalization + integration)
on a selected set of PXDs, with output organized per-PXD for provenance.

Output structure per PXD:
  {output_dir}/{PXD}/
    _input/                          # manuscript text
    Biological_annotations/temp_0.0/ # raw extraction
    technical_metadata_output/temp_0.0/
    experimental_design_output/temp_0.0/
    normalized_output/{Agent}/temp_0.0/  # ontology-normalized
    integrated_output/{Agent}/temp_0.0/  # PRIDE-enriched
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.biological_agent import BiologicalAgent
from agents.technical_agent import TechnicalAgent
from agents.experimental_agent import ExperimentalDesignAgent
from agents.integration_agent import IntegrationAgent
from agents.normalization_agent import NormalizationAgent
from core.reproducibility import set_seed
import yaml


def load_config():
    """Load config.yaml from project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    config = {
        "agents": {"temperatures": [0.0], "validate": True, "max_retries": 1, "confidence_threshold": 0.6},
        "concurrency": {"max_workers": 4},
        "paths": {"ontology_dir": "ontologies"},
    }
    if config_path.exists():
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
            for section in ("agents", "concurrency", "paths", "llm", "reproducibility"):
                if section in loaded:
                    if section in config:
                        config[section].update(loaded[section])
                    else:
                        config[section] = loaded[section]
    return config


def run_pxd(pxd_id: str, manuscript_path: str, runassessor_path: str,
            output_base: str, config: dict):
    """Run the full pipeline for a single PXD."""
    
    pxd_dir = Path(output_base) / pxd_id
    input_dir = pxd_dir / "_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage manuscript
    ms_dest = input_dir / f"{pxd_id}_manuscript.txt"
    if not ms_dest.exists():
        shutil.copy2(manuscript_path, ms_dest)
    
    # Config values
    temperatures = config["agents"].get("temperatures", [0.0])
    validate = config["agents"].get("validate", True)
    max_workers = config["concurrency"].get("max_workers", 4)
    llm_config = config.get("llm", {})
    max_retries = config["agents"].get("max_retries", 1)
    confidence_threshold = config["agents"].get("confidence_threshold", 0.6)
    ontology_dir = config["paths"].get("ontology_dir", "ontologies")
    
    common_kwargs = dict(
        temperatures=temperatures,
        use_validation=validate,
        max_workers=1,  # single file per PXD, no parallelism needed
        llm_config=llm_config,
        max_retries=max_retries,
        confidence_threshold=confidence_threshold,
    )
    
    # ── Step 1: Extraction (all 3 agents) ──
    agents = [
        ("BiologicalAgent", BiologicalAgent(str(input_dir), str(pxd_dir / "Biological_annotations"), **common_kwargs)),
        ("TechnicalAgent", TechnicalAgent(str(input_dir), str(pxd_dir / "technical_metadata_output"), **common_kwargs)),
        ("ExperimentalDesignAgent", ExperimentalDesignAgent(str(input_dir), str(pxd_dir / "experimental_design_output"), **common_kwargs)),
    ]
    
    pipeline_results = {}
    for agent_name, agent in agents:
        print(f"  [{pxd_id}] Running {agent_name}...")
        pipeline_results[agent_name] = agent.run()
    
    # ── Step 2: Normalization ──
    print(f"  [{pxd_id}] Normalizing...")
    try:
        normalizer = NormalizationAgent(ontology_dir)
        
        for agent_name, temp_results in pipeline_results.items():
            for temp, file_results in temp_results.items():
                normalized = normalizer.normalize_batch(file_results)
                pipeline_results[agent_name][temp] = normalized
                
                out_dir = pxd_dir / "normalized_output" / agent_name / f"temp_{temp:.1f}"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                for filename, data in normalized.items():
                    out_file = out_dir / (Path(filename).stem + "_normalized.json")
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [{pxd_id}] WARNING: Normalization failed: {e}")
    
    # ── Step 3: Integration ──
    if runassessor_path and Path(runassessor_path).exists():
        print(f"  [{pxd_id}] Integrating with PRIDE data...")
        # IntegrationAgent expects a directory with runassessor files
        ra_staging = pxd_dir / "_runassessor"
        ra_staging.mkdir(parents=True, exist_ok=True)
        ra_dest = ra_staging / f"{pxd_id}_aggregated_results.json"
        if not ra_dest.exists():
            shutil.copy2(runassessor_path, ra_dest)
        
        try:
            integrator = IntegrationAgent(str(ra_staging))
            
            for agent_name, temp_results in pipeline_results.items():
                for temp, file_results in temp_results.items():
                    out_dir = pxd_dir / "integrated_output" / agent_name / f"temp_{temp:.1f}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    enriched = integrator.enrich_batch(
                        file_results, agent_name=agent_name, output_dir=str(out_dir)
                    )
                    
                    for filename, data in enriched.items():
                        out_file = out_dir / (Path(filename).stem + "_enriched.json")
                        with open(out_file, 'w') as f:
                            json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [{pxd_id}] WARNING: Integration failed: {e}")
    else:
        print(f"  [{pxd_id}] Skipping integration (no runassessor data)")
    
    print(f"  [{pxd_id}] DONE")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline per-PXD with provenance")
    parser.add_argument("--mapping", type=str, required=True,
                        help="Path to dataset_mapping.json")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "new_test"],
                        help="Which split to use")
    parser.add_argument("--output", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of PXDs to process (default: 20)")
    parser.add_argument("--three-source-only", action="store_true", default=True,
                        help="Only include PXDs with all 3 data sources")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Load dataset mapping
    with open(args.mapping) as f:
        mapping = json.load(f)
    
    split = mapping[args.split]
    
    # Filter to 3-source PXDs if requested
    if args.three_source_only:
        candidates = {pxd: files for pxd, files in split.items() if len(files) == 3}
    else:
        candidates = split
    
    # Select N PXDs
    selected = sorted(candidates.keys())[:args.n]
    print(f"Selected {len(selected)} PXDs from '{args.split}' split")
    
    # Load config
    config = load_config()
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed set to {args.seed}")
    
    # Run pipeline per PXD
    start_time = time.time()
    for i, pxd_id in enumerate(selected, 1):
        files = candidates[pxd_id]
        manuscript = files[1]  # index 1 is always manuscript
        runassessor = files[2] if len(files) > 2 else None
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] Processing {pxd_id}")
        print(f"{'='*60}")
        
        try:
            run_pxd(pxd_id, manuscript, runassessor, args.output, config)
        except Exception as e:
            print(f"  ERROR processing {pxd_id}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed {len(selected)} PXDs in {elapsed:.0f}s ({elapsed/len(selected):.1f}s/PXD)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
