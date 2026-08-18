#!/usr/bin/env python3
"""
SDRF Benchmark Runner
=====================
Orchestrates the full benchmark workflow:
1. Convert SDRF files → golden-set JSONs
2. Run extraction pipeline (all agents + validation + normalization) on manuscripts
3. Compare LLM outputs against SDRF goldens using semantic matching
4. Generate comparison plots and reports

Usage:
    python benchmark_data/run_sdrf_benchmark.py                  # All 107 PXDs
    python benchmark_data/run_sdrf_benchmark.py --limit 10       # First 10 only
    python benchmark_data/run_sdrf_benchmark.py --skip-extraction # Just comparison
"""

import argparse
import json
import subprocess
import sys
import os
import re
import time
import multiprocessing
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DATA = Path(__file__).resolve().parent

# Add project root + benchmark to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))

from benchmark.semantic_matcher import HierarchicalMatcher, calculate_weighted_metrics
from core.field_mappings import (
    LLM_TO_GOLDEN as FIELD_ALIASES,
    FIELD_ONTOLOGY_MAP,
    METADATA_ONLY_FIELDS,
)

# Ontologies to load for hierarchical matching
HIERARCHICAL_ONTOLOGIES = {
    "cl": "cl.obo",
    "uberon": "uberon.obo",
    "doid": "doid.obo",
}


def _build_reverse_lookup(llm_output: dict, golden_fields: dict) -> dict:
    """Build a mapping from golden_field_name → llm_value.
    
    For each golden field, find the best matching LLM field using aliases.
    Returns: {golden_field_name: llm_value_or_dict}
    """
    # Normalize LLM output keys
    normalized_llm = {}
    for k, v in llm_output.items():
        if k.startswith("_"):  # Skip metadata keys like _confidence
            continue
        # Try direct alias lookup
        golden_key = FIELD_ALIASES.get(k, FIELD_ALIASES.get(k.lower(), k))
        normalized_llm[golden_key] = v
    
    return normalized_llm


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: SDRF → Golden Conversion
# ═══════════════════════════════════════════════════════════════════════

def step_convert_sdrfs(matched_dir: Path, golden_dir: Path, limit: int = None):
    """Convert SDRF files to golden-set format."""
    print("\n" + "="*70)
    print("  STEP 1: Converting SDRFs to Golden-Set Format")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, str(BENCHMARK_DATA / "sdrf_to_golden.py"),
        "--input-dir", str(matched_dir),
        "--output-dir", str(golden_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    
    count = len(list(golden_dir.glob("*_golden.json")))
    print(f"  → {count} golden-set files generated\n")
    return count


# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Run Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════

def _process_single_pxd(args_tuple):
    """Process a single PXD — top-level function for multiprocessing.Pool."""
    pxd_dir, output_dir, config_path, force, idx, total, integrate, runassessor_dir = args_tuple
    pxd_id = pxd_dir.name
    manuscript = pxd_dir / "manuscript.txt"
    
    if not manuscript.exists():
        print(f"  [{idx}/{total}] SKIP {pxd_id}: no manuscript.txt", flush=True)
        return (pxd_id, "skip", None)
    
    pxd_output = output_dir / pxd_id
    
    # Check if already processed (skip unless --force-extraction)
    if not force and (pxd_output / "Biological_annotations").exists():
        existing_jsons = list((pxd_output / "Biological_annotations").rglob("*.json"))
        if existing_jsons:
            print(f"  [{idx}/{total}] SKIP {pxd_id}: already processed ({len(existing_jsons)} files)", flush=True)
            return (pxd_id, "success", None)
    
    # If forcing, clean old output for this PXD
    if force:
        import shutil
        for subdir in ["Biological_annotations", "technical_metadata_output",
                       "experimental_design_output", "normalized_output"]:
            old = pxd_output / subdir
            if old.exists():
                shutil.rmtree(old)
    
    print(f"  [{idx}/{total}] Processing {pxd_id}...", flush=True)
    start = time.time()
    
    # Create a temp input dir with just this manuscript
    tmp_input = pxd_output / "_input"
    tmp_input.mkdir(parents=True, exist_ok=True)
    
    # Symlink manuscript
    link_path = tmp_input / f"{pxd_id}_manuscript.txt"
    if not link_path.exists():
        try:
            os.symlink(manuscript.resolve(), link_path)
        except FileExistsError:
            pass
    
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "all",
        "--validate",
        "--normalize",
        "--input", str(tmp_input),
        "--output", str(pxd_output),
        "--config", str(config_path),
    ]
    
    if integrate and runassessor_dir:
        cmd += ["--integrate", "--runassessor-dir", str(runassessor_dir)]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT)
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"  [{idx}/{total}] {pxd_id} OK ({elapsed:.1f}s)", flush=True)
            return (pxd_id, "success", None)
        else:
            print(f"  [{idx}/{total}] {pxd_id} FAIL ({elapsed:.1f}s)", flush=True)
            # Save error log
            err_file = pxd_output / "error.log"
            with open(err_file, "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
            return (pxd_id, "fail", None)
    
    except subprocess.TimeoutExpired:
        print(f"  [{idx}/{total}] {pxd_id} TIMEOUT (300s)", flush=True)
        return (pxd_id, "fail", "timeout")
    except Exception as e:
        print(f"  [{idx}/{total}] {pxd_id} ERROR: {e}", flush=True)
        return (pxd_id, "fail", str(e))


def step_run_extraction(matched_dir: Path, output_dir: Path, config_path: Path,
                        limit: int = None, force: bool = False, workers: int = 4,
                        integrate: bool = False, runassessor_dir: str = None):
    """Run the extraction pipeline on each manuscript using multiprocessing."""
    print("\n" + "="*70)
    print(f"  STEP 2: Running Extraction Pipeline ({workers} parallel workers)")
    print("="*70 + "\n")
    
    pxd_dirs = sorted([d for d in matched_dir.iterdir() if d.is_dir() and d.name.startswith("PXD")])
    if limit:
        pxd_dirs = pxd_dirs[:limit]
    
    total = len(pxd_dirs)
    
    # Build args for each PXD
    pool_args = [
        (pxd_dir, output_dir, config_path, force, i, total, integrate, runassessor_dir)
        for i, pxd_dir in enumerate(pxd_dirs, 1)
    ]
    
    # Run with multiprocessing pool
    success = 0
    failed = []
    
    if workers > 1:
        print(f"  Processing {total} PXDs with {workers} parallel workers...\n")
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_process_single_pxd, pool_args)
        
        for pxd_id, status, err in results:
            if status == "success":
                success += 1
            elif status == "fail":
                failed.append(pxd_id)
    else:
        # Sequential fallback
        for args_tuple in pool_args:
            pxd_id, status, err = _process_single_pxd(args_tuple)
            if status == "success":
                success += 1
            elif status == "fail":
                failed.append(pxd_id)
    
    print(f"\n  → {success}/{total} succeeded, {len(failed)} failed")
    if failed:
        print(f"  → Failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    return success, failed


# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Compare LLM Outputs vs SDRF Goldens
# ═══════════════════════════════════════════════════════════════════════

def load_golden_set(golden_dir: Path) -> dict:
    """Load golden set JSON files into a lookup dictionary."""
    golden_data = {}
    for json_file in golden_dir.glob("*_golden.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        pxd_id = data.get("pxd_id")
        agent_type = data.get("agent_type")
        if pxd_id and agent_type:
            golden_data[(pxd_id, agent_type)] = data
    return golden_data


def find_llm_outputs(extraction_dir: Path) -> dict:
    """Find all LLM output JSON files organized by agent and PXD.
    
    Returns: {agent_name: {pxd_id: path_to_json}}
    """
    outputs = defaultdict(dict)
    
    # The pipeline outputs in sub-dirs like:
    #   extraction_output/<PXD>/Biological_annotations/temp_0.0/<file>.json
    #   extraction_output/<PXD>/normalized_output/BiologicalAgent/temp_0.0/<file>_normalized.json
    
    agent_dir_mapping = {
        "Biological_annotations": "BiologicalAgent",
        "technical_metadata_output": "TechnicalAgent",
        "experimental_design_output": "ExperimentalDesignAgent",
    }
    
    for pxd_dir in extraction_dir.iterdir():
        if not pxd_dir.is_dir() or not pxd_dir.name.startswith("PXD"):
            continue
        pxd_id = pxd_dir.name
        
        # Check for integrated output first (preferred), then normalized, then raw
        for dir_name, agent_name in agent_dir_mapping.items():
            # Check integrated output (enriched with runassessor)
            int_dir = pxd_dir / "integrated_output" / agent_name
            if int_dir.exists():
                for jf in int_dir.rglob("*.json"):
                    if jf.name == "ra_disagreements.json":
                        continue
                    outputs[agent_name][pxd_id] = jf
                    break
            else:
                # Check normalized output
                norm_dir = pxd_dir / "normalized_output" / agent_name
                if norm_dir.exists():
                    for jf in norm_dir.rglob("*.json"):
                        outputs[agent_name][pxd_id] = jf
                        break
                else:
                    # Fall back to raw agent output
                    agent_dir = pxd_dir / dir_name
                    if agent_dir.exists():
                        for jf in agent_dir.rglob("*.json"):
                            outputs[agent_name][pxd_id] = jf
                            break
    
    return dict(outputs)


# METADATA_ONLY_FIELDS imported from core.field_mappings


# ═══════════════════════════════════════════════════════════════════════
#  Label Normalization: SDRF channel labels → reagent name
# ═══════════════════════════════════════════════════════════════════════

def _normalize_label_value(value: str) -> str:
    """Normalize SDRF per-channel labels to reagent type names.
    
    Converts channel-level encoding (e.g. 'TMT126; TMT127; TMT128') 
    to reagent name (e.g. 'TMT 3-plex'). This is a deterministic
    mapping that bridges the representation gap between SDRF encoding
    and LLM extraction.
    """
    if not value or not isinstance(value, str):
        return value
    
    val_lower = value.lower().strip()
    
    # TMT channels → TMT N-plex
    if 'tmt' in val_lower and ';' in val_lower:
        channels = [c.strip() for c in value.split(';') if c.strip()]
        n = len(channels)
        if all('tmt' in c.lower() for c in channels):
            return f"TMT {n}-plex"
    
    # iTRAQ channels → iTRAQ N-plex
    if 'itraq' in val_lower and ';' in val_lower:
        channels = [c.strip() for c in value.split(';') if c.strip()]
        n = len(channels)
        if all('itraq' in c.lower() for c in channels):
            return f"iTRAQ {n}-plex"
    
    # SILAC channels → SILAC
    if 'silac' in val_lower and ';' in val_lower:
        return "SILAC"
    
    return value


def step_compare(golden_dir: Path, extraction_dir: Path, reports_dir: Path,
                 filter_extractable: bool = True):
    """Compare LLM outputs against SDRF goldens using semantic matching."""
    print("\n" + "="*70)
    print("  STEP 3: Comparing LLM Outputs vs SDRF Golden Set (Semantic Matching)")
    if filter_extractable:
        print(f"  (Filtered to manuscript-extractable fields only, excluding {len(METADATA_ONLY_FIELDS)} metadata-only fields)")
    print("="*70 + "\n")
    
    # Load goldens
    golden_data = load_golden_set(golden_dir)
    print(f"  Loaded {len(golden_data)} golden-set entries")
    
    # Find LLM outputs
    llm_outputs = find_llm_outputs(extraction_dir)
    for agent, pxds in llm_outputs.items():
        print(f"  Found {len(pxds)} LLM outputs for {agent}")
    
    # ── Load ontologies for hierarchical matching ──
    term_normalizer = None
    try:
        from normalization.normalizer import TermNormalizer
        from normalization.config import NormalizationConfig
        
        config = NormalizationConfig()
        ontology_dir = Path(config.ontology_dir)
        if not ontology_dir.is_absolute():
            ontology_dir = PROJECT_ROOT / ontology_dir
        
        term_normalizer = TermNormalizer(config)
        loaded_ontologies = []
        for ont_id, ont_file in HIERARCHICAL_ONTOLOGIES.items():
            ont_path = ontology_dir / ont_file
            if ont_path.exists():
                print(f"  Loading ontology: {ont_id} ({ont_path.name})...")
                term_normalizer.load_ontology(ont_id, str(ont_path))
                loaded_ontologies.append(ont_id)
            else:
                print(f"  WARNING: Ontology file not found: {ont_path}")
        
        if loaded_ontologies:
            print(f"  Loaded {len(loaded_ontologies)} ontologies for hierarchical matching: {loaded_ontologies}")
        else:
            print("  WARNING: No ontologies loaded, hierarchical matching disabled")
            term_normalizer = None
    except Exception as e:
        print(f"  WARNING: Could not load ontologies for hierarchical matching: {e}")
        term_normalizer = None
    
    # Initialize matcher
    print("\n  Loading SciBERT for semantic matching...")
    matcher = HierarchicalMatcher(
        semantic_threshold=0.70,
        term_normalizer=term_normalizer,
        field_ontology_map=FIELD_ONTOLOGY_MAP if term_normalizer else {},
    )
    
    all_agent_metrics = {}
    all_results = {}
    all_field_stats = {}
    
    for agent_name in ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]:
        pxd_outputs = llm_outputs.get(agent_name, {})
        if not pxd_outputs:
            print(f"\n  [{agent_name}] No outputs found, skipping")
            continue
        
        print(f"\n  [{agent_name}] Comparing {len(pxd_outputs)} outputs...")
        
        results = []
        field_stats = defaultdict(lambda: {
            "EXACT": 0, "NORMALIZED": 0, "ONTOLOGY": 0, "HIERARCHICAL": 0,
            "SEMANTIC": 0, "NO_MATCH": 0,
            "total": 0, "scores": [], "similarities": []
        })
        
        matched_pxds = 0
        for pxd_id, json_path in sorted(pxd_outputs.items()):
            key = (pxd_id, agent_name)
            if key not in golden_data:
                continue
            
            matched_pxds += 1
            
            with open(json_path) as f:
                llm_output = json.load(f)
            
            # If this is an enriched/integrated output, extract 'resolved' values
            # and merge with the normalized/raw LLM output to retain fields the
            # integration agent doesn't cover (e.g. cleavage_agent, label, etc.)
            is_enriched = any(
                isinstance(v, dict) and "resolved" in v
                for v in llm_output.values() if isinstance(v, dict)
            )
            if is_enriched:
                resolved_output = {}
                for k, v in llm_output.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, dict) and "resolved" in v:
                        resolved_val = v["resolved"]
                        if resolved_val and str(resolved_val).lower() not in ("unknown", "none", "null", "", "??"):
                            resolved_output[k] = resolved_val
                        else:
                            resolved_output[k] = None
                    else:
                        resolved_output[k] = v
                
                # Merge with normalized/raw LLM output for fields the integration
                # agent doesn't cover
                pxd_dir = json_path.parent.parent.parent.parent  # up from temp_0.0/Agent/integrated_output
                agent_dir_map = {
                    "BiologicalAgent": "Biological_annotations",
                    "TechnicalAgent": "technical_metadata_output",
                    "ExperimentalDesignAgent": "experimental_design_output",
                }
                norm_dir = pxd_dir / "normalized_output" / agent_name
                raw_dir_name = agent_dir_map.get(agent_name)
                raw_dir = pxd_dir / raw_dir_name if raw_dir_name else None
                
                fallback_path = None
                if norm_dir and norm_dir.exists():
                    fallback_files = list(norm_dir.rglob("*.json"))
                    if fallback_files:
                        fallback_path = fallback_files[0]
                elif raw_dir and raw_dir.exists():
                    fallback_files = list(raw_dir.rglob("*.json"))
                    if fallback_files:
                        fallback_path = fallback_files[0]
                
                if fallback_path:
                    with open(fallback_path) as ff:
                        fallback_data = ff.load() if hasattr(ff, 'load') else json.load(ff)
                    # Add LLM fields that are missing from the resolved output
                    for k, v in fallback_data.items():
                        if k.startswith("_"):
                            continue
                        if k not in resolved_output or resolved_output[k] is None:
                            # Extract value from normalized output format
                            if isinstance(v, dict) and "value" in v:
                                val = v["value"]
                            elif isinstance(v, list) and len(v) >= 1:
                                val = v[0]
                            else:
                                val = v
                            if val and str(val).lower() not in ("unknown", "none", "null", ""):
                                resolved_output[k] = val
                
                llm_output = resolved_output
            
            golden = golden_data[key]
            golden_fields = golden.get("fields", {})
            
            # Normalize LLM field names to match golden field names via aliases
            normalized_llm = _build_reverse_lookup(llm_output, golden_fields)

            for field_name, golden_value in golden_fields.items():
                if golden_value is None:
                    continue
                
                # Skip metadata-only fields if filtering is enabled
                if filter_extractable and field_name in METADATA_ONLY_FIELDS:
                    continue
                
                # Normalize label values before comparison
                compare_golden = golden_value
                if field_name == "label" and isinstance(golden_value, str):
                    compare_golden = _normalize_label_value(golden_value)
                
                # Look up in normalized LLM output (field names already aliased)
                llm_field = normalized_llm.get(field_name, {})

                
                # Use hierarchical semantic matching (pass field_name for hierarchy)
                match_type, score = matcher.compare(
                    llm_field, compare_golden, field_name=field_name
                )
                
                llm_val = matcher.extract_value(llm_field)
                golden_val = str(golden_value) if golden_value else None
                
                field_stats[field_name]["total"] += 1
                field_stats[field_name][match_type] += 1
                field_stats[field_name]["scores"].append(score)
                
                if llm_val and golden_val:
                    try:
                        sim = matcher.semantic_matcher.cosine_similarity(llm_val, golden_val)
                        field_stats[field_name]["similarities"].append(sim)
                    except Exception:
                        pass
                
                results.append({
                    "pxd_id": pxd_id,
                    "field": field_name,
                    "golden": golden_value,
                    "llm": llm_val,
                    "match_type": match_type,
                    "score": score
                })
        
        print(f"    Matched {matched_pxds} PXDs, {len(results)} field comparisons")
        
        # Calculate weighted metrics
        comparison_results = [
            {"score": r["score"], "llm_has_value": r["llm"] is not None, "golden_has_value": r["golden"] is not None}
            for r in results
        ]
        weighted_metrics = calculate_weighted_metrics(comparison_results)
        
        match_counts = defaultdict(int)
        for r in results:
            match_counts[r["match_type"]] += 1
        
        all_agent_metrics[agent_name] = {**weighted_metrics, **dict(match_counts)}
        all_results[agent_name] = results
        all_field_stats[agent_name] = dict(field_stats)
        
        print(f"    Weighted P={weighted_metrics['weighted_precision']:.3f}, "
              f"R={weighted_metrics['weighted_recall']:.3f}, "
              f"F1={weighted_metrics['weighted_f1']:.3f}")
        print(f"    Match types: {dict(match_counts)}")
    
    return all_agent_metrics, all_results, all_field_stats


# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Generate Plots and Reports
# ═══════════════════════════════════════════════════════════════════════

def step_generate_plots(all_agent_metrics: dict, all_results: dict, 
                        all_field_stats: dict, reports_dir: Path, model_name: str = "Llama-4-Scout"):
    """Generate comparison plots and CSV reports."""
    print("\n" + "="*70)
    print("  STEP 4: Generating Plots & Reports")
    print("="*70 + "\n")
    
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Per-Agent Field Comparison Plots ──
    for agent_name, field_stats in all_field_stats.items():
        _plot_agent_fields(agent_name, field_stats, plots_dir, model_name)
        _save_field_csv(agent_name, field_stats, reports_dir)
    
    # ── Summary Chart ──
    if all_agent_metrics:
        _plot_summary(all_agent_metrics, plots_dir, model_name)
    
    # ── Per-PXD Results CSV ──
    all_rows = []
    for agent_name, results in all_results.items():
        for r in results:
            all_rows.append({"agent": agent_name, **r})
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = reports_dir / "sdrf_benchmark_detailed.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # 3. Overall single chart
    # Calculate total PXDs compared
    total_pxds = len(set(r["pxd_id"] for results in all_results.values() for r in results))
    _plot_overall_metrics(all_agent_metrics, plots_dir, n_pxds=total_pxds, model_name=model_name)
    _plot_overall_agent_metrics(all_agent_metrics, plots_dir, n_pxds=total_pxds, model_name=model_name)
    
    # ── Summary JSON ──
    summary = {
        "metrics": {},
        "total_pxds_compared": len(set(r["pxd_id"] for results in all_results.values() for r in results)),
    }
    for agent, metrics in all_agent_metrics.items():
        summary["metrics"][agent] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
    
    # Overall macro-averaged metrics
    if all_agent_metrics:
        n = len(all_agent_metrics)
        summary["overall"] = {
            "precision": round(sum(m.get("weighted_precision", 0) for m in all_agent_metrics.values()) / n, 4),
            "recall": round(sum(m.get("weighted_recall", 0) for m in all_agent_metrics.values()) / n, 4),
            "f1": round(sum(m.get("weighted_f1", 0) for m in all_agent_metrics.values()) / n, 4),
            "method": "macro-average across agents"
        }
    
    with open(reports_dir / "sdrf_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {reports_dir / 'sdrf_benchmark_summary.json'}")


def _plot_agent_fields(agent_name: str, field_stats: dict, plots_dir: Path, model_name: str):
    """Create per-agent field comparison plot (4 subplots)."""
    metrics = []
    for field, stats in field_stats.items():
        if stats["total"] == 0:
            continue
        avg_score = sum(stats["scores"]) / stats["total"]
        matches = stats["EXACT"] + stats["NORMALIZED"] + stats["ONTOLOGY"] + stats["HIERARCHICAL"] + stats["SEMANTIC"]
        match_rate = matches / stats["total"]
        
        metrics.append({
            "field": field, "avg_score": avg_score, "match_rate": match_rate,
            "exact": stats["EXACT"], "normalized": stats["NORMALIZED"],
            "ontology": stats["ONTOLOGY"], "hierarchical": stats["HIERARCHICAL"],
            "semantic": stats["SEMANTIC"],
            "no_match": stats["NO_MATCH"], "total": stats["total"]
        })
    
    if not metrics:
        return
    
    df = pd.DataFrame(metrics).sort_values("avg_score", ascending=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"SDRF Benchmark: {agent_name}\n({model_name} vs SDRF Ground Truth)", 
                 fontsize=14, fontweight="bold")
    
    # Plot 1: Average Score by Field
    ax = axes[0, 0]
    colors = ["#e74c3c" if s < 0.3 else "#f39c12" if s < 0.6 else "#2ecc71" for s in df["avg_score"]]
    bars = ax.barh(df["field"], df["avg_score"], color=colors)
    ax.set_xlabel("Average Match Score")
    ax.set_title("Weighted Score by Field")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, df["avg_score"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=8)
    
    # Plot 2: Match Type Distribution (Stacked)
    ax = axes[0, 1]
    fields = df["field"].tolist()
    bottom = np.zeros(len(fields))
    stack_colors = ["#27ae60", "#2ecc71", "#82e0aa", "#58d68d", "#f39c12", "#e74c3c"]
    stack_labels = ["Exact", "Normalized", "Ontology", "Hierarchical", "Semantic", "No Match"]
    
    for col, color, label in zip(["exact", "normalized", "ontology", "hierarchical", "semantic", "no_match"], stack_colors, stack_labels):
        values = df[col].values
        ax.barh(fields, values, left=bottom, color=color, label=label, height=0.6)
        bottom += values
    
    ax.set_xlabel("Count")
    ax.set_title("Match Type Distribution by Field")
    ax.legend(loc="lower right", fontsize=8)
    
    # Plot 3: Similarity Distribution
    ax = axes[1, 0]
    all_sims = [s for stats in field_stats.values() for s in stats.get("similarities", [])]
    if all_sims:
        ax.hist(all_sims, bins=25, range=(0, 1), color="#3498db", edgecolor="black", alpha=0.7)
        ax.axvline(x=0.75, color="red", linestyle="--", label="Threshold (0.75)")
        ax.set_xlabel("Semantic Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Semantic Similarities")
        ax.legend()
    
    # Plot 4: Match Type Pie
    ax = axes[1, 1]
    counts = [df["exact"].sum(), df["normalized"].sum(), df["ontology"].sum(),
              df["hierarchical"].sum(), df["semantic"].sum(), df["no_match"].sum()]
    if sum(counts) > 0:
        wedges, texts, autotexts = ax.pie(
            counts, labels=stack_labels, colors=stack_colors, 
            autopct="%1.1f%%", startangle=90
        )
        ax.set_title("Overall Match Type Distribution")
    
    plt.tight_layout()
    out_path = plots_dir / f"{agent_name.lower()}_sdrf_benchmark.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def _save_field_csv(agent_name: str, field_stats: dict, reports_dir: Path):
    """Save per-field metrics CSV."""
    rows = []
    for field, stats in field_stats.items():
        if stats["total"] == 0:
            continue
        avg_score = sum(stats["scores"]) / stats["total"]
        matches = stats["EXACT"] + stats["NORMALIZED"] + stats["ONTOLOGY"] + stats["HIERARCHICAL"] + stats["SEMANTIC"]
        rows.append({
            "field": field, "avg_score": round(avg_score, 4),
            "match_rate": round(matches / stats["total"], 4),
            "exact": stats["EXACT"], "normalized": stats["NORMALIZED"],
            "ontology": stats["ONTOLOGY"], "hierarchical": stats["HIERARCHICAL"],
            "semantic": stats["SEMANTIC"],
            "no_match": stats["NO_MATCH"], "total": stats["total"]
        })
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = reports_dir / f"{agent_name.lower()}_sdrf_field_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def _plot_summary(all_agent_metrics: dict, plots_dir: Path, model_name: str):
    """Create summary P/R/F1 bar chart across all agents."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"SDRF Benchmark: {model_name} vs SDRF Ground Truth\nSemantic Matching Summary",
                 fontsize=14, fontweight="bold")
    
    agents = list(all_agent_metrics.keys())
    x = np.arange(len(agents))
    width = 0.25
    
    precisions = [m.get("weighted_precision", 0) for m in all_agent_metrics.values()]
    recalls = [m.get("weighted_recall", 0) for m in all_agent_metrics.values()]
    f1s = [m.get("weighted_f1", 0) for m in all_agent_metrics.values()]
    
    # Left: P/R/F1
    ax = axes[0]
    bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
    bars2 = ax.bar(x, recalls, width, label="Recall", color="#e74c3c")
    bars3 = ax.bar(x + width, f1s, width, label="F1", color="#2ecc71")
    
    ax.set_xlabel("Agent")
    ax.set_ylabel("Score")
    ax.set_title("Weighted Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("Agent", "") for a in agents], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    
    # Right: Match type breakdown
    ax = axes[1]
    match_types = ["EXACT", "NORMALIZED", "ONTOLOGY", "HIERARCHICAL", "SEMANTIC", "NO_MATCH"]
    mt_colors = ["#27ae60", "#2ecc71", "#82e0aa", "#58d68d", "#f39c12", "#e74c3c"]
    bottom = np.zeros(len(agents))
    
    for mt, color in zip(match_types, mt_colors):
        vals = [all_agent_metrics[a].get(mt, 0) for a in agents]
        ax.bar(x, vals, 0.5, label=mt.title().replace("_", " "), bottom=bottom, color=color)
        bottom += np.array(vals)
    
    ax.set_xlabel("Agent")
    ax.set_ylabel("Count")
    ax.set_title("Match Type Distribution by Agent")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("Agent", "") for a in agents], rotation=15)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    out_path = plots_dir / "sdrf_benchmark_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_overall_metrics(all_agent_metrics: dict, plots_dir: Path, n_pxds: int, model_name: str):
    """Create a single overall P/R/F1 plot (macro-averaged across agents)."""
    n = len(all_agent_metrics)
    if n == 0: return

    overall_p = sum(m.get("weighted_precision", 0) for m in all_agent_metrics.values()) / n
    overall_r = sum(m.get("weighted_recall", 0) for m in all_agent_metrics.values()) / n
    overall_f1 = sum(m.get("weighted_f1", 0) for m in all_agent_metrics.values()) / n
    
    metrics = {'Precision': overall_p, 'Recall': overall_r, 'F1': overall_f1}

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.5, edgecolor='white', linewidth=1.5)
 
    # Value labels
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Overall Benchmark ({model_name} vs SDRF)\n{n_pxds} PXDs · Manuscript-Extractable Fields', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    out_path = plots_dir / "sdrf_benchmark_overall.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_overall_agent_metrics(all_agent_metrics: dict, plots_dir: Path,
                                n_pxds: int, model_name: str):
    """Create grouped bar chart showing P/R/F1 per agent plus overall."""
    n = len(all_agent_metrics)
    if n == 0:
        return

    overall_p = sum(m.get("weighted_precision", 0) for m in all_agent_metrics.values()) / n
    overall_r = sum(m.get("weighted_recall", 0) for m in all_agent_metrics.values()) / n
    overall_f1 = sum(m.get("weighted_f1", 0) for m in all_agent_metrics.values()) / n

    agents = ['Biological', 'Technical', 'Exp. Design', 'Overall']
    precision = [
        all_agent_metrics.get('BiologicalAgent', {}).get('weighted_precision', 0),
        all_agent_metrics.get('TechnicalAgent', {}).get('weighted_precision', 0),
        all_agent_metrics.get('ExperimentalDesignAgent', {}).get('weighted_precision', 0),
        overall_p
    ]
    recall = [
        all_agent_metrics.get('BiologicalAgent', {}).get('weighted_recall', 0),
        all_agent_metrics.get('TechnicalAgent', {}).get('weighted_recall', 0),
        all_agent_metrics.get('ExperimentalDesignAgent', {}).get('weighted_recall', 0),
        overall_r
    ]
    f1 = [
        all_agent_metrics.get('BiologicalAgent', {}).get('weighted_f1', 0),
        all_agent_metrics.get('TechnicalAgent', {}).get('weighted_f1', 0),
        all_agent_metrics.get('ExperimentalDesignAgent', {}).get('weighted_f1', 0),
        overall_f1
    ]

    x = np.arange(len(agents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#6BAED6', '#FD8D3C', '#74C476']
    bars1 = ax.bar(x - width, precision, width, label='Precision', color=colors[0],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label='Recall', color=colors[1],
                   edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color=colors[2],
                   edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Overall & Per-Agent Metrics \u2014 SDRF Benchmark\n'
                 f'{n_pxds} Test Set PXDs \u00b7 {model_name}',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.2)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            weight = 'bold' if bars == bars3 else 'normal'
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight=weight)

    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    out_path = plots_dir / "overall_agent_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run SDRF Benchmark Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N PXDs (for testing)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip pipeline extraction, only compare+plot")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip SDRF→golden conversion (reuse existing)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Include ALL fields (including metadata-only fields not in manuscripts)")
    parser.add_argument("--force-extraction", action="store_true",
                        help="Clear existing outputs and re-extract (for prompt changes)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to LLM config YAML (default: project config.yaml)")
    parser.add_argument("--model-label", type=str, default=None,
                        help="Model label for output dirs/reports (e.g. 'gpt', 'gemini')")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel PXD workers (default: 4)")
    parser.add_argument("--input-dir", type=str, default="matched",
                        help="Input directory name (e.g. 'matched' or 'test_set') within benchmark_data")
    parser.add_argument("--integrate", action="store_true",
                        help="Enable integration agent with runassessor/aggregated data")
    parser.add_argument("--runassessor-dir", type=str, default=None,
                        help="Directory containing aggregated results JSON files for integration")
    args = parser.parse_args()
    filter_extractable = not args.no_filter
    
    # Resolve config path
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    
    # Determine model label from config or flag
    model_label = args.model_label
    if not model_label:
        # Auto-detect from config filename
        stem = config_path.stem  # e.g. 'config_gpt' -> 'gpt'
        if "_" in stem:
            model_label = stem.split("_", 1)[1]
        else:
            model_label = None  # default (Llama)
    
    # Set up dirs based on model label
    matched_dir = BENCHMARK_DATA / args.input_dir
    golden_dir = BENCHMARK_DATA / "sdrf_golden"
    
    if model_label:
        extraction_dir = BENCHMARK_DATA / f"extraction_output_{args.input_dir}_{model_label}"
        reports_dir = BENCHMARK_DATA / f"reports_{args.input_dir}_{model_label}"
        display_name = model_label.upper()
    else:
        extraction_dir = BENCHMARK_DATA / f"extraction_output_{args.input_dir}"
        reports_dir = BENCHMARK_DATA / f"reports_{args.input_dir}"
        display_name = "Llama-4-Scout"
    
    extraction_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  SDRF BENCHMARK PIPELINE")
    print(f"  Model: {display_name}")
    print(f"  Config: {config_path}")
    print(f"  Workers: {args.workers}")
    print(f"  Flags: --validate --normalize")
    print(f"  Limit: {'ALL' if not args.limit else args.limit}")
    print("="*70)
    
    # Step 1
    if not args.skip_conversion:
        step_convert_sdrfs(matched_dir, golden_dir, args.limit)
    else:
        print("\n  [SKIP] SDRF conversion (reusing existing goldens)")
    
    # Step 2
    if not args.skip_extraction:
        step_run_extraction(matched_dir, extraction_dir, config_path,
                            args.limit, force=args.force_extraction,
                            workers=args.workers,
                            integrate=args.integrate,
                            runassessor_dir=args.runassessor_dir)
    else:
        print("\n  [SKIP] Pipeline extraction")
    
    # Step 3
    all_agent_metrics, all_results, all_field_stats = step_compare(
        golden_dir, extraction_dir, reports_dir,
        filter_extractable=filter_extractable
    )
    
    # Step 4
    if all_results:
        step_generate_plots(all_agent_metrics, all_results, all_field_stats, reports_dir,
                            model_name=display_name)
    else:
        print("\n  No results to plot!")
    
    # Final summary
    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE")
    print("="*70)
    for agent, metrics in all_agent_metrics.items():
        print(f"\n  {agent}:")
        print(f"    Weighted Precision: {metrics.get('weighted_precision', 0):.3f}")
        print(f"    Weighted Recall:    {metrics.get('weighted_recall', 0):.3f}")
        print(f"    Weighted F1:        {metrics.get('weighted_f1', 0):.3f}")
    
    # Overall (macro-averaged across agents)
    if all_agent_metrics:
        n = len(all_agent_metrics)
        overall_p = sum(m.get("weighted_precision", 0) for m in all_agent_metrics.values()) / n
        overall_r = sum(m.get("weighted_recall", 0) for m in all_agent_metrics.values()) / n
        overall_f1 = sum(m.get("weighted_f1", 0) for m in all_agent_metrics.values()) / n
        
        print(f"\n  {'─'*40}")
        print(f"  OVERALL (macro-averaged across {n} agents):")
        print(f"    Precision: {overall_p:.3f}")
        print(f"    Recall:    {overall_r:.3f}")
        print(f"    F1:        {overall_f1:.3f}")
    print()


if __name__ == "__main__":
    main()
