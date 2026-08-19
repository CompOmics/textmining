"""
Combine multi-source metadata into a unified run-level TSV.

Sources (priority: sdrf > name > agent > MLmarker):
  1. SDRF (run_metadata_sdrf.tsv)
  2. Filename extraction (run_meta_name.tsv)
  3. MLMarker tissue prediction (run_meta_mlmarker.tsv)
  4. Agent/manuscript extraction (metadata.tsv)

Output: run_metadata_combined.tsv
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALUE_FIELDS = [
    "organism", "tissue", "disease", "cell_part", "cell_line",
    "instrument", "fragmentation", "enzymes", "modifications",
    "collision_energy", "gradient_time_min", "lc_column",
    "acquisition", "labeling", "ionization",
    "treatment_type", "treatment_name", "treatment_class",
    "fractionation", "enrichment",
]

SDRF_FIELDS = [
    "organism", "tissue", "disease", "cell_line", "instrument",
    "fragmentation", "collision_energy", "enzymes", "modifications",
    "labeling", "enrichment", "fractionation",
]

NAME_FIELDS = [
    "organism", "tissue", "disease", "cell_part", "cell_line",
    "instrument", "fragmentation", "acquisition", "labeling",
    "enrichment", "fractionation",
]

MLM_FIELDS = ["tissue"]

# Fields used to match agent setups to runs
MATCH_FIELDS = ["organism", "tissue", "cell_line", "instrument", "labeling"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(val):
    """Normalize a value for comparison: lowercase, strip whitespace."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val).strip().lower()


def _norm_set(val):
    """Normalize a potentially multi-value field into a set of values."""
    n = _norm(val)
    if n is None:
        return set()
    return {v.strip() for v in n.split(";") if v.strip()}


def _values_agree(existing, incoming):
    """Check if two values agree (case-insensitive, multi-value aware)."""
    e_set = _norm_set(existing)
    i_set = _norm_set(incoming)
    if not e_set or not i_set:
        return True  # can't disagree if one is empty
    # Agree if any overlap (e.g., "HCD" in incoming matches "HCD; ETD" in existing)
    return bool(e_set & i_set)


def _normalize_multivalue(val):
    """Deduplicate and sort semicolon-separated values for consistent representation."""
    if pd.isna(val) or str(val).strip() == "":
        return val
    parts = [p.strip() for p in str(val).split(";") if p.strip()]
    # Deduplicate preserving first occurrence, then sort
    seen = set()
    unique = []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    unique.sort(key=str.lower)
    return "; ".join(unique) if unique else val


def _append_evidence(current_evidence, source_label, agrees):
    """Append a source label to evidence string."""
    prefix = "" if agrees else "-"
    entry = f"{prefix}{source_label}"
    if pd.isna(current_evidence) or str(current_evidence).strip() == "":
        return entry
    return f"{current_evidence}; {entry}"


# ---------------------------------------------------------------------------
# Step 1: Load all sources
# ---------------------------------------------------------------------------

def load_sources():
    print("Loading sources...")

    # Target run index
    pq = pd.read_parquet(
        PROJECT_ROOT / "quant_data" / "pride_quant.parquet",
        columns=["pxd", "run"],
    )
    target = pq.drop_duplicates(subset=["pxd", "run"]).reset_index(drop=True)
    print(f"  Target runs: {len(target)}")

    # SDRF
    sdrf = pd.read_csv(PROJECT_ROOT / "notebooks" / "run_metadata_sdrf.tsv", sep="\t")
    print(f"  SDRF rows: {len(sdrf)}")

    # Name extraction
    name = pd.read_csv(PROJECT_ROOT / "notebooks" / "run_meta_name.tsv", sep="\t")
    print(f"  Name rows: {len(name)}")

    # MLMarker
    mlm = pd.read_csv(PROJECT_ROOT / "notebooks" / "run_meta_mlmarker.tsv", sep="\t")
    print(f"  MLMarker rows: {len(mlm)}")

    # Agent (project-level)
    agent = pd.read_csv(PROJECT_ROOT / "agentic_metadata" / "results" / "agent_metadata.tsv", sep="\t")
    print(f"  Agent rows: {len(agent)}")

    # Sample name mapping (for SDRF fraction → sample)
    mapping = pd.read_csv(PROJECT_ROOT / "sample_name_mapping.tsv", sep="\t")
    print(f"  Sample mapping rows: {len(mapping)}")

    return target, sdrf, name, mlm, agent, mapping


# ---------------------------------------------------------------------------
# Step 2: Prepare SDRF with fraction-to-sample mapping
# ---------------------------------------------------------------------------

def prepare_sdrf(sdrf, mapping, target):
    print("\nPreparing SDRF...")
    mapped_pxds = mapping["pxd"].unique()
    target_keys = set(zip(target["pxd"], target["run"]))

    # Mapped PXDs: strip fraction suffix, join via mapping
    sdrf_mapped = sdrf[sdrf["pxd"].isin(mapped_pxds)].copy()
    sdrf_mapped["sample_prefix"] = sdrf_mapped["run"].apply(
        lambda r: re.sub(r"[_\-][fF]\d{1,3}$", "", r)
    )
    # Deduplicate per (pxd, sample_prefix) — fractions share metadata
    sdrf_dedup = sdrf_mapped.drop_duplicates(subset=["pxd", "sample_prefix"])

    # Join with mapping to get generic_name
    sdrf_with_generic = sdrf_dedup.merge(
        mapping[["pxd", "descriptive_name", "generic_name"]],
        left_on=["pxd", "sample_prefix"],
        right_on=["pxd", "descriptive_name"],
        how="inner",
    )
    sdrf_with_generic["run"] = sdrf_with_generic["generic_name"]
    sdrf_with_generic = sdrf_with_generic.drop(
        columns=["sample_prefix", "descriptive_name", "generic_name"]
    )
    print(f"  Mapped SDRF (PXD000561/PXD020192): {len(sdrf_with_generic)} rows")

    # Direct-match PXDs
    sdrf_direct = sdrf[~sdrf["pxd"].isin(mapped_pxds)].copy()
    sdrf_direct = sdrf_direct[
        sdrf_direct.apply(lambda r: (r["pxd"], r["run"]) in target_keys, axis=1)
    ]
    print(f"  Direct SDRF matches: {len(sdrf_direct)} rows")

    # Combine
    sdrf_out = pd.concat([sdrf_with_generic, sdrf_direct], ignore_index=True)
    # Drop non-output columns
    sdrf_out = sdrf_out.drop(
        columns=["fraction_identifier", "fractionation"],
        errors="ignore",
    )
    # Rename SDRF fractionation if present — it's a boolean, not the same as the method field
    # Actually, SDRF has 'fractionation' as True/False. Keep it but note it's different from agent format.
    # Re-read: the SDRF TSV has fractionation column. Let me check what we dropped.
    # We dropped it above. For SDRF, fractionation info is in fraction_identifier.
    # The agent fractionation field is "True; method". SDRF doesn't have the method.
    # So we don't fill fractionation from SDRF. Remove from SDRF_FIELDS.

    sdrf_out = sdrf_out.drop_duplicates(subset=["pxd", "run"], keep="first")
    print(f"  Total usable SDRF: {len(sdrf_out)} rows")
    return sdrf_out


# ---------------------------------------------------------------------------
# Step 3: Normalize MLMarker
# ---------------------------------------------------------------------------

def prepare_mlmarker(mlm):
    print("\nPreparing MLMarker...")
    mlm = mlm.copy()
    mlm["tissue"] = mlm["tissue"].str.lower()
    mlm = mlm.drop(columns=["confidence"], errors="ignore")
    print(f"  MLMarker: {len(mlm)} rows, {mlm['tissue'].nunique()} unique tissues")
    return mlm


# ---------------------------------------------------------------------------
# Step 4: Prepare agent metadata
# ---------------------------------------------------------------------------

def prepare_agent(agent):
    print("\nPreparing agent metadata...")
    # Drop score columns and group_name
    score_cols = [c for c in agent.columns if c.endswith("_score")]
    agent_clean = agent.drop(columns=score_cols + ["group_name"], errors="ignore")

    # Count setups per PXD
    setup_counts = agent_clean.groupby("pxd").size()
    single_pxds = set(setup_counts[setup_counts == 1].index)
    multi_pxds = set(setup_counts[setup_counts > 1].index)
    print(f"  Single-setup PXDs: {len(single_pxds)}")
    print(f"  Multi-setup PXDs: {len(multi_pxds)}")

    return agent_clean, single_pxds, multi_pxds


# ---------------------------------------------------------------------------
# Step 5: Core fill logic
# ---------------------------------------------------------------------------

def fill_from_source(target, source, source_label, fields):
    """Fill target fields from source, tracking evidence and disagreements."""
    # Merge source onto target
    src_cols = ["pxd", "run"] + [f for f in fields if f in source.columns]
    source_sub = source[src_cols].drop_duplicates(subset=["pxd", "run"], keep="first")
    merged = target[["pxd", "run"]].merge(source_sub, on=["pxd", "run"], how="left")

    filled = 0
    agreed = 0
    disagreed = 0

    for field in fields:
        if field not in source.columns:
            continue

        src_col = f"_src_{field}"
        merged_vals = merged[field].rename(src_col)

        for idx in target.index:
            src_val = merged_vals.iloc[idx] if idx < len(merged_vals) else None
            if pd.isna(src_val) or str(src_val).strip() == "":
                continue

            existing = target.at[idx, field]
            ev_col = f"{field}_evidence"

            if pd.isna(existing) or str(existing).strip() == "":
                # Empty → fill
                target.at[idx, field] = src_val
                target.at[idx, ev_col] = _append_evidence(
                    target.at[idx, ev_col], source_label, True
                )
                filled += 1
            else:
                # Both have values → check agreement
                agrees = _values_agree(existing, src_val)
                target.at[idx, ev_col] = _append_evidence(
                    target.at[idx, ev_col], source_label, agrees
                )
                if agrees:
                    agreed += 1
                else:
                    disagreed += 1

    print(f"    {source_label}: filled={filled}, agreed={agreed}, disagreed={disagreed}")


def fill_from_source_vectorized(target, source, source_label, fields):
    """Vectorized version of fill_from_source for better performance."""
    available_fields = [f for f in fields if f in source.columns]
    if not available_fields:
        return

    source_sub = source[["pxd", "run"] + available_fields].drop_duplicates(
        subset=["pxd", "run"], keep="first"
    )

    # Merge to align source values with target rows
    merged = target[["pxd", "run"]].merge(
        source_sub, on=["pxd", "run"], how="left", suffixes=("", "_src")
    )

    filled_total = 0
    agreed_total = 0
    disagreed_total = 0

    for field in available_fields:
        src_vals = merged[field].values if field in merged.columns else None
        if src_vals is None:
            continue

        ev_col = f"{field}_evidence"
        tgt_vals = target[field].values
        tgt_ev = target[ev_col].values

        new_vals = tgt_vals.copy()
        new_ev = tgt_ev.copy()

        for i in range(len(target)):
            src_v = src_vals[i]
            if pd.isna(src_v) or str(src_v).strip() == "":
                continue

            existing = tgt_vals[i]
            if pd.isna(existing) or str(existing).strip() == "":
                # Fill empty
                new_vals[i] = src_v
                new_ev[i] = _append_evidence(tgt_ev[i], source_label, True)
                filled_total += 1
            else:
                # Compare
                agrees = _values_agree(existing, src_v)
                new_ev[i] = _append_evidence(tgt_ev[i], source_label, agrees)
                if agrees:
                    agreed_total += 1
                else:
                    disagreed_total += 1

        target[field] = new_vals
        target[ev_col] = new_ev

    print(f"    {source_label}: filled={filled_total}, agreed={agreed_total}, disagreed={disagreed_total}")


# ---------------------------------------------------------------------------
# Step 6: Agent multi-setup resolution
# ---------------------------------------------------------------------------

def resolve_agent_for_target(target, agent_clean, single_pxds, multi_pxds):
    """Resolve agent metadata to run-level, handling multi-setup PXDs."""
    print("\nResolving agent metadata...")

    # For single-setup PXDs: broadcast to all runs
    agent_single = agent_clean[agent_clean["pxd"].isin(single_pxds)].copy()
    # Each run in a single-setup PXD gets that PXD's metadata
    single_resolved = target[["pxd", "run"]].merge(
        agent_single, on="pxd", how="inner"
    )
    print(f"  Single-setup: {len(single_resolved)} run-level rows")

    # For multi-setup PXDs: match based on already-filled fields
    multi_rows = []
    multi_agent = agent_clean[agent_clean["pxd"].isin(multi_pxds)]
    multi_runs = target[target["pxd"].isin(multi_pxds)]

    matched = 0
    merged_count = 0

    for pxd in multi_pxds:
        setups = multi_agent[multi_agent["pxd"] == pxd]
        pxd_runs = multi_runs[multi_runs["pxd"] == pxd]

        for _, run_row in pxd_runs.iterrows():
            # Score each setup
            scores = []
            for _, setup in setups.iterrows():
                score = 0
                for f in MATCH_FIELDS:
                    run_val = _norm(run_row.get(f))
                    if run_val is None:
                        continue
                    setup_set = _norm_set(setup.get(f))
                    if run_val in setup_set:
                        score += 1
                scores.append(score)

            max_score = max(scores) if scores else 0
            if max_score > 0 and scores.count(max_score) == 1:
                # Unique best match
                best = setups.iloc[scores.index(max_score)].copy()
                best["run"] = run_row["run"]
                multi_rows.append(best)
                matched += 1
            else:
                # Tie or no match: merge all setups
                merged_row = {"pxd": pxd, "run": run_row["run"]}
                for f in VALUE_FIELDS:
                    if f not in setups.columns:
                        continue
                    # Collect all individual values across setups, split multi-values, deduplicate
                    all_parts = []
                    for v in setups[f].dropna():
                        for part in str(v).split(";"):
                            p = part.strip()
                            if p:
                                all_parts.append(p)
                    unique_vals = list(dict.fromkeys(all_parts))  # dedupe, preserve order
                    if len(unique_vals) == 1:
                        merged_row[f] = unique_vals[0]
                    elif len(unique_vals) > 1:
                        merged_row[f] = "; ".join(unique_vals)
                    else:
                        merged_row[f] = np.nan
                multi_rows.append(pd.Series(merged_row))
                merged_count += 1

    multi_resolved = pd.DataFrame(multi_rows) if multi_rows else pd.DataFrame()
    print(f"  Multi-setup: {matched} matched, {merged_count} merged")

    # Combine
    agent_resolved = pd.concat(
        [single_resolved, multi_resolved], ignore_index=True
    )
    print(f"  Total agent resolved: {len(agent_resolved)} rows")
    return agent_resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target, sdrf_raw, name, mlm_raw, agent_raw, mapping = load_sources()

    # Initialize output columns as object dtype (strings + NaN)
    for f in VALUE_FIELDS:
        target[f] = pd.array([pd.NA] * len(target), dtype="object")
        target[f"{f}_evidence"] = pd.array([pd.NA] * len(target), dtype="object")

    # Prepare sources
    sdrf = prepare_sdrf(sdrf_raw, mapping, target)
    mlm = prepare_mlmarker(mlm_raw)
    agent_clean, single_pxds, multi_pxds = prepare_agent(agent_raw)

    # Fill in priority order
    # Note: we removed fractionation from SDRF fill since SDRF only has True/False,
    # not the method info. Agent has the full "True; method" format.
    sdrf_fill_fields = [f for f in SDRF_FIELDS if f != "fractionation"]
    # Fill in priority order: sdrf > name > agent > MLmarker
    # Higher priority sources fill first, so their values are kept on disagreement.
    print("\nFilling from sources...")
    fill_from_source_vectorized(target, sdrf, "sdrf", sdrf_fill_fields)
    fill_from_source_vectorized(target, name, "name", NAME_FIELDS)

    # Agent: resolve multi-setup then fill (before MLMarker — higher priority)
    agent_resolved = resolve_agent_for_target(
        target, agent_clean, single_pxds, multi_pxds
    )
    fill_from_source_vectorized(target, agent_resolved, "agent", VALUE_FIELDS)

    # MLMarker last (lowest priority)
    fill_from_source_vectorized(target, mlm, "MLM", MLM_FIELDS)

    # Normalize multi-value fields: deduplicate and sort consistently
    print("\nNormalizing multi-value fields...")
    for f in VALUE_FIELDS:
        target[f] = target[f].apply(_normalize_multivalue)

    # Export
    out_cols = ["pxd", "run"]
    for f in VALUE_FIELDS:
        out_cols.extend([f, f"{f}_evidence"])
    output = target[out_cols]

    out_path = PROJECT_ROOT / "run_metadata_combined.tsv"
    output.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total rows: {len(output)}")

    # Coverage summary
    print("\nCoverage summary:")
    for f in VALUE_FIELDS:
        n = output[f].notna().sum()
        pct = n / len(output) * 100
        print(f"  {f:25s}: {n:6d} / {len(output)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
