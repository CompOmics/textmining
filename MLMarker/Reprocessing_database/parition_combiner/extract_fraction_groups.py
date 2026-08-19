"""
Extract fraction groups from run names within each PXD.

Logic: strip the trailing fraction identifier from run names to get a "sample base".
Runs with the same (pxd, sample_base) are fractions of the same sample.

Output: TSV with columns: pxd, run, sample_base, fraction_id, group_size
for manual review before use as training data.
"""

import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fraction suffix patterns, ordered from most specific to least
FRACTION_PATTERNS = [
    # Explicit fraction markers: _fr15, _F14, _frac01, _FR3
    (r"^(.+?)[_\-](?:[fF](?:r(?:ac(?:tion)?)?)?|FR)(\d{1,3})(?:[_\-]?\d)?$", "explicit_f"),
]


def extract_fraction_groups(runs_df):
    """
    Given a DataFrame with columns ['pxd', 'run'], group runs into fraction groups.
    Returns DataFrame with: pxd, run, sample_base, fraction_id, pattern_type, group_size
    """
    results = []

    for pxd, group in runs_df.groupby("pxd"):
        run_names = group["run"].tolist()

        # Try each pattern on all runs
        grouped = {}  # sample_base -> [(run, frac_id, pattern)]
        ungrouped = []

        for run in run_names:
            matched = False
            for pattern, ptype in FRACTION_PATTERNS:
                m = re.match(pattern, run)
                if m:
                    base = m.group(1)
                    frac_id = m.group(2)
                    key = (pxd, base)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append((run, frac_id, ptype))
                    matched = True
                    break
            if not matched:
                ungrouped.append(run)

        # Only keep groups with >= 2 fractions (single runs aren't fraction groups)
        for (pxd_key, base), members in grouped.items():
            if len(members) >= 2:
                for run, frac_id, ptype in members:
                    results.append({
                        "pxd": pxd_key,
                        "run": run,
                        "sample_base": base,
                        "fraction_id": frac_id,
                        "pattern_type": ptype,
                        "group_size": len(members),
                    })

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract fraction groups from run names")
    parser.add_argument("--quant", required=True, help="Path to quant parquet (pxd, run columns)")
    parser.add_argument("--metadata", required=True, help="Path to run_metadata_combined.tsv (for fractionation flag)")
    parser.add_argument("--output", default=None, help="Output path (default: parition_combiner/fraction_groups.tsv)")
    args = parser.parse_args()

    quant = pd.read_parquet(args.quant, columns=["pxd", "run"])
    print(f"Total runs: {len(quant)}, PXDs: {quant['pxd'].nunique()}")

    # Filter to runs/PXDs where fractionation metadata says True
    combined = pd.read_csv(args.metadata, usecols=["pxd", "run", "fractionation"],
                           sep="\t", low_memory=False)
    # A run is fractionated if its fractionation field contains "True" (e.g., "True", "True; SCX")
    frac_runs = combined[combined["fractionation"].astype(str).str.contains("True", case=False, na=False)]
    frac_pxds = set(frac_runs["pxd"].unique())
    frac_run_keys = set(zip(frac_runs["pxd"], frac_runs["run"]))

    # Keep runs that are either:
    # 1. Individually flagged as fractionated, OR
    # 2. In a PXD where ANY run is fractionated (fractions share the same project)
    quant_frac = quant[quant["pxd"].isin(frac_pxds)]
    print(f"Runs in fractionated PXDs: {len(quant_frac)} / {len(quant)} ({len(quant_frac)/len(quant):.1%})")
    print(f"Fractionated PXDs: {len(frac_pxds)}")

    fraction_groups = extract_fraction_groups(quant_frac)

    # Summary stats
    n_grouped = fraction_groups["run"].nunique()
    n_samples = fraction_groups.drop_duplicates(["pxd", "sample_base"]).shape[0]
    n_pxds = fraction_groups["pxd"].nunique()

    print(f"\nFraction groups found:")
    print(f"  Runs in groups: {n_grouped} / {len(quant)} ({n_grouped/len(quant):.1%})")
    print(f"  Unique samples: {n_samples}")
    print(f"  PXDs with fractions: {n_pxds}")

    print(f"\nPattern breakdown:")
    print(fraction_groups.groupby("pattern_type").agg(
        n_runs=("run", "count"),
        n_samples=("sample_base", "nunique"),
    ).to_string())

    print(f"\nGroup size distribution:")
    sizes = fraction_groups.drop_duplicates(["pxd", "sample_base"])["group_size"]
    print(f"  min={sizes.min()}, median={sizes.median():.0f}, max={sizes.max()}")

    # Show examples per pattern type
    for ptype in fraction_groups["pattern_type"].unique():
        sub = fraction_groups[fraction_groups["pattern_type"] == ptype]
        example_pxd = sub["pxd"].value_counts().index[0]
        example = sub[sub["pxd"] == example_pxd].head(6)
        print(f"\n--- Example: {ptype} ({example_pxd}) ---")
        print(example[["pxd", "run", "sample_base", "fraction_id", "group_size"]].to_string(index=False))

    # Save for manual review
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "parition_combiner" / "fraction_groups.tsv"
    fraction_groups.sort_values(["pxd", "sample_base", "fraction_id"]).to_csv(
        out_path, sep="\t", index=False
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
