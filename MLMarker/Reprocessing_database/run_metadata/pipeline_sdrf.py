"""Extract run-level metadata from PRIDE SDRF files."""

import logging
import os
import re
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

# SDRF column name -> our metadata field
SDRF_COL_MAP = {
    "characteristics[organism]": "organism",
    "characteristics[organism part]": "tissue",
    "characteristics[disease]": "disease",
    "characteristics[cell line]": "cell_line",
    "comment[instrument]": "instrument",
    "comment[dissociation method]": "fragmentation",
    "comment[collision energy]": "collision_energy",
    "comment[cleavage agent details]": "enzymes",
    "comment[modification parameters]": "modifications",
    "comment[label]": "labeling",
    "characteristics[enrichment process]": "enrichment",
}


def _parse_sdrf_value(val):
    """Extract the human-readable name from an SDRF ontology annotation.
    E.g. 'NT=Homo sapiens;AC=NCBITaxon:9606' -> 'Homo sapiens'
    """
    if pd.isna(val):
        return None
    val = str(val).strip()
    if not val or val.lower() in ("not applicable", "not available", "none"):
        return None
    # Parse NT= format
    if "NT=" in val or "AC=" in val:
        for part in val.split(";"):
            part = part.strip()
            if part.startswith("NT="):
                return part[3:].strip()
        return None
    return val


def _stem_filename(name: str) -> str:
    """Strip common MS file extensions for run name matching."""
    return re.sub(r"\.(raw|mzML|d|wiff|mzXML)(\.gz)?$", "", name, flags=re.IGNORECASE)


def _normalize_sdrf_columns(sdrf):
    """Build case-insensitive column lookup."""
    return {c.lower(): c for c in sdrf.columns}


def _find_duplicate_columns(sdrf, sdrf_key):
    """Find columns matching sdrf_key, including pandas duplicate suffixes (.1, .2)."""
    col_lookup = _normalize_sdrf_columns(sdrf)
    matches = [c for c in sdrf.columns if c.lower() == sdrf_key or
               re.match(rf"^{re.escape(sdrf_key)}\.\d+$", c.lower())]
    return matches


def sdrf_to_run_metadata(sdrf: pd.DataFrame, pxd_id: str) -> pd.DataFrame:
    """Convert an SDRF dataframe to run-level metadata rows."""
    col_lookup = _normalize_sdrf_columns(sdrf)
    data_file_col = col_lookup.get("comment[data file]")
    if data_file_col is None:
        log.warning("%s: SDRF has no 'comment[data file]' column", pxd_id)
        return pd.DataFrame()

    # Check for fraction identifier
    frac_col = col_lookup.get("comment[fraction identifier]")

    rows = []
    for _, srow in sdrf.iterrows():
        run_name = _stem_filename(str(srow[data_file_col]))
        row = {"pxd": pxd_id, "run": run_name}

        for sdrf_key, meta_field in SDRF_COL_MAP.items():
            matching_cols = _find_duplicate_columns(sdrf, sdrf_key)
            if not matching_cols:
                row[meta_field] = None
                continue
            parsed_vals = []
            for col in matching_cols:
                v = _parse_sdrf_value(srow[col])
                if v and v not in parsed_vals:
                    parsed_vals.append(v)
            row[meta_field] = "; ".join(parsed_vals) if parsed_vals else None

        # Fraction info
        if frac_col:
            row["fraction_identifier"] = srow[frac_col] if pd.notna(srow[frac_col]) else None
            row["fractionation"] = True
        else:
            row["fraction_identifier"] = None
            row["fractionation"] = False

        rows.append(row)

    return pd.DataFrame(rows)


def _normalize_sdrf_values(result: pd.DataFrame, norm_cfg: dict) -> pd.DataFrame:
    """Normalize SDRF metadata values using SapBERT + exact cell line matching.

    Uses the same normalization pipeline as the agentic_metadata extraction.
    """
    from agentic_metadata.normalization.config import NormalizationConfig
    from agentic_metadata.normalization.normalizer import TermNormalizer
    from agentic_metadata.normalization.normalize_extraction import CellLineExactMatcher

    config = NormalizationConfig.from_dict(norm_cfg)
    normalizer = TermNormalizer(config)
    normalizer.load_all_ontologies()

    # Cell line exact matcher
    cell_line_path = config.get_ontology_path("cell_lines")
    cell_line_matcher = CellLineExactMatcher(cell_line_path) if cell_line_path and cell_line_path.exists() else None

    # Post-normalization map — try agentic config first, then standalone YAML
    from agentic_metadata.normalization.normalize_extraction import _load_post_normalization_map, _apply_post_normalization
    agentic_config_path = Path(config.ontology_dir).parent / "config.yaml"
    agentic_config = None
    if agentic_config_path.exists():
        import yaml
        agentic_config = yaml.safe_load(agentic_config_path.read_text(encoding="utf-8"))
    post_map = _load_post_normalization_map(config.ontology_dir, config=agentic_config)

    # Field -> entity_type mapping (same as in agentic_metadata)
    FIELD_ENTITY_MAP = {
        "organism": "organism", "tissue": "tissue", "disease": "disease",
        "instrument": "instrument", "fragmentation": "fragmentation",
        "labeling": "labeling", "modifications": "modifications",
        "enzymes": "enzymes", "enrichment": "enrichment",
    }

    # Fields that should NOT be normalized (keep raw)
    SKIP_FIELDS = {"collision_energy", "fraction_identifier", "fractionation"}

    for field in result.columns:
        if field in ("pxd", "run") or field in SKIP_FIELDS:
            continue

        entity_type = FIELD_ENTITY_MAP.get(field)
        if entity_type is None and field == "cell_line":
            # Exact match for cell lines
            if cell_line_matcher:
                def _match_cell_line(val):
                    if pd.isna(val):
                        return val
                    parts = [p.strip() for p in str(val).split(";")]
                    normalized = []
                    for p in parts:
                        matched = cell_line_matcher.match(p)
                        if matched:
                            normalized.append(matched)
                        # else: drop — not in cell line vocabulary
                    return "; ".join(normalized) if normalized else None
                result[field] = result[field].apply(_match_cell_line)
            continue

        if entity_type is None:
            continue

        # SapBERT normalization — unmapped values are dropped (set to None)
        def _normalize_field(val, etype=entity_type):
            if pd.isna(val):
                return val
            parts = [p.strip() for p in str(val).split(";")]
            normalized = []
            for p in parts:
                if not p:
                    continue
                res = normalizer.normalize(p, entity_type=etype)
                if res.is_normalized:
                    # Apply post-normalization override
                    item = {"ontology_name": res.ontology_name}
                    _apply_post_normalization(item, etype, post_map)
                    normalized.append(item["ontology_name"])
                # else: drop — value doesn't match controlled vocabulary
            return "; ".join(normalized) if normalized else None

        result[field] = result[field].apply(_normalize_field)

    log.info("SDRF values normalized via SapBERT")
    return result


def run(quant_path: Path, output_path: Path, sdrf_cache_dir: Path,
        sample_name_mapping_path: Path = None, normalization_cfg: dict = None):
    """Run the SDRF extraction pipeline.

    Args:
        quant_path: Path to pride_quant.parquet
        output_path: Where to write the output TSV
        sdrf_cache_dir: Directory with cached SDRF .tsv files
        sample_name_mapping_path: Path to sample_name_mapping.tsv (for PXD000561 etc.)
        normalization_cfg: Normalization config dict (ontology_dir, cache_dir, etc.)
    """
    log.info("Loading runs from %s", quant_path)
    all_runs = pd.read_parquet(quant_path, columns=["pxd", "run"])
    all_runs = all_runs.drop_duplicates(subset=["pxd", "run"]).reset_index(drop=True)
    target_keys = set(zip(all_runs["pxd"], all_runs["run"]))

    # Load sample name mapping if available
    mapping = None
    mapped_pxds = set()
    if sample_name_mapping_path and sample_name_mapping_path.exists():
        mapping = pd.read_csv(sample_name_mapping_path, sep="\t")
        mapped_pxds = set(mapping["pxd"].unique())
        log.info("Sample name mapping: %d entries for PXDs %s", len(mapping), mapped_pxds)

    # Process all cached SDRF files
    all_meta = []
    sdrf_files = sorted(sdrf_cache_dir.glob("*.sdrf.tsv"))
    log.info("Processing %d SDRF files from %s", len(sdrf_files), sdrf_cache_dir)

    for sdrf_file in sdrf_files:
        pxd_id = sdrf_file.stem.replace(".sdrf", "")
        try:
            sdrf = pd.read_csv(sdrf_file, sep="\t")
        except Exception as e:
            log.warning("%s: failed to read SDRF: %s", pxd_id, e)
            continue

        meta = sdrf_to_run_metadata(sdrf, pxd_id)
        if meta.empty:
            continue

        # For mapped PXDs: match SDRF run names to db_run_name via raw_files
        if pxd_id in mapped_pxds and mapping is not None:
            pxd_map = mapping[mapping["pxd"] == pxd_id]
            # Build lookup: raw_file -> db_run_name
            raw_to_db = {}
            for _, mrow in pxd_map.iterrows():
                for raw in str(mrow["raw_files"]).split(";"):
                    raw = raw.strip()
                    if raw:
                        raw_to_db[raw] = mrow["db_run_name"]

            # Map each SDRF run to its db_run_name
            meta["db_run"] = meta["run"].map(raw_to_db)
            meta_mapped = meta[meta["db_run"].notna()].copy()
            meta_mapped["run"] = meta_mapped["db_run"]
            meta_mapped = meta_mapped.drop(columns=["db_run"])
            # Deduplicate: fractions of the same sample share metadata
            meta_mapped = meta_mapped.drop_duplicates(subset=["pxd", "run"], keep="first")
            all_meta.append(meta_mapped)
        else:
            # Direct match: only keep runs that exist in quant data
            meta = meta[meta.apply(lambda r: (r["pxd"], r["run"]) in target_keys, axis=1)]
            all_meta.append(meta)

    if not all_meta:
        log.warning("No SDRF metadata extracted")
        result = pd.DataFrame(columns=["pxd", "run"])
    else:
        result = pd.concat(all_meta, ignore_index=True)
        result = result.drop_duplicates(subset=["pxd", "run"], keep="first")

    # Normalize values if config provided
    if normalization_cfg and len(result) > 0:
        result = _normalize_sdrf_values(result, normalization_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    log.info("SDRF metadata: %d runs from %d PXDs -> %s", len(result), result["pxd"].nunique(), output_path)
