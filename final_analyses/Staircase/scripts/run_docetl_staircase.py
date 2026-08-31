#!/usr/bin/env python3
"""
DocETL Pipeline Runner — All Agents
=====================================
Runs DocETL extraction pipelines for all three agents (BiologicalAgent,
TechnicalAgent, ExperimentalDesignAgent) and optionally computes
ValidationAgent confidence scores — producing output in the same JSON
format as the existing ``BaseExtractor`` pipeline.

Post-processing steps (run after extraction):
- NormalizationAgent: maps extracted terms to ontology IDs
- IntegrationAgent: enriches with PRIDE/METI data from final_files

Usage
-----
::

    # All agents on a directory of manuscripts
    python docetl_pipeline/run_docetl.py \\
        --input  docs/ \\
        --output framework_output/docetl/ \\
        --config config.yaml

    # Full pipeline including normalization + integration
    python docetl_pipeline/run_docetl.py \\
        --input  docs/ \\
        --output framework_output/docetl/ \\
        --meti-dir benchmark_data/Technical_pipeline_outputs_train_test/final_files/

    # Single agent, single file, extraction only
    python docetl_pipeline/run_docetl.py \\
        --input  docs/PXD001234.txt \\
        --output framework_output/docetl/ \\
        --agents biological \\
        --no-confidence \\
        --no-normalize \\
        --no-integrate

Environment
-----------
Set the API key for the configured provider before running:

  - OpenAI/compat (default) : ``OPENAI_API_KEY``
  - Anthropic (Claude)      : ``ANTHROPIC_API_KEY``
  - Gemini                  : ``GEMINI_API_KEY``

The key can also be placed in the ``llm.api_key_env_var`` field of the config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

# Suppress LiteLLM's noisy stdout feedback/info messages
import litellm
litellm.suppress_debug_info = True
litellm.verbose = False
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)

# ── Project root on path for internal imports ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3] / "framework"
sys.path.insert(0, str(PROJECT_ROOT))

# Module-level singleton for the cross-field checker (lazy init)
_cross_field_checker = None
# Module-level singleton for the negation detector (lazy init)
_negation_detector = None
# Module-level singleton for the numeric mismatch detector (lazy init)
_numeric_mismatch_detector = None

PIPELINE_DIR = Path(__file__).parent

# Agent configuration: name → (yaml filename, output subdir key)
AGENTS = {
    "biological": (
        "pipeline_biological.yaml",
        "BiologicalAgent",
        "_biological",          # output filename suffix
    ),
    "technical": (
        "pipeline_technical.yaml",
        "TechnicalAgent",
        "_technical",
    ),
    "experimental": (
        "pipeline_experimental.yaml",
        "ExperimentalDesignAgent",
        "_experimental",
    ),
}

SKIP_KEYS = {"id", "text"}   # DocETL passes these through; strip before writing
_truncation_log_lock = threading.Lock()


def _install_truncation_logger() -> None:
    """Add machine-readable PXD logging around DocETL prompt truncation."""
    log_path = os.getenv("DOCETL_TRUNCATION_LOG")
    if not log_path:
        return

    import re
    import docetl.operations.utils.api as docetl_api

    if getattr(docetl_api.truncate_messages, "_hamlet_truncation_logger", False):
        return

    original = docetl_api.truncate_messages
    id_re = re.compile(r"DOCUMENT ID:\s*(PXD\d+)")
    marker_re = re.compile(r"\[(\d+) tokens truncated\]")

    def logged_truncate(messages, model, from_agent=False):
        document_id = "unknown"
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str) and (match := id_re.search(content)):
                document_id = match.group(1)
                break

        result = original(messages, model, from_agent)
        removed = None
        for message in result:
            content = message.get("content", "")
            if isinstance(content, str) and (match := marker_re.search(content)):
                removed = int(match.group(1))
                break

        if removed is not None:
            event = f"{document_id}\t{model}\t{removed}\t{int(from_agent)}\n"
            with _truncation_log_lock:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(event)
            print(
                f"DOCETL_TRUNCATION pxd={document_id} model={model} "
                f"removed_tokens={removed} from_agent={int(from_agent)}",
                flush=True,
            )
        return result

    logged_truncate._hamlet_truncation_logger = True
    docetl_api.truncate_messages = logged_truncate


def _require_complete_qc() -> None:
    """Fail before extraction if any mandatory hallucination-QC stage is unavailable."""
    errors: list[str] = []

    # Cross-field QC needs these source ontologies. Its derived JSON caches are
    # optional because the checker can rebuild them from the source files.
    ontology_dir = PROJECT_ROOT / "ontologies"
    required_ontologies = ("clo.owl", "bto.obo", "doid.obo", "cl.obo")
    missing = [name for name in required_ontologies if not (ontology_dir / name).is_file()]
    if missing:
        errors.append("cross-field ontology resources missing: " + ", ".join(missing))
    else:
        try:
            from validation.cross_field_checker import CrossFieldConsistencyChecker
            CrossFieldConsistencyChecker()
        except Exception as exc:
            errors.append(f"cross-field checker unavailable: {exc}")

    try:
        from validation.negation_detector import _load_nlp
        if _load_nlp() is None:
            errors.append("negation detector unavailable (spaCy/negspacy/en_core_web_sm)")
    except Exception as exc:
        errors.append(f"negation detector unavailable: {exc}")

    try:
        from validation.numeric_mismatch_detector import NumericMismatchDetector
        NumericMismatchDetector()
    except Exception as exc:
        errors.append(f"numeric mismatch detector unavailable: {exc}")

    if errors:
        detail = "\n  - ".join(errors)
        raise RuntimeError(f"Mandatory QC preflight failed:\n  - {detail}")

    print("  QC preflight : PASS (cross-field, negation, numeric mismatch)")


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: Optional[str]) -> dict:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


# Map provider → litellm env var name
_PROVIDER_KEY_ENV = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# litellm model prefix per provider
_PROVIDER_PREFIX = {
    "anthropic":  "anthropic",
    "gemini":     "gemini",
    "openai":     "openai",
    "openrouter": "openrouter",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _apply_env(cfg: dict) -> None:
    """Export the correct API key env var(s) for the configured provider.

    DocETL delegates to litellm, which reads provider-specific env vars:
      - OpenAI/compat  → OPENAI_API_KEY  (+ optionally OPENAI_BASE_URL)
      - Anthropic      → ANTHROPIC_API_KEY
      - Gemini         → GEMINI_API_KEY
      - OpenRouter     → OPENROUTER_API_KEY
    """
    llm = cfg.get("llm", {})

    # Resolve API key from config or env var
    api_key = llm.get("api_key") or os.getenv(
        llm.get("api_key_env_var", "LLM_API_KEY"), ""
    )

    # Auto-detect provider
    model = llm.get("model", "")
    if llm.get("provider"):
        provider = llm["provider"]
    elif model.startswith("claude"):
        provider = "anthropic"
    elif model.startswith("gemini"):
        provider = "gemini"
    elif llm.get("base_url", "").startswith("https://openrouter.ai"):
        provider = "openrouter"
    else:
        provider = "openai"

    # Set the provider-specific key
    key_env = _PROVIDER_KEY_ENV.get(provider, "OPENAI_API_KEY")
    os.environ[key_env] = api_key or os.getenv(key_env, "")

    if provider == "openrouter":
        # litellm routes openrouter/* models via OPENROUTER_API_KEY
        # No base_url needed — litellm handles it natively
        pass

    if provider == "openai":
        # litellm rejects empty string for OpenAI-compat endpoints
        if not os.environ["OPENAI_API_KEY"]:
            os.environ["OPENAI_API_KEY"] = "dummy-key"
        base_url = llm.get("base_url", "")
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # Store resolved provider for use in _run_pipeline
    cfg.setdefault("_resolved", {})["provider"] = provider



# ─────────────────────────────────────────────────────────────────────────────
# Manuscript loading
# ─────────────────────────────────────────────────────────────────────────────

def _collect_manuscripts(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    # Benchmark datasets store each manuscript as PXDxxxxxx/manuscript.txt.
    # Recursive discovery also retains support for directories of flat .txt files.
    return sorted(input_path.rglob("*.txt"))


def _build_records(manuscripts: list[Path]) -> list[dict]:
    records = []
    for path in manuscripts:
        text = path.read_text(encoding="utf-8", errors="replace")
        record_id = path.parent.name if path.name == "manuscript.txt" else path.stem
        records.append({"id": record_id, "text": text})
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DocETL runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(
    records: list[dict],
    yaml_file: Path,
    cfg: dict,
    tmp_dir: Path,
    bypass_cache: bool = False,
    archive_dir: Path | None = None,
) -> list[dict]:
    """Load + execute one DocETL pipeline; return result records."""
    from docetl.runner import DSLRunner
    _install_truncation_logger()

    provider = cfg.get("_resolved", {}).get("provider", "openai")
    litellm_prefix = _PROVIDER_PREFIX.get(provider, "openai")
    model = litellm_prefix + "/" + cfg.get("llm", {}).get("model", "llama-4-scout")

    in_path  = tmp_dir / "input.json"
    out_path = tmp_dir / "output.json"
    cfg_path = tmp_dir / "pipeline.yaml"

    in_path.write_text(json.dumps(records, indent=2))

    with open(yaml_file) as f:
        pipeline_cfg = yaml.safe_load(f)

    pipeline_cfg["default_model"] = model
    pipeline_cfg["datasets"]["manuscripts"]["path"] = str(in_path)
    pipeline_cfg["pipeline"]["output"]["path"] = str(out_path)

    # Read timeout from config (concurrency.request_timeout), fall back to 120s.
    # Always override the YAML value so config.yaml is the single source of truth.
    timeout_secs = cfg.get("concurrency", {}).get("request_timeout", 120)

    for op in pipeline_cfg.get("operations", []):
        if bypass_cache:
            op["bypass_cache"] = True
        # Freeze sampling explicitly. Local model generation configs can specify
        # nonzero vendor defaults (GLM-4.7 specifies 1.0), so relying on an
        # omitted client parameter would violate the staircase protocol.
        op.setdefault("litellm_completion_kwargs", {})["temperature"] = 0
        # Always override timeout — docetl defaults to 120s which is too short for
        # large local models. No retries on timeout: a slow model won't get faster
        # on retry; the timeout itself signals something is wrong.
        op["timeout"] = timeout_secs
        op["max_retries_per_timeout"] = 0

    with open(cfg_path, "w") as f:
        yaml.dump(pipeline_cfg, f)

    max_threads = cfg.get("concurrency", {}).get("max_workers", 1)
    runner = DSLRunner.from_yaml(str(cfg_path), max_threads=max_threads)
    runner.load_run_save()

    with open(out_path) as f:
        results = json.load(f)

    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "input.json").write_bytes(in_path.read_bytes())
        (archive_dir / "docetl_output.json").write_bytes(out_path.read_bytes())
        (archive_dir / "executed_pipeline.yaml").write_bytes(cfg_path.read_bytes())

    # Cleanup temp artefacts
    for p in (in_path, out_path, cfg_path):
        try:
            p.unlink()
        except OSError:
            pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Confidence estimation
# ─────────────────────────────────────────────────────────────────────────────

def _add_hallucination_flags(record: dict) -> dict:
    """
    Append ``_hallucination_flags`` to *record* in-place.

    Combines results from three complementary checks:
    - Cross-field ontology consistency (CLO / DOID / CL / UBERON)
    - Negation detection via NegEx (negspacy)
    - Numeric exact-match check (e.g. 50 mM extracted vs 5 mM in evidence)
    """
    global _cross_field_checker, _negation_detector, _numeric_mismatch_detector
    flags: list[dict] = []

    # ── Cross-field ontology consistency ─────────────────────────────────
    try:
        if _cross_field_checker is None:
            from validation.cross_field_checker import CrossFieldConsistencyChecker
            _cross_field_checker = CrossFieldConsistencyChecker()
        flags.extend(_cross_field_checker.check(record))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("Cross-field check error: %s", exc)

    # ── Negation detection (negspacy / NegEx) ─────────────────────────────
    try:
        if _negation_detector is None:
            from validation.negation_detector import NegationDetector
            _negation_detector = NegationDetector()
        flags.extend(_negation_detector.check(record))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("Negation check error: %s", exc)

    # ── Numeric mismatch (exact-only for concentrations/energies) ────────────
    try:
        if _numeric_mismatch_detector is None:
            from validation.numeric_mismatch_detector import NumericMismatchDetector
            _numeric_mismatch_detector = NumericMismatchDetector()
        flags.extend(_numeric_mismatch_detector.check(record))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("Numeric mismatch check error: %s", exc)

    if flags:
        record["_hallucination_flags"] = flags
    return record


def _add_confidence(
    record: dict,
    text: str,
) -> dict:
    """Attach ValidationAgent confidence metrics to a record in-place."""
    try:
        from validation.validator import ValidationAgent, ValidationMode
        validator = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        # strip input pass-through keys before scoring
        scoreable = {k: v for k, v in record.items() if k not in SKIP_KEYS}
        scored = validator._add_confidence(scoreable, text)
        record["_confidence"] = scored["_confidence"]
    except Exception as exc:
        # Non-fatal — confidence is optional
        record["_confidence"] = {"error": str(exc)}
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def _write_outputs(
    results: list[dict],
    text_lookup: dict[str, str],
    output_dir: Path,
    agent_dir_name: str,
    suffix: str,
    use_confidence: bool,
    model_tag: str = "",
) -> None:
    agent_dir = output_dir / agent_dir_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{model_tag}" if model_tag else ""

    for record in results:
        pxd_id = record.get("id", "unknown")
        payload = {k: v for k, v in record.items() if k not in SKIP_KEYS}

        if use_confidence and pxd_id in text_lookup:
            _add_confidence(payload, text_lookup[pxd_id])

        _add_hallucination_flags(payload)

        out_file = agent_dir / f"{pxd_id}{suffix}{tag}.json"
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Written: {out_file.relative_to(output_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: Normalization + Integration
# ─────────────────────────────────────────────────────────────────────────────

def _load_agent_outputs(agent_dir: Path) -> dict[str, dict]:
    """Load all per-doc JSON files from an agent output directory."""
    results = {}
    for json_file in sorted(agent_dir.glob("*.json")):
        with open(json_file) as f:
            results[json_file.name] = json.load(f)
    return results


def _run_normalization(
    output_dir: Path,
    agents_run: list[tuple[str, str, str]],  # (yaml_name, agent_dir_name, suffix)
) -> dict[str, dict[str, dict]]:
    """
    Run NormalizationAgent over all per-doc extraction outputs.

    Returns:
        {agent_dir_name: {filename: normalized_data}}
    """
    from agents.normalization_agent import NormalizationAgent

    norm_agent = NormalizationAgent()
    all_normalized: dict[str, dict[str, dict]] = {}

    for _, agent_dir_name, suffix in agents_run:
        agent_dir = output_dir / agent_dir_name
        if not agent_dir.exists():
            print(f"  [Normalization] Skipping {agent_dir_name} — dir not found")
            continue

        raw = _load_agent_outputs(agent_dir)
        if not raw:
            continue

        print(f"  Normalizing {len(raw)} docs for {agent_dir_name}...")
        normalized = norm_agent.normalize_batch(raw)

        # Write normalized outputs (with hallucination flags)
        norm_dir = output_dir / "NormalizedAgent" / agent_dir_name
        norm_dir.mkdir(parents=True, exist_ok=True)
        for fname, data in normalized.items():
            _add_hallucination_flags(data)
            out = norm_dir / fname
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
        print(f"  Written: NormalizedAgent/{agent_dir_name}/ ({len(normalized)} docs)")

        all_normalized[agent_dir_name] = normalized

    return all_normalized


def _run_integration(
    output_dir: Path,
    agents_run: list[tuple[str, str, str]],
    meti_dir: Path,
    normalized: dict[str, dict[str, dict]],
) -> None:
    """
    Run IntegrationAgent over normalized (or raw) extraction outputs.

    Uses normalized outputs when available; falls back to raw extraction.
    """
    from agents.integration_agent import IntegrationAgent

    int_agent = IntegrationAgent(str(meti_dir))

    for _, agent_dir_name, suffix in agents_run:
        # Prefer normalized, fall back to raw extraction
        if agent_dir_name in normalized:
            file_results = normalized[agent_dir_name]
            src_label = "normalized"
        else:
            agent_dir = output_dir / agent_dir_name
            if not agent_dir.exists():
                print(f"  [Integration] Skipping {agent_dir_name} — dir not found")
                continue
            file_results = _load_agent_outputs(agent_dir)
            src_label = "raw extraction"

        if not file_results:
            continue

        print(f"  Integrating {len(file_results)} docs for {agent_dir_name} (from {src_label})...")

        int_dir = output_dir / "IntegratedAgent" / agent_dir_name
        int_dir.mkdir(parents=True, exist_ok=True)

        enriched = int_agent.enrich_batch(
            file_results,
            agent_name=agent_dir_name,
            output_dir=str(int_dir),
        )

        for fname, data in enriched.items():
            _add_hallucination_flags(data)
            out = int_dir / fname
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
        print(f"  Written: IntegratedAgent/{agent_dir_name}/ ({len(enriched)} docs)")


# ─────────────────────────────────────────────────────────────────────────────
# Judge merge
# ─────────────────────────────────────────────────────────────────────────────

def _run_judge_merge(
    output_dir: Path,
    agents_run: list[tuple[str, str, str]],
    judge_dir: Path,
) -> None:
    """
    Attach LLM-as-judge verdicts to each field in IntegrationAgent output files.

    For each PXD, loads {judge_dir}/{pxd_id}.json and patches every matching
    field dict with a ``_judge`` sub-key::

        "fragmentation_method": {
            "resolved": "HCD",
            "status": "AGREE",
            "_judge": {
                "verdict": "high",
                "type_mismatch": false,
                "judged_value": "HCD",
                "issue_summary": { "TYPE_CHECK": "...", ... }
            }
        }

    Field matching uses a two-step lookup:
    1. Direct match on the LLM field name (e.g. ``collision_energy``).
    2. Map the LLM field name → golden name via ``LLM_TO_GOLDEN``, then look
       up the golden name in the judge (e.g. ``fractionation_method`` →
       ``fractionation``).
    """
    from core.field_mappings import LLM_TO_GOLDEN

    int_base = output_dir / "IntegratedAgent"
    if not int_base.exists():
        print("  [JudgeMerge] IntegratedAgent dir not found — skipping.")
        return

    patched_total = 0

    for _, agent_dir_name, _ in agents_run:
        agent_dir = int_base / agent_dir_name
        if not agent_dir.exists():
            continue

        for json_file in sorted(agent_dir.glob("*.json")):
            pxd_id = json_file.stem.split("_")[0]
            judge_file = judge_dir / f"{pxd_id}.json"
            if not judge_file.exists():
                continue

            with open(judge_file) as f:
                judge_data = json.load(f)

            # annotation_type → annotation dict
            ann_lookup: dict[str, dict] = {
                ann["annotation_type"]: ann
                for ann in judge_data.get("annotations", [])
            }

            with open(json_file) as f:
                doc = json.load(f)

            changed = False
            for field_name, field_value in doc.items():
                if not isinstance(field_value, dict) or "resolved" not in field_value:
                    continue

                # Step 1: direct match
                annotation = ann_lookup.get(field_name)
                # Step 2: via golden name
                if annotation is None:
                    golden = LLM_TO_GOLDEN.get(field_name)
                    if golden:
                        annotation = ann_lookup.get(golden)

                if annotation is None:
                    continue

                field_value["_judge"] = {
                    "verdict":       annotation.get("verdict"),
                    "type_mismatch": annotation.get("type_mismatch"),
                    "judged_value":  annotation.get("extracted_value"),
                    "issue_summary": annotation.get("issue_summary"),
                }
                changed = True

            if changed:
                with open(json_file, "w") as f:
                    json.dump(doc, f, indent=2)
                patched_total += 1

    print(f"  Patched: {patched_total} files with judge annotations")


# ─────────────────────────────────────────────────────────────────────────────
# Single output merger
# ─────────────────────────────────────────────────────────────────────────────

# Keys that belong to METI / meta — handled separately, not merged as fields
_METI_KEYS  = {"_meti_data", "modification_site_fractions"}
_META_KEYS  = {"_confidence", "_enrichment", "_hallucination_flags",
               "pxd_id", "pipeline_version"}

def _merge_to_single_output(output_dir: Path, agents_run: list[tuple]) -> None:
    """
    Merge IntegratedAgent outputs for all three agents into one JSON per PXD.

    Output: output_dir/SingleOutput/{pxd_id}.json

    Structure
    ---------
    {
      "pxd_id": "PXD001856",
      // All extracted+enriched fields (flat, one entry per field)
      "species":     { "resolved": ..., "confidence": ..., "status": ..., "sources": {...} },
      "instrument":  { ... },
      ...
      // METI technical pipeline data — included once
      "_meti_data":                  { ... },
      "modification_site_fractions": { ... },
      // Per-agent confidence scores
      "_confidence": {
        "BiologicalAgent":         { ... },
        "TechnicalAgent":          { ... },
        "ExperimentalDesignAgent": { ... }
      },
      // Provenance: which agent owns each field
      "_field_provenance": {
        "species":    "BiologicalAgent",
        "instrument": "TechnicalAgent",
        ...
      }
    }
    """
    int_base = output_dir / "IntegratedAgent"
    if not int_base.exists():
        print("  [SingleOutput] IntegratedAgent dir not found — skipping.")
        return

    single_dir = output_dir / "SingleOutput"
    single_dir.mkdir(parents=True, exist_ok=True)

    # Build {pxd_id: {agent_dir_name: Path}} index
    pxd_files: dict[str, dict[str, Path]] = {}
    for _, agent_dir_name, _ in agents_run:
        agent_dir = int_base / agent_dir_name
        if not agent_dir.exists():
            continue
        for f in sorted(agent_dir.glob("*.json")):
            pxd_match = f.stem.split("_")[0]   # e.g. PXD001856
            pxd_files.setdefault(pxd_match, {})[agent_dir_name] = f

    written = 0
    for pxd_id, agent_paths in sorted(pxd_files.items()):
        merged: dict = {"pxd_id": pxd_id}
        field_provenance: dict[str, str] = {}
        per_agent_confidence: dict[str, dict] = {}
        meti_data = None
        mod_site_fractions = None

        for _, agent_dir_name, _ in agents_run:
            path = agent_paths.get(agent_dir_name)
            if not path:
                continue
            doc = json.loads(path.read_text())

            # Capture METI data once (all agents have the same copy)
            if meti_data is None and doc.get("_meti_data"):
                meti_data = doc["_meti_data"]
            if mod_site_fractions is None and doc.get("modification_site_fractions"):
                mod_site_fractions = doc["modification_site_fractions"]

            # Capture per-agent confidence
            if doc.get("_confidence"):
                per_agent_confidence[agent_dir_name] = doc["_confidence"]

            # Merge extracted fields (skip meta and METI keys)
            for key, value in doc.items():
                if key in _META_KEYS or key in _METI_KEYS:
                    continue
                if key not in merged:
                    merged[key] = value
                    field_provenance[key] = agent_dir_name

        # Attach shared/meta sections
        if meti_data is not None:
            merged["_meti_data"] = meti_data
        if mod_site_fractions is not None:
            merged["modification_site_fractions"] = mod_site_fractions
        if per_agent_confidence:
            merged["_confidence"] = per_agent_confidence
        merged["_field_provenance"] = field_provenance

        out_file = single_dir / f"{pxd_id}.json"
        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2)
        written += 1

    print(f"  Written: SingleOutput/ ({written} docs)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DocETL extraction pipelines for all three agents"
    )
    parser.add_argument("--input", "-i", required=True,
                        help=".txt file or directory of .txt manuscript files")
    parser.add_argument("--output", "-o", default="framework_output/docetl",
                        help="Output directory (default: framework_output/docetl/)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config.yaml (default: auto-detect in project root)")
    parser.add_argument(
        "--agents", nargs="+",
        choices=list(AGENTS.keys()),
        default=list(AGENTS.keys()),
        help="Which agents to run (default: all three)",
    )
    parser.add_argument("--no-confidence", action="store_true",
                        help="Skip ValidationAgent confidence scoring")
    parser.add_argument("--bypass-cache", action="store_true",
                        help="Force fresh LLM calls, ignoring DocETL's disk cache")
    parser.add_argument("--meti-dir", default=None,
                        help="Directory containing final_files aggregated_results JSONs "
                             "(default: auto-detect benchmark_data/Technical_pipeline_outputs_train_test/final_files/)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip NormalizationAgent post-processing")
    parser.add_argument("--no-integrate", action="store_true",
                        help="Skip IntegrationAgent post-processing")
    parser.add_argument("--single-output", action="store_true",
                        help="Merge all three IntegratedAgent outputs into one JSON per PXD "
                             "(output_dir/SingleOutput/{pxd_id}.json)")
    parser.add_argument("--model-tag", default="",
                        help="String appended to every output filename, e.g. 'llama' → PXD000312_manuscript_biological_llama.json")
    parser.add_argument("--judge-dir", default=None,
                        help="Directory containing {PXD}.json LLM-as-judge files. "
                             "When set, judge verdicts are merged into IntegratedAgent outputs "
                             "as a _judge key on each field.")
    parser.add_argument("--pipeline-dir", required=True,
                        help="Frozen arm directory containing pipeline_*.yaml")
    parser.add_argument("--base-url", default=None,
                        help="Override llm.base_url without changing the frozen config")
    args = parser.parse_args()

    global PIPELINE_DIR
    PIPELINE_DIR = Path(args.pipeline_dir).resolve()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    truncation_log = os.getenv("DOCETL_TRUNCATION_LOG")
    if truncation_log:
        Path(truncation_log).parent.mkdir(parents=True, exist_ok=True)
        Path(truncation_log).write_text(
            "pxd\tmodel\tremoved_tokens\tfrom_agent\n", encoding="utf-8"
        )

    cfg = _load_config(args.config)
    if args.base_url:
        cfg.setdefault("llm", {})["base_url"] = args.base_url
    _apply_env(cfg)

    # QC is mandatory: do not spend GPU time on an extraction run whose
    # hallucination checks would silently degrade.
    _require_complete_qc()

    manuscripts = _collect_manuscripts(input_path)
    if not manuscripts:
        print(f"No .txt files found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    records      = _build_records(manuscripts)
    text_lookup  = {r["id"]: r["text"] for r in records}
    use_conf     = not args.no_confidence
    bypass_cache = args.bypass_cache
    do_normalize = not args.no_normalize
    do_integrate = not args.no_integrate
    judge_dir    = Path(args.judge_dir) if args.judge_dir else None
    model_name   = cfg.get("llm", {}).get("model", "llama-4-scout")

    # Resolve METI dir
    meti_dir: Optional[Path] = None
    if do_integrate:
        if args.meti_dir:
            meti_dir = Path(args.meti_dir)
        else:
            default_ra = PROJECT_ROOT / "benchmark_data" / "Technical_pipeline_outputs_train_test" / "final_files"
            if default_ra.exists():
                meti_dir = default_ra
        if not meti_dir or not meti_dir.exists():
            print("  [Integration] WARNING: METI dir not found - skipping integration.")
            do_integrate = False

    print(f"\nDocETL Extraction Runner")
    print(f"  Input        : {input_path}  ({len(manuscripts)} files)")
    print(f"  Output       : {output_dir}")
    print(f"  Model        : {model_name}")
    print(f"  Agents       : {', '.join(args.agents)}")
    print(f"  Confidence   : {'yes' if use_conf else 'no'}")
    print(f"  Normalize    : {'yes' if do_normalize else 'no'}")
    print(f"  Integrate    : {'yes' if do_integrate else 'no'}")
    print(f"  Single output: {'yes' if args.single_output else 'no'}")
    if do_integrate:
        print(f"  METI dir     : {meti_dir}")
    print(f"  Bypass cache : {'yes' if bypass_cache else 'no'}")
    print(f"  Judge dir    : {judge_dir if judge_dir else 'none'}\n")

    agents_run = []
    with tempfile.TemporaryDirectory(prefix="docetl_") as tmp:
        tmp_dir = Path(tmp)

        for agent_key in args.agents:
            yaml_name, agent_dir_name, suffix = AGENTS[agent_key]
            yaml_file = PIPELINE_DIR / yaml_name

            print(f"─── [{agent_dir_name}] ──────────────────────────────────")
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    results = _run_pipeline(
                        records, yaml_file, cfg, tmp_dir,
                        bypass_cache=bypass_cache,
                        archive_dir=output_dir / "runtime" / agent_key,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries and ("InternalServerError" in type(e).__name__
                                                   or "Connection" in str(e)):
                        wait = 30 * attempt
                        print(f"  [retry {attempt}/{max_retries}] server error, "
                              f"waiting {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        raise
            _write_outputs(
                results, text_lookup, output_dir,
                agent_dir_name, suffix, use_conf,
                model_tag=args.model_tag,
            )
            agents_run.append((yaml_name, agent_dir_name, suffix))
            print(f"─── [{agent_dir_name}] done ({len(results)} docs)\n")

    # ── Post-processing ────────────────────────────────────────────────────
    normalized: dict[str, dict[str, dict]] = {}

    if do_normalize and agents_run:
        print("─── [NormalizationAgent] ──────────────────────────────────")
        try:
            normalized = _run_normalization(output_dir, agents_run)
        except Exception as exc:
            print(f"  WARNING: Normalization failed — {exc}")
            import traceback; traceback.print_exc()
        print("─── [NormalizationAgent] done\n")

    if do_integrate and agents_run:
        print("─── [IntegrationAgent] ──────────────────────────────────")
        try:
            _run_integration(output_dir, agents_run, meti_dir, normalized)
        except Exception as exc:
            print(f"  WARNING: Integration failed — {exc}")
            import traceback; traceback.print_exc()
        print("─── [IntegrationAgent] done\n")

    if judge_dir and agents_run:
        print("─── [JudgeMerge] ────────────────────────────────────────")
        try:
            _run_judge_merge(output_dir, agents_run, judge_dir)
        except Exception as exc:
            print(f"  WARNING: Judge merge failed — {exc}")
            import traceback; traceback.print_exc()
        print("─── [JudgeMerge] done\n")

    if args.single_output and agents_run:
        print("─── [SingleOutput] ──────────────────────────────────────")
        try:
            _merge_to_single_output(output_dir, agents_run)
        except Exception as exc:
            print(f"  WARNING: Single output merge failed — {exc}")
            import traceback; traceback.print_exc()
        print("─── [SingleOutput] done\n")

    print(f"All done. Output: {output_dir}")


if __name__ == "__main__":
    main()
