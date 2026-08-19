"""Predict tissue from protein abundance using MLMarker."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


def run(quant_path: Path, output_path: Path, confidence_threshold: float = 0.3):
    """Run the MLMarker tissue prediction pipeline.

    Args:
        quant_path: Path to pride_quant.parquet
        output_path: Where to write the output TSV
        confidence_threshold: Minimum prediction confidence to keep
    """
    from mlmarker import MLMarker
    from mlmarker.utils import validate_sample

    log.info("Loading quant data from %s", quant_path)
    quant = pd.read_parquet(quant_path)
    protein_cols = [c for c in quant.columns if c not in ["pxd", "run", "source"]]

    m = MLMarker()
    log.info("MLMarker: %d features, %d tissue classes", len(m.get_model_features()), len(m.get_model_classes()))

    # Validate all samples at once
    all_samples = quant[protein_cols]
    validated = validate_sample(m.features, all_samples)
    log.info("Validated matrix: %s", validated.shape)

    # Predict all at once
    log.info("Running predict_proba...")
    probabilities = m.model.predict_proba(validated)
    classes = m.model.classes_

    # Extract top-1
    top1_idx = probabilities.argmax(axis=1)
    top1_tissue = classes[top1_idx]
    top1_conf = probabilities[np.arange(len(probabilities)), top1_idx]

    results = pd.DataFrame({
        "pxd": quant["pxd"].values,
        "run": quant["run"].values,
        "tissue": top1_tissue,
        "confidence": np.round(top1_conf, 4),
    })

    # Filter to confident predictions
    confident = results[results["confidence"] >= confidence_threshold].copy()
    confident = confident.sort_values(["pxd", "run"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    confident.to_csv(output_path, sep="\t", index=False)

    log.info("Predictions: %d total, %d above %.2f threshold (%.1f%%)",
             len(results), len(confident), confidence_threshold,
             len(confident) / len(results) * 100)
    log.info("Saved to %s", output_path)
