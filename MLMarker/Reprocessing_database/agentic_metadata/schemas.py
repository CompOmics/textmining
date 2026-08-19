"""
JSON schemas for structured output from Ollama.

Single unified schema: one call per paper, outputting an array of sample_groups.
Each sample_group represents a distinct experimental setup within the paper.
"""

# Reusable pattern: array of {value, evidence} pairs, or empty array if not found
EVIDENCE_ARRAY = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["value", "evidence"],
    },
}

SAMPLE_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        # Group identifier
        "name": {"type": "string"},

        # Biological
        "organism": EVIDENCE_ARRAY,
        "tissue": EVIDENCE_ARRAY,
        "disease": EVIDENCE_ARRAY,
        "cell_part": EVIDENCE_ARRAY,
        "cell_line": EVIDENCE_ARRAY,

        # Experimental 1: MS and sample prep
        "instrument": EVIDENCE_ARRAY,
        "fragmentation": EVIDENCE_ARRAY,
        "enzymes": EVIDENCE_ARRAY,
        "modifications": EVIDENCE_ARRAY,
        "collision_energy": EVIDENCE_ARRAY,
        "gradient_time_min": EVIDENCE_ARRAY,
        "lc_column": EVIDENCE_ARRAY,
        "acquisition": EVIDENCE_ARRAY,
        "labeling": EVIDENCE_ARRAY,
        "fractionation": {
            "type": "object",
            "properties": {
                "value": {"type": "boolean"},
                "method": {"type": ["string", "null"]},
                "evidence": {"type": "string"},
            },
            "required": ["value", "method", "evidence"],
        },
        "enrichment": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]},
                "method": {"type": ["string", "null"]},
                "evidence": {"type": "string"},
            },
            "required": ["value", "method", "evidence"],
        },
        "ionization": EVIDENCE_ARRAY,

        # Experimental 2: Treatments / perturbations
        "treatment_type": EVIDENCE_ARRAY,
        "treatment_name": EVIDENCE_ARRAY,
        "treatment_class": EVIDENCE_ARRAY,
    },
    "required": [
        "name",
        "organism", "tissue", "disease", "cell_part", "cell_line",
        "instrument", "fragmentation", "enzymes", "modifications",
        "collision_energy", "gradient_time_min", "lc_column", "acquisition",
        "labeling", "fractionation", "enrichment", "ionization",
        "treatment_type", "treatment_name", "treatment_class",
    ],
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_groups": {
            "type": "array",
            "items": SAMPLE_GROUP_SCHEMA,
        },
    },
    "required": ["sample_groups"],
}
