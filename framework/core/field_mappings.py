"""
Canonical Field Name Mappings
=============================
Single source of truth for all field name mappings between:
- LLM extraction output keys
- SDRF column headers
- Annotation characteristic keys
- Golden-set / benchmark field names
- Ontology entity types

All other modules should import from here instead of defining local mappings.
"""

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
#  LLM Output Key → Canonical Golden Field Name
# ═══════════════════════════════════════════════════════════════════════
# Maps every known LLM output key variant to the canonical golden/SDRF
# field name used in benchmark evaluation. Multiple LLM keys may map
# to the same golden field.

LLM_TO_GOLDEN = {
    # ── Biological Agent ──
    "species":              "species",
    "organism":             "species",
    "tissue":               "organ",          # SDRF uses "organ" for organismpart
    "organ":                "organ",
    "cell_type":            "cell_type",
    "celltype":             "cell_type",
    "cell type":            "cell_type",
    "cell_line":            "cell_line",
    "cellline":             "cell_line",
    "cell line":            "cell_line",
    "disease_state":        "disease",
    "disease":              "disease",
    "sex":                  "sex",
    "age":                  "age",
    "developmental_stage":  "developmental_stage",
    "ethnicity":            "ethnicity",
    "ancestrycategory":     "ethnicity",
    "material_type":        "material_type",
    "materialtype":         "material_type",
    "strain":               "strain",
    "BMI":                  "BMI",
    "bmi":                  "BMI",

    # ── Technical Agent ──
    "instrument":           "instrument",
    "cleavage agent":       "cleavage_agent",
    "cleavage_agent":       "cleavage_agent",
    "labeling":             "label",
    "labelling":            "label",
    "label":                "label",
    "fragmentation method": "fragmentation",
    "fragmentation_method": "fragmentation",
    "fragmentation":        "fragmentation",
    "precursor_tolerance":  "precursor_tolerance",
    "precursor tolerance":  "precursor_tolerance",
    "fragment_tolerance":   "fragment_tolerance",
    "fragment tolerance":   "fragment_tolerance",
    "collision energy":     "collision_energy",
    "collision_energy":     "collision_energy",
    "mass_analyzer":        "mass_analyzer",
    "mass analyzer":        "mass_analyzer",
    "ms2massanalyzer":      "mass_analyzer",
    "ptm":                  "ptm",
    "modification":         "ptm",
    "post_translational_modification": "ptm",
    "acquisition method":   "acquisition_method",
    "acquisition_method":   "acquisition_method",
    "enrichment method":    "enrichment_method",
    "enrichment_method":    "enrichment_method",
    "fractionation method": "fractionation",
    "fractionation_method": "fractionation",
    "ionization type":      "ionization",
    "ionization_type":      "ionization",
    "reduction reagent":    "reduction_reagent",
    "reduction_reagent":    "reduction_reagent",
    "alkylation reagent":   "alkylation_reagent",
    "alkylation_reagent":   "alkylation_reagent",
    "flow_rate":            "flow_rate",
    "flow rate":            "flow_rate",
    "gradient_time":        "gradient_time",
    "gradient time":        "gradient_time",
    "chromatography":       "chromatography",

    # ── Experimental Design Agent ──
    "biological_replicate":              "replicates",
    "number_of_biological_replicates":   "replicates",
    "replicates":                        "replicates",
    "technical_replicate":               "technical_replicates",
    "number_of_technical_replicates":    "technical_replicates",
    "number_of_fractions":               "fractions",
    "fractions":                         "fractions",
    "factor_value":                      "factor_value",
    "experimental_design":               "experimental_design",
    "technology_type":                   "technology_type",
    "number_of_samples":                 "number_of_samples",
    "missed_cleavages":                  "missed_cleavages",
}


def resolve_field_name(name: str) -> str:
    """Resolve any LLM output key to its canonical golden field name.

    If the name is already canonical or unknown, returns it unchanged.
    Tries exact match first, then case-insensitive fallback.
    """
    if name in LLM_TO_GOLDEN:
        return LLM_TO_GOLDEN[name]
    lower = name.lower()
    if lower in LLM_TO_GOLDEN:
        return LLM_TO_GOLDEN[lower]
    return name


# ═══════════════════════════════════════════════════════════════════════
#  SDRF Column Header → Canonical Golden Field Name
# ═══════════════════════════════════════════════════════════════════════
# Keys are lowercased, stripped SDRF header names (after removing
# brackets, comments, and spaces via normalize_header()).

SDRF_BIOLOGICAL = {
    "organism":           "species",
    "organismpart":       "organ",
    "celltype":           "cell_type",
    "cellline":           "cell_line",
    "disease":            "disease",
    "sex":                "sex",
    "age":                "age",
    "developmentalstage": "developmental_stage",
    "ancestrycategory":   "ethnicity",
    "materialtype":       "material_type",
}

SDRF_TECHNICAL = {
    "instrument":              "instrument",
    "cleavageagent":           "cleavage_agent",
    "label":                   "label",
    "fragmentationmethod":     "fragmentation",
    "precursormasstolerance":  "precursor_tolerance",
    "fragmentmasstolerance":   "fragment_tolerance",
    "collisionenergy":         "collision_energy",
    "ms2massanalyzer":         "mass_analyzer",
}

SDRF_EXPERIMENTAL = {
    "technology type": "technology_type",
}

# Combined for convenience
SDRF_TO_GOLDEN = {**SDRF_BIOLOGICAL, **SDRF_TECHNICAL, **SDRF_EXPERIMENTAL}


# ═══════════════════════════════════════════════════════════════════════
#  Annotation Characteristic Key → (Golden Field, Agent)
# ═══════════════════════════════════════════════════════════════════════
# Used by annotation_to_golden.py to convert annotation JSONs.

ANNOTATION_TO_GOLDEN = {
    # Biological fields
    "Organism":           ("species",             "BiologicalAgent"),
    "OrganismPart":       ("organ",               "BiologicalAgent"),
    "CellType":           ("cell_type",           "BiologicalAgent"),
    "CellLine":           ("cell_line",           "BiologicalAgent"),
    "Disease":            ("disease",             "BiologicalAgent"),
    "Sex":                ("sex",                 "BiologicalAgent"),
    "Age":                ("age",                 "BiologicalAgent"),
    "DevelopmentalStage": ("developmental_stage", "BiologicalAgent"),
    "AncestryCategory":   ("ethnicity",           "BiologicalAgent"),
    "MaterialType":       ("material_type",       "BiologicalAgent"),
    "Strain":             ("strain",              "BiologicalAgent"),
    "BMI":                ("BMI",                 "BiologicalAgent"),

    # Technical fields
    "CleavageAgent":      ("cleavage_agent",      "TechnicalAgent"),
    "Label":              ("label",               "TechnicalAgent"),
    "Modification":       ("ptm",                 "TechnicalAgent"),
    "Instrument":         ("instrument",          "TechnicalAgent"),
    "ReductionReagent":   ("reduction_reagent",   "TechnicalAgent"),

    # Experimental design fields
    "NumberOfBiologicalReplicates": ("replicates",            "ExperimentalDesignAgent"),
    "NumberOfTechnicalReplicates":  ("technical_replicates",  "ExperimentalDesignAgent"),
    "NumberOfSamples":              ("number_of_samples",     "ExperimentalDesignAgent"),
}


# ═══════════════════════════════════════════════════════════════════════
#  Canonical Field Name → Ontology Entity Type (for normalization)
# ═══════════════════════════════════════════════════════════════════════
# Maps each normalizable field to the entity type expected by the
# TermNormalizer. This determines which ontology index is queried.

FIELD_TO_ENTITY_TYPE = {
    "species":        "species",
    "organism":       "species",
    "cell_type":      "cell_type",
    "cell_line":      "cell_line",
    "tissue":         "tissue",
    "organ":          "tissue",
    "disease":        "disease",
    "disease_state":  "disease",
    "instrument":     "instrument",
    "labelling":      "labelling",
    "label":          "labelling",
    "modification":   "modification",
    "ptm":            "modification",
    "fractionation":  "fractionation",
    "sample_source":  "sample_source",
}


# ═══════════════════════════════════════════════════════════════════════
#  Canonical Field Name → Ontology ID (for hierarchical matching)
# ═══════════════════════════════════════════════════════════════════════
# Used by the benchmark semantic matcher to enable hierarchy-aware
# comparison (e.g. "heart" is_a "cardiovascular system" in UBERON).

FIELD_ONTOLOGY_MAP = {
    "cell_type": "cl",
    "organ":     "uberon",
    "disease":   "mondo",
}


# ═══════════════════════════════════════════════════════════════════════
#  Agent Type → Expected Fields
# ═══════════════════════════════════════════════════════════════════════
# Canonical field lists per agent, used by the integration agent and
# for filtering benchmark results.

AGENT_FIELDS = {
    "BiologicalAgent": [
        "species", "organism", "tissue", "organ", "cell_type", "cell_line",
        "disease", "disease_state", "age", "BMI", "sex", "strain", "sample_source",
    ],
    "TechnicalAgent": [
        "instrument", "detector", "source", "analyzer", "chromatography",
        "column", "injection_volume", "flow_rate", "gradient", "solvent_A",
        "solvent_B", "MS1_range", "MS2_range", "fragmentation",
        "precursor_selection", "resolution", "software", "database",
        "processing_parameters", "ptm", "modification",
    ],
    "ExperimentalDesignAgent": [
        "experiment_type", "experimental_design", "control_group",
        "treatment_group", "replicates", "time_points",
        "quantification_method", "statistical_test", "software_used",
    ],
}

# Flat list of all agent fields
ALL_AGENT_FIELDS = (
    AGENT_FIELDS["BiologicalAgent"] +
    AGENT_FIELDS["TechnicalAgent"] +
    AGENT_FIELDS["ExperimentalDesignAgent"]
)


# ═══════════════════════════════════════════════════════════════════════
#  Metadata-Only Fields (not extractable from manuscripts)
# ═══════════════════════════════════════════════════════════════════════
# Fields where LLM extraction rate is <25% — these are metadata-only
# fields populated from submission forms, not from manuscript text.
# Determined empirically from LLM extraction rates on 20 PXDs.

METADATA_ONLY_FIELDS = {
    "age", "sex", "ethnicity", "developmental_stage",
    "material_type", "technology_type",
    "precursor_tolerance", "fragment_tolerance", "mass_analyzer",
    "ptm",  # LLM uses different sub-field names for modifications
}
