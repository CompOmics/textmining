"""Shared vocabularies for the HAMLET-vs-MLMarker analyses.

Recovered from the former ExtendedTrainingSet/scripts/build_atlas.py so that
MLMarkerPenalty is self-contained.
"""
OLD_CLASSES = {
    "Adipose tissue", "Adrenal gland", "Appendix", "B-cells", "Bone marrow", "Brain", "Colon",
    "Duodenum", "Endometrium", "Esophagus", "Heart", "Kidney", "Liver", "Lung", "Lymph node",
    "Monocytes", "Nasal Polyps", "Ovary", "Oviduct", "Parotid gland", "Pituitary gland", "Placenta",
    "Prostate", "Rectum", "Salivary gland", "Skeletal muscle", "Small intestine", "Smooth muscle",
    "Spleen", "Stomach", "Testis", "Thyroid", "Tonsil", "Urinary bladder",
}
CULTURED = {"pluripotent stem cell", "stem cell", "stem cells", "induced pluripotent stem cell",
            "embryonic stem cell", "organoid", "cell culture"}
NON_TISSUE = {"dental plaque", "secretion of lacrimal gland", "renal system"}
# material context. material_type is HAMLET's dedicated field and decides;
# sample_source and cell_type only exclude unambiguous derivatives.
MATERIAL_OK = r"tissue|primary cell|single fibre|aspirate|biops|ffpe|resection|autops"
MATERIAL_BAD = r"cell culture|cell line|organoid|xenograft|pdx|mummy|recombinant|synthetic"
DERIVATIVE = (r"fibroblast|stem cell|ipsc|induced pluripotent|perfus|supernatant|secretome|"
              r"conditioned medium|sperm|spermato|semen|seminal|exosome|extracellular vesicle|keratinocyte")
BIOFLUIDS = {
    "blood", "blood plasma", "blood serum", "plasma", "serum", "urine",
    "cerebrospinal fluid", "saliva", "tear fluid", "tears", "sweat", "milk",
    "sputum", "cervicovaginal fluid", "seminal fluid", "semen", "sperm",
    "synovial fluid", "bronchoalveolar lavage fluid", "amniotic fluid",
    "bile", "ascites", "pleural fluid", "lymph", "exosome", "extracellular vesicle",
}
# UBERON-style HAMLET labels -> MLMarker class names where the class exists
CLASS_MAP = {
    "thyroid gland": "Thyroid", "thyroid": "Thyroid", "brain": "Brain",
    "liver": "Liver", "lung": "Lung", "heart": "Heart", "kidney": "Kidney",
    "cortex of kidney": "Kidney", "colon": "Colon", "rectum": "Rectum",
    "stomach": "Stomach", "duodenum": "Duodenum", "small intestine": "Small intestine",
    "esophagus": "Esophagus", "spleen": "Spleen", "lymph node": "Lymph node",
    "tonsil": "Tonsil", "vermiform appendix": "Appendix", "appendix": "Appendix",
    "testis": "Testis", "ovary": "Ovary", "prostate": "Prostate", "prostate gland": "Prostate",
    "placenta": "Placenta", "uterine endometrium": "Endometrium", "endometrium": "Endometrium",
    "fallopian tube": "Oviduct", "oviduct": "Oviduct", "urinary bladder": "Urinary bladder",
    "adrenal gland": "Adrenal gland", "pituitary gland": "Pituitary gland", "hypophysis": "Pituitary gland",
    "salivary gland": "Salivary gland", "parotid gland": "Parotid gland",
    "skeletal muscle": "Skeletal muscle", "skeletal muscle tissue": "Skeletal muscle", "muscle": "Skeletal muscle",
    "smooth muscle": "Smooth muscle", "adipose tissue": "Adipose tissue", "subcutaneous fat": "Adipose tissue",
    "bone marrow": "Bone marrow", "monocytes": "Monocytes", "monocyte": "Monocytes",
    "b cells": "B-cells", "b-cells": "B-cells", "b cell": "B-cells", "nasal polyps": "Nasal Polyps",
    "saliva-secreting gland": "Salivary gland", "cerebral cortex": "Brain", "prefrontal cortex": "Brain",
    "frontal lobe": "Brain", "temporal lobe": "Brain", "cortex": "Brain",
}
