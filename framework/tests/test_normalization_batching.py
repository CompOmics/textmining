from types import SimpleNamespace

import numpy as np


def test_index_search_many_embeds_queries_once(monkeypatch):
    from normalization.index import OntologyIndex

    index = OntologyIndex()
    index.embeddings = np.ones((1, 3), dtype=np.float32)
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(index, "_embed_texts", fake_embed)
    monkeypatch.setattr(
        index,
        "_search_index",
        lambda embedding, top_k: [("ID:1", "term", 0.9)],
    )

    results = index.search_many(["one", "two", "three"])

    assert calls == [["one", "two", "three"]]
    assert len(results) == 3


def test_agent_batches_values_by_entity_type():
    from agents.normalization_agent import NormalizationAgent

    calls = []

    class FakeNormalizer:
        def normalize_batch(self, terms, entity_type=None):
            calls.append((entity_type, list(terms)))
            return [
                SimpleNamespace(
                    is_normalized=True,
                    ontology_id=f"{entity_type}:{i}",
                    ontology_name=term.upper(),
                    similarity=0.91,
                )
                for i, term in enumerate(terms)
            ]

    agent = NormalizationAgent(auto_load=False)
    agent.normalizer = FakeNormalizer()
    agent._loaded = True

    output = agent.normalize_batch(
        {
            "a.json": {
                "species": ["Homo sapiens", "human samples"],
                "tissue": ["liver", "liver tissue"],
            },
            "b.json": {
                "species": ["Mus musculus", "mouse samples"],
                "tissue": ["brain", "brain tissue"],
            },
        }
    )

    assert calls == [
        ("species", ["Homo sapiens", "Mus musculus"]),
        ("tissue", ["liver", "brain"]),
    ]
    assert output["a.json"]["species"]["ontology_id"] == "species:0"
    assert output["b.json"]["species"]["ontology_id"] == "species:1"
    assert output["a.json"]["tissue"]["evidence"] == "liver tissue"


def test_agent_normalizes_each_multi_value_independently():
    from agents.normalization_agent import NormalizationAgent

    calls = []

    class FakeNormalizer:
        def normalize_batch(self, terms, entity_type=None):
            calls.append((entity_type, list(terms)))
            return [
                SimpleNamespace(
                    is_normalized=True,
                    ontology_id=f"{entity_type}:{i}",
                    ontology_name=term.upper(),
                    similarity=0.95,
                )
                for i, term in enumerate(terms)
            ]

    agent = NormalizationAgent(auto_load=False)
    agent.normalizer = FakeNormalizer()
    agent._loaded = True

    output = agent.normalize_batch(
        {
            "multi.json": {
                "species": [
                    {"value": "Homo sapiens", "evidence": "human samples"},
                    {"value": "Mus musculus", "evidence": "mouse samples"},
                ],
                "tissue": [{"value": "unknown", "evidence": ""}],
            }
        }
    )

    assert calls == [("species", ["Homo sapiens", "Mus musculus"])]
    assert [entry["value"] for entry in output["multi.json"]["species"]] == [
        "Homo sapiens",
        "Mus musculus",
    ]
    assert [entry["ontology_id"] for entry in output["multi.json"]["species"]] == [
        "species:0",
        "species:1",
    ]
    assert output["multi.json"]["tissue"] == [
        {"value": "unknown", "evidence": "", "is_normalized": False}
    ]


def test_agent_emits_namespace_and_taxon_qc_flags():
    from agents.normalization_agent import NormalizationAgent
    from normalization.ontology import OntologyGraph, OntologyNode

    class FakeNormalizer:
        def __init__(self):
            uberon = OntologyGraph("uberon")
            uberon.add_node(OntologyNode(
                id="UBERON:6110636",
                name="insect adult cerebral ganglion",
                taxon_ids=["NCBITaxon:50557"],
            ))
            self.graphs = {"uberon": uberon}

        def normalize_batch(self, terms, entity_type=None):
            mappings = {
                "species": ("NCBITaxon:9606", "Homo sapiens"),
                "tissue": (
                    "UBERON:6110636", "insect adult cerebral ganglion"
                ),
                "cell_type": ("UBERON:0000955", "brain"),
            }
            ontology_id, ontology_name = mappings[entity_type]
            return [
                SimpleNamespace(
                    is_normalized=True,
                    ontology_id=ontology_id,
                    ontology_name=ontology_name,
                    similarity=1.0,
                    normalization_method="semantic",
                    matched_text=term,
                    synonym_scope=None,
                    candidates=[],
                    qc_flags=[],
                )
                for term in terms
            ]

    agent = NormalizationAgent(auto_load=False)
    agent.normalizer = FakeNormalizer()
    agent._loaded = True

    record = agent.normalize_batch({
        "x.json": {
            "species": [{"value": "human", "evidence": "human"}],
            "tissue": [{"value": "brain", "evidence": "brain"}],
            "cell_type": [{"value": "brain", "evidence": "brain"}],
        }
    })["x.json"]

    assert record["tissue"][0]["normalization_qc_flags"] == [
        "taxon_incompatible_mapping"
    ]
    assert record["tissue"][0]["ontology_taxon_constraints"] == [
        "NCBITaxon:50557"
    ]
    assert record["cell_type"][0]["normalization_qc_flags"] == [
        "cross_namespace_mapping"
    ]


def test_taxon_qc_uses_loaded_taxonomy_ancestors():
    from agents.normalization_agent import NormalizationAgent
    from normalization.ontology import OntologyGraph, OntologyNode

    taxonomy = OntologyGraph("cl")
    taxonomy.add_node(OntologyNode(
        id="NCBITaxon:7776", name="Gnathostomata"
    ))
    taxonomy.add_node(OntologyNode(
        id="NCBITaxon:9606",
        name="Homo sapiens",
        parents=["NCBITaxon:7776"],
    ))

    uberon = OntologyGraph("uberon")
    uberon.add_node(OntologyNode(
        id="UBERON:0002048",
        name="lung",
        taxon_ids=["NCBITaxon:7776"],
    ))

    agent = NormalizationAgent(auto_load=False)
    agent.normalizer = SimpleNamespace(graphs={
        "cl": taxonomy,
        "uberon": uberon,
    })
    record = {
        "species": [{"ontology_id": "NCBITaxon:9606"}],
        "tissue": [{"ontology_id": "UBERON:0002048"}],
    }

    agent._add_taxon_qc(record)

    assert "normalization_qc_flags" not in record["tissue"][0]
