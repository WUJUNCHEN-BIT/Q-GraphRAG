from __future__ import annotations

import copy
import hashlib
import re

import pytest
import torch

from qgraphrag.config import DEFAULTS
from qgraphrag.schema import Document, Edge, GraphIndex, Provenance, assessment_values


def make_reflection(final, decision="KEEP", **changes):
    return {
        "id": "t0",
        "decision": decision,
        "final": copy.deepcopy(final),
        "unique_correction": decision == "REVISE",
        "entity_binding_verified": True,
        "support_status": "full",
        "relation_retention": "complete",
        "equivalent": [True, True, True],
        "qualifier_assessments": [],
        "residual_issues": [],
        **changes,
    }


def with_assessment(source, *, assessment=None, **changes):
    result = copy.deepcopy(source)
    for key, value in changes.items():
        setattr(result, key, value)
    result.reflection = {
        **result.reflection,
        "final": copy.deepcopy(result.final),
        "decision": result.decision,
        **(assessment or {}),
    }
    result.support = assessment_values(result.reflection, retained=False)[0]
    result.equivalent = list(result.reflection["equivalent"])
    result.assessment_fingerprint = result.assessment_digest()
    return result


@pytest.fixture
def reflection_record():
    return make_reflection


@pytest.fixture
def assessed_source():
    return with_assessment


@pytest.fixture
def small_config(tmp_path):
    config = copy.deepcopy(DEFAULTS)
    config["device"] = "cpu"
    config["model"].update(input_dim=4, hidden_dims=[4, 4], quality_hidden_dim=4, edge_chunk_size=2)
    config["paths"].update(
        index=str(tmp_path / "index"),
        output=str(tmp_path / "output"),
        corpus=str(tmp_path / "corpus.json"),
        train=str(tmp_path / "train.json"),
    )
    config["construction"]["workers"] = 1
    config["llm"]["cache_dir"] = str(tmp_path / "llm-cache")
    config["pcst"].update(top_m=2, hops=2)
    config["retrieval"].update(
        entity_top_k=3, graph_doc_top_k=2, text_top_k=2, bm25_top_k=3, vector_top_k=3, final_top_k=2
    )
    for stage in ("pretrain", "finetune"):
        config[stage].update(epochs=1, batch_size=2, steps_per_epoch=1)
    config["pretrain"]["num_negative"] = 3
    return config


class TinyEncoder:
    dimension = 8
    signature = {"model": "test-fixture-only", "normalize": False, "dimension": 8}

    def encode(self, texts):
        matrix = torch.zeros((len(texts), self.dimension))
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                column = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
                matrix[i, column] += 1
        return matrix


@pytest.fixture
def tiny_encoder():
    return TinyEncoder()


@pytest.fixture
def toy_graph():
    rows = [
        (
            "d1",
            "Ada Stone",
            "Ada Stone was born in Riverford.",
            "ada stone",
            "born in",
            "riverford",
        ),
        ("d2", "Riverford", "Riverford is in Lumen.", "riverford", "located in", "lumen"),
        ("d3", "Mira Vale", "Mira Vale was born in Eastport.", "mira vale", "born in", "eastport"),
        ("d4", "Eastport", "Eastport is in Norland.", "eastport", "located in", "norland"),
    ]
    documents, edges, document_entities = [], [], {}
    for i, (doc_id, title, text, head, relation, tail) in enumerate(rows):
        documents.append(Document(doc_id, title, text))
        triple = {"head": head, "relation": relation, "tail": tail, "qualifiers": []}
        source = Provenance(
            "p" + str(i),
            doc_id,
            "c" + str(i),
            0,
            len(text),
            text,
            0,
            len(text),
            triple if i % 2 else {**triple, "relation": "lives in"},
            triple,
            "KEEP" if i % 2 else "REVISE",
            support=1,
            equivalent=[True, bool(i % 2), True],
        )
        source.reflection = make_reflection(
            triple,
            source.decision,
            equivalent=source.equivalent,
            evidence=text,
        )
        source.assessment_fingerprint = source.assessment_digest()
        edges.append(
            Edge(
                "e" + str(i),
                head,
                relation,
                tail,
                [],
                [source],
                [1.0, 1.0 if i % 2 else 2 / 3, 1.0],
            )
        )
        document_entities[doc_id] = [head, tail]
    entities = sorted({x for e in edges for x in (e.head, e.tail)})
    graph = GraphIndex(documents, entities, edges, document_entities)
    graph.validate()
    return graph


class RecordingBackend:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def complete(self, task, payload, validate=None):
        self.calls.append((task, copy.deepcopy(payload)))
        result = self.handler(task, payload)
        if validate:
            validate(result)
        return result


@pytest.fixture
def recording_backend():
    return RecordingBackend
