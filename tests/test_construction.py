import copy

import pytest

from qgraphrag.construction import (
    Canonicalizer,
    assess_edge,
    build_index,
    locate_evidence,
    merge_sources,
    reflect_chunk,
)
from qgraphrag.io import read_json, write_json
from qgraphrag.schema import Chunk, GraphIndex, Triple, split_chunks


def test_exact_and_normalized_evidence_offsets():
    chunk = Chunk("c", "d", "Title", "Ada  Stone was born\nin Riverford.", 15, 47)
    found = locate_evidence(chunk, "Ada Stone was born in Riverford.")
    assert found == (15, 15 + len(chunk.text), chunk.text)
    assert locate_evidence(chunk, "Ada Stone was born in Eastport.") is None
    assert locate_evidence(chunk, "Riverford", 0, 3)[2] == "Riverford"


def test_reflection_revises_unique_correction_and_drops_forged_span(
    recording_backend,
    reflection_record,
):
    chunk = Chunk("c", "d", "Ada", "Ada was born in Riverford.", 0, 25)
    initial = {"id": "t0", "head": "Ada", "relation": "born in", "tail": "Eastport"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [initial]}
        return {
            "decisions": [
                reflection_record(
                    {**initial, "tail": "Riverford"},
                    "REVISE",
                    evidence=chunk.text,
                    equivalent=[True, True, False],
                )
            ]
        }

    backend = recording_backend(handler)
    kept, dropped = reflect_chunk(chunk, backend)
    assert not dropped and len(kept) == 1
    assert kept[0].initial["tail"] == "Eastport"
    assert kept[0].final["tail"] == "Riverford"
    assert kept[0].values() == pytest.approx((1, 2 / 3, 1))
    assert [task for task, _ in backend.calls] == ["extract", "reflect"]

    def forged(task, payload):
        result = handler(task, payload)
        if task == "reflect":
            result["decisions"][0]["evidence"] = "Ada was born in Atlantis."
        return result

    kept, dropped = reflect_chunk(chunk, recording_backend(forged))
    assert not kept and dropped[0]["reason"] == "Evidence span not found in source"


def test_keep_revise_drop_have_traceable_outcomes(recording_backend, reflection_record):
    chunk = Chunk("c", "d", "Title", "Ada lives in Lumen.", 0, 19)
    triples = [
        {"id": str(i), "head": "Ada", "relation": "lives in", "tail": "Lumen"} for i in range(3)
    ]

    def handler(task, payload):
        if task == "extract":
            return {"triples": triples}
        return {
            "decisions": [
                reflection_record(triples[0], id="0", evidence=chunk.text),
                reflection_record(
                    triples[1], "REVISE", id="1", evidence=chunk.text, unique_correction=False
                ),
                {"id": "2", "decision": "DROP", "reason": "not recoverable"},
            ]
        }

    retained, rejected = reflect_chunk(chunk, recording_backend(handler))
    assert len(retained) == 1 and retained[0].decision == "KEEP"
    assert len(rejected) == 2


def test_conflicting_qualifiers_not_merged(toy_graph, assessed_source):
    source = toy_graph.edges[0].sources[0]
    variants = [
        assessed_source(
            source,
            id="p" + str(i),
            final={**source.final, "qualifiers": [qualifier]},
        )
        for i, qualifier in enumerate(["before 1990", "after 2000", "not after 2000"])
    ]
    edges = merge_sources(variants, Canonicalizer())
    assert len(edges) == 3
    assert {tuple(e.qualifiers) for e in edges} == {
        ("before 1990",),
        ("after 2000",),
        ("not after 2000",),
    }


def test_canonicalization_requires_already_assessed_aliases(toy_graph, assessed_source):
    canonicalizer = Canonicalizer(
        {
            "entities": {"NYC": "New York City"},
            "relations": {"is situated in": "located in"},
        }
    )
    triple = canonicalizer.triple(Triple("NYC", "is situated in", "C++ Labs"))
    assert triple == Triple("new york city", "located in", "c++ labs")
    with pytest.raises(ValueError, match="Cycle"):
        Canonicalizer({"entities": {"a": "b", "b": "a"}})
    source = toy_graph.edges[0].sources[0]
    unsafe = Canonicalizer({"relations": {"born in": "not born in"}})
    with pytest.raises(ValueError, match="previously assessed"):
        merge_sources([source], unsafe)


def test_reflection_checks_aliases_before_assessment(recording_backend, reflection_record):
    chunk = Chunk("c", "d", "Title", "Ada lives in NYC.", 0, 17)
    triple = {"id": "t0", "head": "Ada", "relation": "lives in", "tail": "NYC"}
    canonicalizer = Canonicalizer({"entities": {"NYC": "New York City"}})

    def handler(task, payload):
        if task == "extract":
            return {"triples": [triple]}
        assert payload["canonical_aliases"]["entities"]["nyc"] == "new york city"
        return {
            "decisions": [
                reflection_record(
                    {**triple, "tail": "New York City"},
                    evidence=chunk.text,
                )
            ]
        }

    kept, _ = reflect_chunk(chunk, recording_backend(handler), canonicalizer)
    assert kept[0].values() == (1, 1, 1)
    assert merge_sources(kept, canonicalizer)[0].tail == "new york city"


def test_graph_roundtrip_validates_provenance_and_fingerprint(toy_graph, tmp_path):
    toy_graph.save(tmp_path)
    loaded = GraphIndex.load(tmp_path)
    assert loaded.fingerprint == toy_graph.fingerprint
    loaded.edges[0].sources[0].evidence = "forged"
    with pytest.raises(ValueError, match="locatable"):
        loaded.validate()


def test_chunk_offsets_cover_original_document(toy_graph):
    doc = toy_graph.documents[0]
    chunks = split_chunks(doc, 12, 3)
    assert chunks[0].start == 0 and chunks[-1].end == len(doc.text)
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)
    assert all(left.end >= right.start for left, right in zip(chunks, chunks[1:]))


def test_index_build_persists_reflection_before_pure_aggregation(
    small_config,
    recording_backend,
    reflection_record,
    monkeypatch,
):
    from pathlib import Path

    from qgraphrag import construction

    text = "Ada lives in Lumen."
    write_json(small_config["paths"]["corpus"], [{"id": "d", "title": "Ada", "text": text}])
    triple = {"id": "t0", "head": "Ada", "relation": "lives in", "tail": "Lumen"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [triple]}
        assert task == "reflect"
        return {"decisions": [reflection_record(triple, evidence=text)]}

    def aggregate(edge):
        records = read_json(Path(small_config["paths"]["index"]) / "reflection_records.json")
        assert len(records["sources"]) == 1
        original = copy.deepcopy(edge.sources)
        result = assess_edge(edge)
        assert edge.sources == original
        return result

    backend = recording_backend(handler)
    monkeypatch.setattr(construction, "assess_edge", aggregate)
    graph = build_index(small_config, backend)
    assert graph.edges[0].features == [1, 1, 1]
    assert [task for task, _ in backend.calls] == ["extract", "reflect"]
    assert GraphIndex.load(small_config["paths"]["index"]).fingerprint == graph.fingerprint
