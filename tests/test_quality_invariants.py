import copy

import pytest
import torch

from qgraphrag.construction import Canonicalizer, assess_edge, merge_sources, reflect_chunk
from qgraphrag.gnn.model import QualityGNNRetriever
from qgraphrag.schema import QUALITY_FEATURE_VERSION, Chunk, GraphIndex, assessment_values
from qgraphrag.tensors import tensorize


@pytest.mark.parametrize(
    "qualifiers,expected",
    [
        ([], 1),
        ([{"text": "in 2020", "preserved": False}], 2 / 3),
        (
            [
                {"text": "in 2020", "preserved": True},
                {"text": "only as chair", "preserved": True},
                {"text": "subject to renewal", "preserved": False},
            ],
            8 / 9,
        ),
    ],
)
def test_source_fidelity_uses_exact_qualifier_fraction(reflection_record, qualifiers, expected):
    final = {"head": "Ada", "relation": "led", "tail": "Institute"}
    record = reflection_record(final, qualifier_assessments=qualifiers)
    assert assessment_values(record) == pytest.approx((1, 1, expected))


def test_fidelity_is_mean_of_source_scores_not_pooled_components(toy_graph, assessed_source):
    edge = copy.deepcopy(toy_graph.edges[0])
    first = assessed_source(edge.sources[0], assessment={"equivalent": [True, True, True]})
    second = assessed_source(
        edge.sources[0],
        id="second",
        document_id="another-document",
        assessment={"qualifier_assessments": [{"text": "in 2020", "preserved": False}]},
    )
    edge.sources = [first, second]
    assert assess_edge(edge).features == pytest.approx([1, 5 / 6, 5 / 6])


def test_full_support_and_incomplete_fidelity_are_independent(toy_graph, assessed_source):
    edge = copy.deepcopy(toy_graph.edges[0])
    edge.sources = [
        assessed_source(
            edge.sources[0],
            assessment={
                "relation_retention": "generalized",
                "qualifier_assessments": [{"text": "in 2020", "preserved": False}],
            },
        )
    ]
    assert assess_edge(edge).features == pytest.approx([1, 2 / 3, 0.5])


def test_revise_with_added_relation_qualifier_has_consistency_two_thirds(
    recording_backend,
    reflection_record,
):
    text = "Ada won Prize B in 2020."
    chunk = Chunk("c", "d", "Ada", text, 0, len(text))
    initial = {"id": "t0", "head": "Ada", "relation": "won", "tail": "Prize B"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [initial]}
        return {
            "decisions": [
                reflection_record(
                    {**initial, "qualifiers": ["in 2020"]},
                    "REVISE",
                    evidence=text,
                    equivalent=[True, False, True],
                    qualifier_assessments=[{"text": "in 2020", "preserved": True}],
                )
            ]
        }

    kept, dropped = reflect_chunk(chunk, recording_backend(handler))
    edge = assess_edge(merge_sources(kept, Canonicalizer())[0])
    assert not dropped
    assert edge.features == pytest.approx([1, 2 / 3, 1])


def test_stable_dedup_selects_one_complete_record_for_all_features(toy_graph, assessed_source):
    source = toy_graph.edges[0].sources[0]
    first = assessed_source(
        source,
        id="a",
        chunk_id="first",
        assessment={
            "support_status": "partial",
            "residual_issues": ["unresolved scope"],
            "equivalent": [True, True, True],
        },
    )
    second = assessed_source(
        source,
        id="b",
        chunk_id="second",
        initial={
            **source.initial,
            "tail": "a different initial extraction",
        },
        assessment={
            "equivalent": [False, False, False],
            "relation_retention": "generalized",
        },
    )
    results = []
    for records in ([first, second], [second, first]):
        duplicates = []
        edges = merge_sources(records, Canonicalizer(), duplicates)
        edge = assess_edge(edges[0])
        assert len(edge.sources) == 1 and edge.sources[0].id == "a"
        assert edge.features == pytest.approx([0.5, 1, 1])
        assert len(duplicates) == 1 and duplicates[0]["assessment_disagrees"]
        assert duplicates[0]["duplicate"]["initial"] == second.initial
        results.append((edge.features, duplicates))
    assert results[0] == results[1]


def test_overlapping_chunks_dedup_uses_original_document_offsets(
    recording_backend,
    reflection_record,
):
    fact = "Ada lives in Lumen."
    prefix = "Preface. "
    document = prefix + fact + " Postscript."
    chunks = [
        Chunk("a", "d", "Title", document, 0, len(document)),
        Chunk("b", "d", "Title", fact + " Postscript.", len(prefix), len(document)),
    ]
    triple = {"id": "t0", "head": "Ada", "relation": "lives in", "tail": "Lumen"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [triple]}
        start = payload["chunk"]["text"].index(fact)
        return {
            "decisions": [
                reflection_record(
                    triple,
                    evidence=fact,
                    start=start,
                    end=start + len(fact),
                )
            ]
        }

    sources = [reflect_chunk(c, recording_backend(handler))[0][0] for c in chunks]
    assert sources[0].evidence_start == sources[1].evidence_start == len(prefix)
    assert len(merge_sources(sources, Canonicalizer())[0].sources) == 1


def test_dedup_keeps_full_relation_identity(toy_graph, assessed_source):
    source = toy_graph.edges[0].sources[0]
    other = assessed_source(source, id="other", final={**source.final, "relation": "visited"})
    assert len(merge_sources([source, other], Canonicalizer())) == 2


@pytest.mark.parametrize(
    "field",
    [
        "support_status",
        "relation_retention",
        "entity_binding_verified",
        "equivalent",
        "qualifier_assessments",
        "residual_issues",
        "unique_correction",
    ],
)
def test_missing_assessment_is_never_replaced_with_default_scores(toy_graph, field):
    source = toy_graph.edges[0].sources[0]
    source.reflection.pop(field)
    source.assessment_fingerprint = source.assessment_digest()
    with pytest.raises(ValueError):
        assess_edge(toy_graph.edges[0])


@pytest.mark.parametrize("binding", [None, 0.5, 1])
def test_binding_must_be_an_explicit_verified_boolean(reflection_record, binding):
    record = reflection_record(
        {"head": "A", "relation": "r", "tail": "B"}, entity_binding_verified=binding
    )
    with pytest.raises(ValueError, match="entity_binding_verified"):
        assessment_values(record)


def test_duplicate_qualifiers_are_not_counted_twice(reflection_record):
    record = reflection_record(
        {"head": "A", "relation": "r", "tail": "B"},
        qualifier_assessments=[
            {"text": "In 2020", "preserved": True},
            {"text": "in 2020 ", "preserved": False},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate qualifier"):
        assessment_values(record)


@pytest.mark.parametrize(
    "change",
    [
        {"support_status": "unsupported"},
        {"support_status": "contradicted"},
        {"entity_binding_verified": False},
        {"relation_retention": "incorrect"},
        {"equivalent": [True, False, True]},
    ],
)
def test_invalid_retained_claims_are_filtered_in_reflection(
    change,
    recording_backend,
    reflection_record,
):
    text = "Ada lives in Lumen."
    chunk = Chunk("c", "d", "Title", text, 0, len(text))
    triple = {"id": "t0", "head": "Ada", "relation": "lives in", "tail": "Lumen"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [triple]}
        return {"decisions": [reflection_record(triple, evidence=text, **change)]}

    kept, dropped = reflect_chunk(chunk, recording_backend(handler))
    assert not kept and len(dropped) == 1


def test_supported_unresolved_ambiguity_is_retained(recording_backend, reflection_record):
    text = "Ada lives in Lumen during the appointment."
    chunk = Chunk("c", "d", "Title", text, 0, len(text))
    triple = {"id": "t0", "head": "Ada", "relation": "lives in", "tail": "Lumen"}

    def handler(task, payload):
        if task == "extract":
            return {"triples": [triple]}
        return {
            "decisions": [
                reflection_record(
                    triple,
                    evidence=text,
                    support_status="partial",
                    residual_issues=["appointment scope is ambiguous"],
                    qualifier_assessments=[{"text": "during the appointment", "preserved": False}],
                )
            ]
        }

    kept, dropped = reflect_chunk(chunk, recording_backend(handler))
    assert not dropped and kept[0].values() == pytest.approx((0.5, 1, 2 / 3))


@pytest.mark.parametrize("field", ["final", "document_id", "evidence_start", "reflection"])
def test_assessment_cannot_be_reused_after_changing_final_or_source(toy_graph, field):
    source = toy_graph.edges[0].sources[0]
    if field == "final":
        source.final["qualifiers"] = ["after 2020"]
    elif field == "reflection":
        source.reflection["reason"] = "changed assessment"
    elif field == "document_id":
        source.document_id = "different-source"
    else:
        source.evidence_start += 1
    with pytest.raises(ValueError, match="binding mismatch"):
        source.values()


def test_saved_features_must_be_recomputable_from_provenance(toy_graph):
    toy_graph.edges[0].features[1] = 0.2
    with pytest.raises(ValueError, match="source assessments"):
        toy_graph.validate()


def test_old_feature_version_is_rejected_at_graph_tensor_and_cache_boundaries(
    toy_graph,
    tiny_encoder,
    small_config,
    tmp_path,
):
    graph = tensorize(toy_graph, tiny_encoder)
    model = QualityGNNRetriever(small_config["model"], 8).eval()
    cache = tmp_path / "gates.pt"
    model.cache_gates(graph, cache)
    data = torch.load(cache, weights_only=True)
    data["quality_feature_version"] = QUALITY_FEATURE_VERSION - 1
    torch.save(data, cache)
    with pytest.raises(ValueError, match="does not match"):
        model.load_gate_cache(graph, cache)
    graph.quality_feature_version -= 1
    with pytest.raises(ValueError, match="version mismatch"):
        model.gates(graph)
    toy_graph.save(tmp_path)
    from qgraphrag.io import read_json, write_json

    data = read_json(tmp_path / "graph.json")
    data["quality_feature_version"] -= 1
    write_json(tmp_path / "graph.json", data)
    with pytest.raises(ValueError, match="version mismatch"):
        GraphIndex.load(tmp_path)


def test_checkpoint_version_mismatch_requires_joint_finetuning(small_config, tmp_path):
    from qgraphrag.checkpoints import initialize_from_pretrain, load_checkpoint, save_checkpoint

    model = QualityGNNRetriever(small_config["model"], 8)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {}, "finetune", quality_feature_version=QUALITY_FEATURE_VERSION)
    data = torch.load(path, weights_only=True)
    data["quality_feature_version"] -= 1
    torch.save(data, path)
    with pytest.raises(ValueError, match="version mismatch"):
        load_checkpoint(path, require_trained_quality=True)
    with pytest.raises(ValueError, match="version mismatch"):
        load_checkpoint(path)
    destination = QualityGNNRetriever(small_config["model"], 8)
    before = copy.deepcopy(destination.quality_encoder.state_dict())
    initialize_from_pretrain(destination, path)
    for name, parameter in destination.quality_encoder.state_dict().items():
        torch.testing.assert_close(parameter, before[name])


def test_completion_pretraining_uses_unit_gates_except_removed_target_edges(
    toy_graph,
    tiny_encoder,
    small_config,
):
    graph = tensorize(toy_graph, tiny_encoder)
    model = QualityGNNRetriever(small_config["model"], 8)
    observed = []
    hook = model.entity_model.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[4].clone())
    )
    logits = model.completion(
        graph,
        graph.edge_index[0, :1],
        graph.edge_type[:1],
        torch.tensor([[0, 1, 2]]),
        torch.tensor([0]),
    )
    logits.sum().backward()
    hook.remove()
    expected = torch.ones(graph.edge_type.numel())
    expected[[0, graph.num_edges]] = 0
    torch.testing.assert_close(observed[0], expected)
    assert all(p.grad is None for p in model.quality_encoder.parameters())
