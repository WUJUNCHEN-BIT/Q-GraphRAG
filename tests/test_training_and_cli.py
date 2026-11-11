import json

import pytest
import torch

from qgraphrag.checkpoints import load_checkpoint
from qgraphrag.cli import main
from qgraphrag.io import write_json
from qgraphrag.linking import EntityLinker
from qgraphrag.metrics import answer_scores, evaluate_predictions
from qgraphrag.retrieval import GraphRetriever
from qgraphrag.tensors import tensorize
from qgraphrag.training import finetune, prepare_queries, pretrain


def test_two_stage_training_checkpoint_and_gated_retrieval(
    toy_graph, tiny_encoder, small_config, tmp_path, recording_backend
):
    torch.manual_seed(15)
    graph = tensorize(toy_graph, tiny_encoder)
    rows = [
        {
            "id": "q1",
            "question": "Where was Ada Stone born?",
            "answer": "Riverford",
            "question_entities": ["Ada Stone"],
            "supporting_facts": ["d1", "d2"],
        },
        {
            "id": "q2",
            "question": "Where was Mira Vale born?",
            "answer": "Eastport",
            "question_entities": ["Mira Vale"],
            "supporting_facts": [["d3", 0], ["d4", 0]],
        },
    ]
    write_json(small_config["paths"]["train"], rows)
    backend = recording_backend(lambda task, payload: {"entities": []})
    linker = EntityLinker(toy_graph, tiny_encoder, backend)
    data = prepare_queries(
        small_config["paths"]["train"], toy_graph, tiny_encoder, linker, tmp_path
    )
    pretrain(small_config, graph)
    pretrained = tmp_path / "pretrain.pt"
    (tmp_path / "output" / "best.pt").replace(pretrained)
    with pytest.raises(ValueError, match="quality-supervised"):
        load_checkpoint(pretrained, require_trained_quality=True)
    finetune(small_config, graph, data, None, pretrained=pretrained)
    model, metadata = load_checkpoint(tmp_path / "output" / "best.pt", require_trained_quality=True)
    assert metadata["stage"] == "finetune"
    assert metadata["epoch"] == 1
    model.eval().load_gate_cache(graph, tmp_path / "output" / "best_gates.pt")
    result = GraphRetriever(toy_graph, graph, model, tiny_encoder, linker, small_config).retrieve(
        "Where was Ada Stone born?", ["Ada Stone"]
    )
    assert result.documents and result.subgraph["nodes"]
    assert result.seed_entities == ["ada stone"]
    assert not backend.calls  # Supplied question-side entities do not consult gold evidence.


def test_evaluation_uses_final_documents_and_answer_aliases(toy_graph):
    rows = [
        {
            "id": "q",
            "answer": "Lumen",
            "answer_aliases": ["The Lumen"],
            "supporting_facts": ["d1", "d2"],
        }
    ]
    result = evaluate_predictions(
        [{"id": "q", "answer": "the Lumen", "retrieved_document_ids": ["d2", "d1"]}],
        rows,
        toy_graph,
        2,
    )
    assert result["recall@2"] == 100 and result["em"] == 100 and result["f1"] == 100
    assert answer_scores("yes", ["no"])[1] == 0
    with pytest.raises(ValueError, match="IDs must match"):
        evaluate_predictions([], rows, toy_graph)


def test_inspect_index_cli(toy_graph, tmp_path, capsys):
    toy_graph.save(tmp_path)
    main(["--set", f"paths.index={tmp_path}", "inspect-index"])
    assert json.loads(capsys.readouterr().out)["provenance_valid"] is True


def test_bad_configuration_is_rejected():
    from qgraphrag.config import load_config

    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(overrides=["pcst.misspelled=2"])
    with pytest.raises(ValueError, match="overlap_chars"):
        load_config(overrides=["construction.overlap_chars=99999"])


def test_named_entity_linking_does_not_skip_other_mentions_after_one_exact_match(
    toy_graph, tiny_encoder, recording_backend
):
    backend = recording_backend(lambda task, payload: {"entities": ["Ada Stone", "Riverford"]})
    linker = EntityLinker(toy_graph, tiny_encoder, backend)
    ids = linker.link("Did Ada Stone live in Riverford?")
    assert {toy_graph.entities[i] for i in ids} == {"ada stone", "riverford"}
    assert len(backend.calls) == 1 and backend.calls[0][0] == "query_entities"


@pytest.mark.parametrize("stage", ["imported_gfm_retrieval", "imported_gfm_pretrain"])
def test_gfm_initialization_preserves_qa_projection_or_initializes_from_relations(
    stage, small_config, tmp_path
):
    from qgraphrag.checkpoints import initialize_from_pretrain, save_checkpoint
    from qgraphrag.gnn.model import QualityGNNRetriever
    from qgraphrag.io import read_json
    from qgraphrag.training import _save_run_config

    source = QualityGNNRetriever(small_config["model"], 8)
    with torch.no_grad():
        source.question_mlp.weight.fill_(2)
        source.rel_mlp.weight.fill_(3)
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(checkpoint, source, {}, stage)
    destination = QualityGNNRetriever(small_config["model"], 8)
    quality = {
        key: value.clone() for key, value in destination.quality_encoder.state_dict().items()
    }
    initialize_from_pretrain(destination, checkpoint)
    expected = source.question_mlp if stage.endswith("retrieval") else source.rel_mlp
    torch.testing.assert_close(destination.question_mlp.weight, expected.weight)
    for key, value in destination.quality_encoder.state_dict().items():
        torch.testing.assert_close(value, quality[key])
    config = dict(small_config, model={"input_dim": 512})
    _save_run_config(tmp_path, config, destination)
    assert read_json(tmp_path / "resolved_config.json")["model"] == destination.model_config


@pytest.mark.parametrize("value", ["-1", "true", "1.5", "null"])
def test_supplementation_budget_configuration_is_validated(value):
    from qgraphrag.config import load_config

    with pytest.raises(ValueError, match="max_supplementation_rounds"):
        load_config(overrides=[f"retrieval.max_supplementation_rounds={value}"])


def test_prediction_manifest_rejects_previous_pipeline_version(
    toy_graph, small_config, tmp_path, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    from qgraphrag import cli
    from qgraphrag.io import file_sha256, fingerprint, redact

    rows = [{"id": "q", "question": "original question"}]
    queries = tmp_path / "queries.json"
    checkpoint = tmp_path / "checkpoint.pt"
    write_json(queries, rows)
    checkpoint.write_bytes(b"checkpoint fingerprint")
    write_json(
        Path(small_config["paths"]["output"]) / "prediction_manifest.json",
        {
            "schema_version": 2,
            "graph_fingerprint": toy_graph.fingerprint,
            "checkpoint_sha256": file_sha256(checkpoint),
            "configuration": redact(small_config),
            "queries_fingerprint": fingerprint(rows),
        },
    )
    monkeypatch.setattr(cli, "_pipeline", lambda *args: pytest.fail("Old run must not be reused"))
    with pytest.raises(ValueError, match="different run"):
        cli._predict(
            small_config,
            toy_graph,
            SimpleNamespace(queries=queries, checkpoint=checkpoint, limit=None),
        )
