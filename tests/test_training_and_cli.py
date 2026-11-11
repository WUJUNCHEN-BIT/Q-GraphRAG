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


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wikimultihopqa", "musique"])
def test_dataset_configuration_resolves_four_training_graphs_and_heldout_graph(dataset):
    from pathlib import Path

    from qgraphrag.cli import _graph_config, _index_configs, _training_configs
    from qgraphrag.config import load_config

    path = Path(__file__).resolve().parents[1] / "configs" / (dataset + ".yaml")
    config = load_config(path)
    assert config["pretrain"]["batch_size"] == config["finetune"]["batch_size"] == 4
    assert config["retrieval"]["entity_top_k"] == 20 and config["pcst"]["top_m"] == 5
    training = _training_configs(config)
    assert list(training) == [f"train_{i}" for i in range(4)]
    assert len(_index_configs(config)) == 5
    evaluation = _graph_config(config)
    assert Path(evaluation["paths"]["index"]).name == "eval"
    assert evaluation["paths"]["train"] is None
    for name, value in training.items():
        assert Path(value["paths"]["index"]).name == name
        assert Path(value["paths"]["train"]).parent.parent.name == name


def _training_graphs(toy_graph, assessed_source):
    import copy

    from qgraphrag.construction import assess_edge

    graphs = {"train_0": copy.deepcopy(toy_graph)}
    for i in range(1, 4):
        graph = copy.deepcopy(toy_graph)
        graph.entities = sorted(graph.entities + [f"a extra node {j}" for j in range(i)])
        edge = copy.deepcopy(graph.edges[0])
        edge.id = "extra-edge"
        edge.relation = f"associated via relation {i}"
        edge.sources = [
            assessed_source(
                edge.sources[0],
                id="extra-source",
                final={**edge.sources[0].final, "relation": edge.relation},
                assessment={"relation_retention": "generalized"},
            )
        ]
        graph.edges.append(assess_edge(edge))
        graph.validate()
        graphs[f"train_{i}"] = graph
    return graphs


def test_shared_multigraph_training_resume_and_unseen_graph_inference(
    toy_graph,
    assessed_source,
    tiny_encoder,
    small_config,
    recording_backend,
    tmp_path,
    monkeypatch,
):
    import copy
    from dataclasses import replace

    from qgraphrag import training as module

    graphs = _training_graphs(toy_graph, assessed_source)
    tensors = {name: tensorize(graph, tiny_encoder) for name, graph in graphs.items()}
    backend = recording_backend(lambda *args: pytest.fail("Question-side entities are supplied"))
    datasets = {}
    for name, graph in graphs.items():
        rows = [
            {
                "id": name + "-q" + str(i),
                "question": question,
                "question_entities": [entity],
                "supporting_facts": supports,
            }
            for i, (question, entity, supports) in enumerate(
                [
                    ("Where was Ada Stone born?", "Ada Stone", ["d1", "d2"]),
                    ("Where was Mira Vale born?", "Mira Vale", ["d3", "d4"]),
                ]
            )
        ]
        path = tmp_path / name / "train.json"
        write_json(path, rows)
        datasets[name] = prepare_queries(
            path, graph, tiny_encoder, EntityLinker(graph, tiny_encoder, backend), tmp_path / name
        )
    config = copy.deepcopy(small_config)
    config["pretrain"].update(epochs=1, steps_per_epoch=None)
    config["finetune"].update(epochs=2, steps_per_epoch=None)
    config["paths"]["output"] = str(tmp_path / "pretrain")
    pretrain(config, tensors)
    checkpoint = tmp_path / "pretrain/last.pt"
    original_forward = module.QualityGNNRetriever.forward
    calls = []

    def forward(self, graph, *args, **kwargs):
        calls.append((id(self), id(self.quality_encoder), graph.graph_fingerprint))
        return original_forward(self, graph, *args, **kwargs)

    monkeypatch.setattr(module.QualityGNNRetriever, "forward", forward)
    original_initialize = module.initialize_from_pretrain
    initialized = []

    def initialize(*args):
        initialized.append(True)
        return original_initialize(*args)

    monkeypatch.setattr(module, "initialize_from_pretrain", initialize)
    config["paths"]["output"] = str(tmp_path / "full")
    full = finetune(config, tensors, datasets, None, pretrained=checkpoint)
    assert len(initialized) == 1
    assert len({item[:2] for item in calls}) == 1
    assert {item[2] for item in calls} == {g.graph_fingerprint for g in tensors.values()}
    _, metadata = load_checkpoint(tmp_path / "full/last.pt", require_trained_quality=True)
    assert set(metadata["graph_fingerprints"]) == set(graphs)
    assert set(metadata["training_fingerprints"]) == set(graphs)
    assert metadata["graph_fingerprint"] is None
    assert max(float(state["step"]) for state in metadata["optimizer"]["state"].values()) == 8
    logs = [
        json.loads(line) for line in (tmp_path / "full/training.jsonl").read_text().splitlines()
    ]
    assert all(row["examples_by_graph"] == {name: 2 for name in graphs} for row in logs)
    assert all(row["steps"] == 4 for row in logs)
    monkeypatch.setattr(module.QualityGNNRetriever, "forward", original_forward)
    config["paths"]["output"] = str(tmp_path / "resume")
    config["finetune"]["epochs"] = 1
    finetune(config, tensors, datasets, None, pretrained=checkpoint)
    config["finetune"]["epochs"] = 2
    continued = finetune(
        config,
        dict(reversed(list(tensors.items()))),
        datasets,
        None,
        resume=tmp_path / "resume/last.pt",
    )
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, continued.state_dict()[key], rtol=0, atol=0)
    changed = dict(tensors)
    changed["train_0"] = replace(tensors["train_0"], graph_fingerprint="changed-graph")
    with pytest.raises(ValueError, match="Resume graph/training data mismatch"):
        finetune(config, changed, datasets, None, resume=tmp_path / "resume/last.pt")
    heldout = copy.deepcopy(graphs["train_3"])
    heldout.entities = sorted(heldout.entities + ["unseen isolated entity"])
    evaluation = tensorize(heldout, tiny_encoder)
    model, _ = load_checkpoint(tmp_path / "full/last.pt", require_trained_quality=True)
    model.eval()
    seeds = torch.zeros(1, evaluation.num_nodes)
    seeds[0, heldout.entities.index("ada stone")] = 1
    with torch.inference_mode():
        result = model(evaluation, tiny_encoder.encode(["Ada Stone"]), seeds)
    assert result.logits.shape == (1, evaluation.num_nodes)
    assert torch.isfinite(result.logits).all()
    model.cache_gates(evaluation, tmp_path / "eval-gates.pt")
    model.load_gate_cache(evaluation, tmp_path / "eval-gates.pt")
    with pytest.raises(ValueError, match="does not match"):
        model.load_gate_cache(tensors["train_0"], tmp_path / "eval-gates.pt")


def test_cli_training_does_not_load_evaluation_graph(
    toy_graph,
    tiny_encoder,
    small_config,
    tmp_path,
    monkeypatch,
):
    from pathlib import Path

    import yaml

    from qgraphrag import cli, training

    config = small_config
    config["datasets"].update(root=str(tmp_path / "data"), train_names=["a", "b"], eval_name="eval")
    for name in ["a", "b"]:
        toy_graph.save(Path(config["paths"]["index"]) / name)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cli, "_encoder", lambda config: tiny_encoder)
    seen = []
    monkeypatch.setattr(
        training, "pretrain", lambda config, graphs, resume=None: seen.append(set(graphs))
    )
    main(["--config", str(path), "pretrain"])
    assert seen == [{"a", "b"}]
    assert not (Path(config["paths"]["index"]) / "eval").exists()


def test_cli_inference_only_needs_evaluation_graph(toy_graph, small_config, tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from qgraphrag import cli

    config = small_config
    config["datasets"].update(root=str(tmp_path / "data"), train_names=["a", "b"], eval_name="eval")
    toy_graph.save(Path(config["paths"]["index"]) / "eval")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    def pipeline(config, graph, checkpoint):
        assert Path(config["paths"]["index"]).name == "eval"
        assert config["llm"]["model"] == "Qwen3-14B-AWQ"
        return SimpleNamespace(answer=lambda question: {"answer": "test answer"})

    monkeypatch.setattr(cli, "_pipeline", pipeline)
    main(["--config", str(path), "ask", "--checkpoint", "unused.pt", "--question", "test"])
    assert not (Path(config["paths"]["index"]) / "a").exists()


def test_builder_override_does_not_change_query_side_configuration(
    toy_graph,
    small_config,
    tmp_path,
    recording_backend,
    monkeypatch,
):
    from pathlib import Path

    import yaml

    from qgraphrag import cli, construction

    config = small_config
    config["datasets"].update(root=str(tmp_path / "data"), train_names=["a", "b"], eval_name="eval")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cli, "_backend", lambda config: recording_backend(lambda *args: {}))
    built = []

    def build(config, backend):
        built.append((Path(config["paths"]["index"]).name, config["llm"]["model"]))
        return toy_graph

    monkeypatch.setattr(construction, "build_index", build)
    main(["--config", str(path), "--set", "llm.model=Qwen3-8B", "build-index"])
    assert built == [("a", "Qwen3-8B"), ("b", "Qwen3-8B"), ("eval", "Qwen3-8B")]
    from qgraphrag.config import load_config

    assert load_config(path)["llm"]["model"] == "Qwen3-14B-AWQ"
