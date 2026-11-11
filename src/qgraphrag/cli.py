"""Commands for the three stages and their inspectable, resumable artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import load_config
from .io import atomic_text, file_sha256, fingerprint, read_json, redact, write_json
from .schema import GraphIndex


def _parser():
    parser = argparse.ArgumentParser(prog="qgraphrag", description="Quality-aware graph RAG")
    parser.add_argument(
        "--config", help="YAML configuration; paths are relative to the current directory"
    )
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "build-index", help="Stage 1: construct verified provenance-aware KG and quality features"
    )
    commands.add_parser("prepare", help="Encode relations and build graph tensors")
    pre = commands.add_parser("pretrain", help="Stage 2a: graph completion with fixed gates")
    pre.add_argument("--resume")
    fine = commands.add_parser("finetune", help="Stage 2b: joint quality-aware retrieval training")
    group = fine.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained")
    group.add_argument("--resume")
    cache = commands.add_parser("cache-gates", help="Cache the trained graph-specific coefficients")
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--output", required=True)
    ask = commands.add_parser("ask", help="Stage 3: retrieve, assess, supplement, select, answer")
    ask.add_argument("--checkpoint", required=True)
    ask.add_argument("--question", required=True)
    predict = commands.add_parser(
        "predict", help="Answer a QA JSON list; writes one atomic record per query"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--queries", required=True)
    predict.add_argument("--limit", type=int)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate final document recall and answer EM/F1"
    )
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--output", required=True)
    imported = commands.add_parser(
        "import-gfm", help="Import a compatible original GFM-RAG pretrained export"
    )
    imported.add_argument("--source", required=True)
    imported.add_argument("--output", required=True)
    commands.add_parser(
        "inspect-index", help="Validate graph fingerprints, quality features and provenance"
    )
    return parser


def _encoder(config):
    from .embeddings import SentenceEncoder

    return SentenceEncoder(config["embedding"])


def _backend(config):
    from .llm import OpenAIJsonLLM

    return OpenAIJsonLLM(config["llm"])


def _graph_tensors(config, graph, encoder):
    from .tensors import prepare_tensors

    return prepare_tensors(graph, encoder, config["paths"]["index"])


def _pipeline(config, graph, checkpoint):
    from .checkpoints import load_checkpoint
    from .linking import EntityLinker
    from .pipeline import QGraphRAG
    from .retrieval import CrossEncoderReranker, GraphRetriever, TextRetriever

    model, metadata = load_checkpoint(checkpoint, config["device"], require_trained_quality=True)
    encoder, backend = _encoder(config), _backend(config)
    if metadata["encoder_signature"] != encoder.signature:
        raise ValueError("QA encoder configuration differs from the checkpoint")
    tensors = _graph_tensors(config, graph, encoder).to(config["device"])
    linker = EntityLinker(graph, encoder, backend, config["retrieval"]["linking_top_k"])
    graph_retriever = GraphRetriever(graph, tensors, model, encoder, linker, config)
    # This cache is fingerprinted by both the graph and quality-encoder weights.
    gate_cache = (
        Path(config["paths"]["index"])
        / "cache"
        / ("gates_" + graph.fingerprint + "_" + model.quality_fingerprint() + ".pt")
    )
    if gate_cache.exists():
        model.load_gate_cache(tensors, gate_cache)
    else:
        model.cache_gates(tensors, gate_cache)
    text_retriever = TextRetriever(
        graph,
        encoder,
        CrossEncoderReranker(config["retrieval"]),
        config["retrieval"],
        Path(config["paths"]["index"]) / "cache",
    )
    return QGraphRAG(
        graph,
        graph_retriever,
        text_retriever,
        backend,
        config["retrieval"]["final_top_k"],
        max_supplementation_rounds=config["retrieval"]["max_supplementation_rounds"],
    )


def _predict(config, graph, args):
    rows = read_json(args.queries)
    if not isinstance(rows, list) or not rows:
        raise ValueError("--queries must be a nonempty JSON list")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate query IDs")
    directory = Path(config["paths"]["output"])
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 4,
        "quality_feature_version": graph.quality_feature_version,
        "graph_fingerprint": graph.fingerprint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "configuration": redact(config),
        "queries_fingerprint": fingerprint(rows),
    }
    manifest_file = directory / "prediction_manifest.json"
    if manifest_file.exists() and read_json(manifest_file) != manifest:
        raise ValueError("Output directory belongs to a different run; choose a new paths.output")
    write_json(manifest_file, manifest)
    pipeline = None
    predictions = []
    for row in rows:
        path = directory / "predictions" / (fingerprint(str(row["id"])) + ".json")
        if path.exists():
            prediction = read_json(path)
            if prediction["id"] != str(row["id"]) or prediction["question"] != row["question"]:
                raise ValueError("Prediction cache does not match query")
        else:
            if pipeline is None:
                pipeline = _pipeline(config, graph, args.checkpoint)
            # Only question-side fields are passed into inference. Gold answers/supports stay here.
            prediction = pipeline.answer(
                row["question"], row.get("question_entities"), str(row["id"])
            )
            write_json(path, prediction)
        predictions.append(prediction)
        print(
            json.dumps(
                {
                    "id": prediction["id"],
                    "answer": prediction["answer"],
                    "documents": prediction["retrieved_document_ids"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    atomic_text(
        directory / "predictions.jsonl",
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
    )
    if all(
        "answer" in row and ("supporting_facts" in row or "supporting_documents" in row)
        for row in rows
    ):
        from .metrics import evaluate_predictions

        metrics = evaluate_predictions(predictions, rows, graph, config["retrieval"]["final_top_k"])
        write_json(directory / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False))


def _graph_config(config, name=None):
    result = copy.deepcopy(config)
    if config["datasets"]["root"] is None:
        return result
    name = name or config["datasets"]["eval_name"]
    raw = Path(config["datasets"]["root"]) / name / "raw"
    result["paths"].update(
        corpus=str(raw / "dataset_corpus.json"),
        index=str(Path(config["paths"]["index"]) / name),
        train=str(raw / "train.json") if name in config["datasets"]["train_names"] else None,
        valid=None,
    )
    return result


def _training_configs(config):
    if config["datasets"]["root"] is None:
        return {"graph": _graph_config(config)}
    names = config["datasets"]["train_names"]
    if not names:
        raise ValueError("Training requires datasets.train_names")
    return {name: _graph_config(config, name) for name in names}


def _index_configs(config):
    if config["datasets"]["root"] is None:
        return {"graph": _graph_config(config)}
    names = config["datasets"]["train_names"] + [config["datasets"]["eval_name"]]
    return {name: _graph_config(config, name) for name in names}


def main(argv=None):
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.set)
    if args.command == "import-gfm":
        from .checkpoints import import_gfm

        print(json.dumps(import_gfm(args.source, args.output, config), indent=2))
        return
    if args.command == "build-index":
        from .construction import build_index

        backend = _backend(config)
        for name, settings in _index_configs(config).items():
            graph = build_index(settings, backend)
            print(
                json.dumps(
                    {
                        "graph": name,
                        "documents": len(graph.documents),
                        "entities": len(graph.entities),
                        "edges": len(graph.edges),
                        "fingerprint": graph.fingerprint,
                    }
                )
            )
        return
    if args.command == "prepare":
        encoder = _encoder(config)
        for name, settings in _index_configs(config).items():
            graph = GraphIndex.load(settings["paths"]["index"])
            tensors = _graph_tensors(settings, graph, encoder)
            print(
                json.dumps(
                    {
                        "graph": name,
                        "nodes": tensors.num_nodes,
                        "edges": tensors.num_edges,
                        "direct_relations": tensors.num_direct_relations,
                    }
                )
            )
        return
    if args.command in ("pretrain", "finetune"):
        from .linking import EntityLinker
        from .training import finetune, prepare_queries, pretrain

        encoder = _encoder(config)
        backend = _backend(config) if args.command == "finetune" else None
        graphs, training, validation = {}, {}, {}
        for name, settings in _training_configs(config).items():
            graph = GraphIndex.load(settings["paths"]["index"])
            graphs[name] = _graph_tensors(settings, graph, encoder)
            if args.command == "finetune":
                linker = EntityLinker(graph, encoder, backend, config["retrieval"]["linking_top_k"])
                cache = Path(settings["paths"]["index"]) / "cache"
                training[name] = prepare_queries(
                    settings["paths"]["train"], graph, encoder, linker, cache
                )
                if settings["paths"]["valid"]:
                    validation[name] = prepare_queries(
                        settings["paths"]["valid"], graph, encoder, linker, cache
                    )
                del linker
            del graph
        if args.command == "pretrain":
            pretrain(config, graphs, resume=args.resume)
        else:
            finetune(config, graphs, training, validation or None, args.pretrained, args.resume)
        return
    config = _graph_config(config)
    graph = GraphIndex.load(config["paths"]["index"])
    if args.command == "inspect-index":
        print(
            json.dumps(
                {
                    "documents": len(graph.documents),
                    "entities": len(graph.entities),
                    "edges": len(graph.edges),
                    "fingerprint": graph.fingerprint,
                    "provenance_valid": True,
                },
                indent=2,
            )
        )
    elif args.command == "cache-gates":
        from .checkpoints import load_checkpoint

        encoder = _encoder(config)
        tensors = _graph_tensors(config, graph, encoder).to(config["device"])
        model, metadata = load_checkpoint(
            args.checkpoint, config["device"], require_trained_quality=True
        )
        if metadata["encoder_signature"] != encoder.signature:
            raise ValueError("Encoder configuration mismatch")
        model.eval().cache_gates(tensors, args.output)
    elif args.command == "ask":
        pipeline = _pipeline(config, graph, args.checkpoint)
        result = pipeline.answer(args.question)
        directory = Path(config["paths"]["output"])
        write_json(directory / "answer.json", result)
        write_json(directory / "resolved_config.json", redact(config))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        _predict(config, graph, args)
    elif args.command == "evaluate":
        from .metrics import evaluate_predictions

        path = Path(args.predictions)
        predictions = (
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if path.suffix == ".jsonl"
            else read_json(path)
        )
        metrics = evaluate_predictions(predictions, read_json(args.queries), graph, args.k)
        write_json(args.output, metrics)
        print(json.dumps(metrics, indent=2))
