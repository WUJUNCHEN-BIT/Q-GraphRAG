"""Stage 2: KG-completion pretraining, then quality-supervised retrieval fine-tuning."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoints import initialize_from_pretrain, load_checkpoint, save_checkpoint
from .gnn.losses import completion_loss, retrieval_loss
from .gnn.model import QualityGNNRetriever
from .io import fingerprint, read_json, redact, save_torch, write_json


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def support_document_ids(record, graph):
    ids = {d.id for d in graph.documents}
    titles = {}
    for doc in graph.documents:
        titles.setdefault(doc.title, []).append(doc.id)
    support = record.get("supporting_facts", record.get("supporting_documents"))
    if not isinstance(support, list) or not support:
        raise ValueError(f"Query {record.get('id')} requires annotated supporting documents")
    result = set()
    for item in support:
        key = item[0] if isinstance(item, (list, tuple)) else item
        if key in ids:
            result.add(key)
        elif key in titles and len(titles[key]) == 1:
            result.add(titles[key][0])
        else:
            raise ValueError(f"Supporting document {key!r} missing/ambiguous in the indexed corpus")
    return sorted(result)


def prepare_queries(path, graph, encoder, linker, cache_dir):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("Training/validation queries must be a nonempty JSON list")
    key = fingerprint(
        {
            "records": records,
            "graph": graph.fingerprint,
            "encoder": encoder.signature,
            "linker": linker.signature,
            "schema": 2,
        }
    )
    cache = Path(cache_dir) / ("queries_" + key + ".pt")
    if cache.exists():
        data = torch.load(cache, weights_only=True, map_location="cpu")
        if data.get("fingerprint") != key:
            raise ValueError("Invalid query cache fingerprint")
        return data
    entity_ids = {name: i for i, name in enumerate(graph.entities)}
    samples = []
    for row in tqdm(records, desc="Prepare query labels/seeds"):
        support = support_document_ids(row, graph)
        positive = sorted({entity_ids[e] for doc in support for e in graph.document_entities[doc]})
        if not positive or len(positive) == len(graph.entities):
            raise ValueError(
                f"Query {row.get('id')} has no usable positive/negative entity partition"
            )
        samples.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "seed_ids": linker.link(row["question"], row.get("question_entities")),
                "positive_ids": positive,
            }
        )
    if len({x["id"] for x in samples}) != len(samples):
        raise ValueError("Duplicate training query IDs")
    data = {
        "fingerprint": key,
        "samples": samples,
        "question_embeddings": encoder.encode([x["question"] for x in samples]),
    }
    save_torch(cache, data)
    return data


def query_batch(data, indices, num_nodes, device):
    questions = data["question_embeddings"][indices].to(device)
    seeds = torch.zeros((len(indices), num_nodes), device=device)
    labels = torch.zeros_like(seeds)
    for i, index in enumerate(indices):
        row = data["samples"][index]
        seeds[i, row["seed_ids"]] = 1
        labels[i, row["positive_ids"]] = 1
    return questions, seeds, labels


class StrictNegativeSampler:
    def __init__(self, graph):
        self.num_nodes = graph.num_nodes
        self.truth = {}
        for (source, target), relation in zip(
            graph.edge_index.cpu().t().tolist(), graph.edge_type.cpu().tolist()
        ):
            self.truth.setdefault((source, relation), set()).add(target)

    def sample(self, source, relation, count, generator):
        forbidden = self.truth[(source, relation)]
        if len(forbidden) == self.num_nodes:
            raise ValueError("A completion query has no valid negative entities")
        if len(forbidden) > self.num_nodes // 2:
            allowed = torch.tensor(sorted(set(range(self.num_nodes)) - forbidden))
            return allowed[torch.randint(len(allowed), (count,), generator=generator)]
        forbidden_tensor = torch.tensor(sorted(forbidden))
        kept = []
        remaining = count
        while remaining:
            proposed = torch.randint(self.num_nodes, (max(remaining * 2, 16),), generator=generator)
            valid = proposed[~torch.isin(proposed, forbidden_tensor)][:remaining]
            kept.append(valid)
            remaining -= valid.numel()
        return torch.cat(kept)


def _optimizer(model, settings):
    return torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )


def _check_encoder(metadata, graph):
    if metadata["encoder_signature"] != graph.encoder_signature:
        raise ValueError("Checkpoint and graph use different frozen text encoders/configurations")


def _save_run_config(directory, config, model):
    effective = copy.deepcopy(config)
    effective["model"] = model.model_config
    write_json(Path(directory) / "resolved_config.json", redact(effective))


def _log(directory, row):
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (Path(directory) / "training.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _graphs(graph):
    items = graph if isinstance(graph, dict) else {"graph": graph}
    if not items:
        raise ValueError("Training requires at least one graph")
    graphs = {name: items[name].to("cpu") for name in sorted(items)}
    first = next(iter(graphs.values()))
    for value in graphs.values():
        value.validate()
        if not value.num_edges or not value.num_nodes:
            raise ValueError("Cannot train on an empty graph")
        if value.encoder_signature != first.encoder_signature:
            raise ValueError("Training graphs use different text encoders")
        if value.relation_embeddings.shape[1] != first.relation_embeddings.shape[1]:
            raise ValueError("Training graph embedding dimensions differ")
    return graphs


def _query_sets(data, graphs, *, validation=False):
    if data is None:
        return None
    if "samples" in data and "question_embeddings" in data:
        if len(graphs) != 1:
            raise ValueError("Provide graph-specific query sets for multi-graph training")
        data = {next(iter(graphs)): data}
    if not isinstance(data, dict) or not data:
        raise ValueError("Missing query sets")
    if not set(data) <= set(graphs) or (not validation and set(data) != set(graphs)):
        raise ValueError("Query sets do not match the training graphs")
    seen = set()
    for name, records in data.items():
        if not records["samples"]:
            raise ValueError(f"Empty query set: {name}")
        ids = [row["id"] for row in records["samples"]]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Training graphs contain duplicate query IDs")
        seen.update(ids)
    return {name: data[name] for name in graphs if name in data}


def _fingerprints(graphs, data=None):
    result = {
        "graph_fingerprints": {name: graph.graph_fingerprint for name, graph in graphs.items()}
    }
    if data is not None:
        result["training_fingerprints"] = {name: rows["fingerprint"] for name, rows in data.items()}
    return result


def _check_resume(metadata, graphs, stage, data=None):
    if metadata["stage"] != stage:
        raise ValueError("Checkpoint training stage mismatch")
    expected = _fingerprints(graphs, data)
    for key, values in expected.items():
        saved = metadata.get(key)
        if saved is None and len(graphs) == 1:
            legacy_key = (
                "graph_fingerprint" if key == "graph_fingerprints" else "training_fingerprint"
            )
            saved = {next(iter(graphs)): metadata.get(legacy_key)}
        if saved != values:
            raise ValueError("Resume graph/training data mismatch")
    _check_encoder(metadata, next(iter(graphs.values())))


def _batch_plan(sizes, batch_size, limit, generator):
    counts = {name: math.ceil(size / batch_size) for name, size in sizes.items()}
    total = sum(counts.values())
    selected = None
    if limit is not None and limit < total:
        selected = set(torch.randperm(total, generator=generator)[:limit].tolist())
    plan, offset = {}, 0
    for name, count in counts.items():
        plan[name] = (
            range(count)
            if selected is None
            else [i for i in range(count) if offset + i in selected]
        )
        offset += count
    names = list(sizes)
    order = [names[i] for i in torch.randperm(len(names), generator=generator).tolist()]
    return order, plan


def pretrain(config, graph, resume=None):
    seed_everything(config["seed"])
    graphs = _graphs(graph)
    first = next(iter(graphs.values()))
    settings = config["pretrain"]
    directory = Path(config["paths"]["output"])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = None
    if resume:
        model, metadata = load_checkpoint(resume, config["device"])
        _check_resume(metadata, graphs, "pretrain")
    else:
        model = QualityGNNRetriever(config["model"], first.relation_embeddings.shape[1]).to(
            config["device"]
        )
    _save_run_config(directory, config, model)
    optimizer = _optimizer(model, settings)
    if metadata:
        optimizer.load_state_dict(metadata["optimizer"])
    start_epoch = metadata["epoch"] if metadata else 0
    best_loss = metadata.get("best_loss", math.inf) if metadata else math.inf
    samplers = {name: StrictNegativeSampler(value) for name, value in graphs.items()}
    sizes = {name: value.edge_type.numel() for name, value in graphs.items()}
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        names, plan = _batch_plan(
            sizes, settings["batch_size"], settings["steps_per_epoch"], generator
        )
        losses, examples = [], {name: 0 for name in graphs}
        for name in names:
            if not len(plan[name]):
                continue
            cpu_graph = graphs[name]
            active = cpu_graph.to(config["device"])
            order = torch.randperm(sizes[name], generator=generator)
            for batch_index in tqdm(plan[name], desc=f"KG epoch {epoch + 1}: {name}"):
                left = batch_index * settings["batch_size"]
                ids = order[left : left + settings["batch_size"]]
                source, positive = cpu_graph.edge_index[:, ids]
                relation = cpu_graph.edge_type[ids]
                negatives = torch.stack(
                    [
                        samplers[name].sample(h, r, settings["num_negative"], generator)
                        for h, r in zip(source.tolist(), relation.tolist())
                    ]
                )
                candidates = torch.cat((positive.unsqueeze(1), negatives), dim=1).to(
                    config["device"]
                )
                blocked = cpu_graph.edge_to_final[ids].unique().to(config["device"])
                optimizer.zero_grad(set_to_none=True)
                logits = model.completion(
                    active,
                    source.to(config["device"]),
                    relation.to(config["device"]),
                    candidates,
                    blocked,
                )
                loss = completion_loss(logits, settings["adversarial_temperature"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite completion loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                losses.append((float(loss.detach()), len(ids)))
                examples[name] += len(ids)
            del active
        mean_loss = sum(value * count for value, count in losses) / sum(
            count for _, count in losses
        )
        improved = mean_loss < best_loss
        best_loss = min(best_loss, mean_loss)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "quality_feature_version": first.quality_feature_version,
            **_fingerprints(graphs),
        }
        for filename in ["last.pt", "best.pt"] if improved else ["last.pt"]:
            save_checkpoint(
                directory / filename,
                model,
                first.encoder_signature,
                "pretrain",
                first.graph_fingerprint if len(graphs) == 1 else None,
                **extra,
            )
        _log(
            directory,
            {
                "stage": "pretrain",
                "epoch": epoch + 1,
                "loss": mean_loss,
                "quality_enabled": False,
                "steps": len(losses),
                "examples_by_graph": examples,
            },
        )
    return model


@torch.no_grad()
def validation_loss(model, graph, data, settings):
    model.eval()
    losses = []
    for left in range(0, len(data["samples"]), settings["batch_size"]):
        indices = list(range(left, min(left + settings["batch_size"], len(data["samples"]))))
        questions, seeds, labels = query_batch(data, indices, graph.num_nodes, graph.quality.device)
        logits = model(graph, questions, seeds).logits
        losses.append((float(retrieval_loss(logits, labels, settings)), len(indices)))
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


def finetune(config, graph, train_data, valid_data, pretrained=None, resume=None):
    if not pretrained and not resume:
        raise ValueError(
            "Fine-tuning requires --pretrained or --resume; stage-one training cannot be skipped"
        )
    seed_everything(config["seed"])
    graphs = _graphs(graph)
    first = next(iter(graphs.values()))
    training = _query_sets(train_data, graphs)
    validation = _query_sets(valid_data, graphs, validation=True)
    if validation:
        train_ids = {row["id"] for data in training.values() for row in data["samples"]}
        valid_ids = {row["id"] for data in validation.values() for row in data["samples"]}
        if train_ids & valid_ids:
            raise ValueError("Training and validation query IDs overlap")
    settings = config["finetune"]
    directory = Path(config["paths"]["output"])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = None
    if resume:
        model, metadata = load_checkpoint(resume, config["device"])
        _check_resume(metadata, graphs, "finetune", training)
    else:
        previous, prior = load_checkpoint(pretrained, for_initialization=True)
        _check_encoder(prior, first)
        model = QualityGNNRetriever(previous.model_config, previous.embedding_dim)
        initialize_from_pretrain(model, pretrained)
        model = model.to(config["device"])
    _save_run_config(directory, config, model)
    optimizer = _optimizer(model, settings)
    if metadata:
        optimizer.load_state_dict(metadata["optimizer"])
    start_epoch = metadata["epoch"] if metadata else 0
    best_loss = metadata.get("best_loss", math.inf) if metadata else math.inf
    quality_gradient_seen = metadata.get("quality_gradient_seen", False) if metadata else False
    sizes = {name: len(data["samples"]) for name, data in training.items()}
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        names, plan = _batch_plan(
            sizes, settings["batch_size"], settings["steps_per_epoch"], generator
        )
        losses, examples = [], {name: 0 for name in graphs}
        for name in names:
            if not len(plan[name]):
                continue
            active = graphs[name].to(config["device"])
            order = torch.randperm(sizes[name], generator=generator).tolist()
            for batch_index in tqdm(plan[name], desc=f"QA epoch {epoch + 1}: {name}"):
                left = batch_index * settings["batch_size"]
                indices = order[left : left + settings["batch_size"]]
                questions, seeds, labels = query_batch(
                    training[name], indices, active.num_nodes, config["device"]
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(active, questions, seeds).logits
                loss = retrieval_loss(logits, labels, settings)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite retrieval loss")
                loss.backward()
                quality_gradient_seen |= any(
                    p.grad is not None and bool(p.grad.abs().sum() > 0)
                    for p in model.quality_encoder.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                losses.append((float(loss.detach()), len(indices)))
                examples[name] += len(indices)
            del active
        train_loss = sum(value * count for value, count in losses) / sum(
            count for _, count in losses
        )
        score = train_loss
        if validation:
            values = []
            for name, data in validation.items():
                active = graphs[name].to(config["device"])
                values.append(
                    (validation_loss(model, active, data, settings), len(data["samples"]))
                )
                del active
            score = sum(value * count for value, count in values) / sum(
                count for _, count in values
            )
        improved = score < best_loss
        best_loss = min(score, best_loss)
        if not quality_gradient_seen:
            raise RuntimeError(
                "No quality-encoder gradient observed; check query seeds/graph supervision"
            )
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "training_fingerprint": next(iter(training.values()))["fingerprint"]
            if len(graphs) == 1
            else None,
            "selection_metric": "validation_loss" if validation else "training_loss",
            "quality_gradient_seen": quality_gradient_seen,
            "quality_feature_version": first.quality_feature_version,
            **_fingerprints(graphs, training),
        }
        for filename in ["last.pt", "best.pt"] if improved else ["last.pt"]:
            save_checkpoint(
                directory / filename,
                model,
                first.encoder_signature,
                "finetune",
                first.graph_fingerprint if len(graphs) == 1 else None,
                **extra,
            )
        _log(
            directory,
            {
                "stage": "finetune",
                "epoch": epoch + 1,
                "loss": train_loss,
                "selection_loss": score,
                "steps": len(losses),
                "examples_by_graph": examples,
                "quality_gradient_seen": quality_gradient_seen,
            },
        )
    model.eval()
    if len(graphs) == 1:
        active = first.to(config["device"])
        model.cache_gates(active, directory / "last_gates.pt")
        if (directory / "best.pt").exists():
            best_model, _ = load_checkpoint(directory / "best.pt", config["device"])
            best_model.eval().cache_gates(active, directory / "best_gates.pt")
    return model
