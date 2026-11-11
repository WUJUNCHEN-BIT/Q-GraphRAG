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


def pretrain(config, graph, resume=None):
    seed_everything(config["seed"])
    settings = config["pretrain"]
    directory = Path(config["paths"]["output"])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = None
    if resume:
        model, metadata = load_checkpoint(resume, config["device"])
        _check_encoder(metadata, graph)
        if (
            metadata["stage"] != "pretrain"
            or metadata["graph_fingerprint"] != graph.graph_fingerprint
        ):
            raise ValueError("Resume requires a pretraining checkpoint of the same graph")
    else:
        model = QualityGNNRetriever(config["model"], graph.relation_embeddings.shape[1]).to(
            config["device"]
        )
    _save_run_config(directory, config, model)
    optimizer = _optimizer(model, settings)
    start_epoch = 0
    if metadata:
        optimizer.load_state_dict(metadata["optimizer"])
        start_epoch = metadata["epoch"]
    sampler = StrictNegativeSampler(graph)
    cpu_edges, cpu_types = graph.edge_index.cpu(), graph.edge_type.cpu()
    graph = graph.to(config["device"])
    best_loss = metadata.get("best_loss", math.inf) if metadata else math.inf
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(cpu_types.numel(), generator=generator)
        if settings["steps_per_epoch"]:
            order = order[: settings["steps_per_epoch"] * settings["batch_size"]]
        losses = []
        for left in tqdm(
            range(0, len(order), settings["batch_size"]), desc=f"KG epoch {epoch + 1}"
        ):
            ids = order[left : left + settings["batch_size"]]
            source, positive = cpu_edges[:, ids]
            relation = cpu_types[ids]
            negatives = torch.stack(
                [
                    sampler.sample(h, r, settings["num_negative"], generator)
                    for h, r in zip(source.tolist(), relation.tolist())
                ]
            )
            candidates = torch.cat((positive.unsqueeze(1), negatives), dim=1).to(config["device"])
            blocked = graph.edge_to_final[ids.to(config["device"])].unique()
            optimizer.zero_grad(set_to_none=True)
            logits = model.completion(
                graph,
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
        mean_loss = sum(x * n for x, n in losses) / sum(n for _, n in losses)
        improved = mean_loss < best_loss
        best_loss = min(best_loss, mean_loss)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "quality_feature_version": graph.quality_feature_version,
        }
        save_checkpoint(
            directory / "last.pt",
            model,
            graph.encoder_signature,
            "pretrain",
            graph.graph_fingerprint,
            **extra,
        )
        if improved:
            save_checkpoint(
                directory / "best.pt",
                model,
                graph.encoder_signature,
                "pretrain",
                graph.graph_fingerprint,
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
    settings = config["finetune"]
    directory = Path(config["paths"]["output"])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = None
    if resume:
        model, metadata = load_checkpoint(resume, config["device"])
        _check_encoder(metadata, graph)
        if (
            metadata["stage"] != "finetune"
            or metadata["graph_fingerprint"] != graph.graph_fingerprint
            or metadata.get("training_fingerprint") != train_data["fingerprint"]
        ):
            raise ValueError("Fine-tuning resume graph/training data/stage mismatch")
    else:
        previous, prior = load_checkpoint(pretrained, for_initialization=True)
        _check_encoder(prior, graph)
        model = QualityGNNRetriever(previous.model_config, previous.embedding_dim)
        initialize_from_pretrain(model, pretrained)
        model = model.to(config["device"])
    _save_run_config(directory, config, model)
    optimizer = _optimizer(model, settings)
    if metadata:
        optimizer.load_state_dict(metadata["optimizer"])
    start_epoch = metadata["epoch"] if metadata else 0
    best_loss = metadata.get("best_loss", math.inf) if metadata else math.inf
    graph = graph.to(config["device"])
    quality_gradient_seen = False
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(train_data["samples"]), generator=generator).tolist()
        if settings["steps_per_epoch"]:
            order = order[: settings["steps_per_epoch"] * settings["batch_size"]]
        losses = []
        for left in tqdm(
            range(0, len(order), settings["batch_size"]), desc=f"QA epoch {epoch + 1}"
        ):
            indices = order[left : left + settings["batch_size"]]
            questions, seeds, labels = query_batch(
                train_data, indices, graph.num_nodes, config["device"]
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph, questions, seeds).logits
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
        train_loss = sum(x * n for x, n in losses) / sum(n for _, n in losses)
        score = validation_loss(model, graph, valid_data, settings) if valid_data else train_loss
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
            "training_fingerprint": train_data["fingerprint"],
            "selection_metric": "validation_loss" if valid_data else "training_loss",
            "quality_gradient_seen": quality_gradient_seen,
            "quality_feature_version": graph.quality_feature_version,
        }
        save_checkpoint(
            directory / "last.pt",
            model,
            graph.encoder_signature,
            "finetune",
            graph.graph_fingerprint,
            **extra,
        )
        if improved:
            save_checkpoint(
                directory / "best.pt",
                model,
                graph.encoder_signature,
                "finetune",
                graph.graph_fingerprint,
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
                "quality_gradient_seen": quality_gradient_seen,
            },
        )
    if start_epoch < settings["epochs"] and not quality_gradient_seen:
        raise RuntimeError(
            "No quality-encoder gradient observed; check query seeds/graph supervision"
        )
    model.eval()
    model.cache_gates(graph, directory / "last_gates.pt")
    best_model, _ = load_checkpoint(directory / "best.pt", config["device"])
    best_model.eval().cache_gates(graph, directory / "best_gates.pt")
    return model
