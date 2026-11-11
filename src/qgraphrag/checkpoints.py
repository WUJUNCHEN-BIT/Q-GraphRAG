"""Checkpoint serialization and GFM-RAG weight initialization."""

from __future__ import annotations

from pathlib import Path

import torch

from .gnn.model import QualityGNNRetriever
from .io import read_json, save_torch
from .schema import QUALITY_FEATURE_VERSION

TRAINED_STAGES = {"finetune"}


def save_checkpoint(
    path,
    model,
    encoder_signature,
    stage,
    graph_fingerprint=None,
    *,
    quality_feature_version=None,
    **training,
):
    if stage in TRAINED_STAGES and quality_feature_version != QUALITY_FEATURE_VERSION:
        raise ValueError("Quality-supervised checkpoints require the current feature version")
    save_torch(
        path,
        {
            "schema_version": 2,
            "quality_feature_version": quality_feature_version,
            "model_config": model.model_config,
            "embedding_dim": model.embedding_dim,
            "encoder_signature": encoder_signature,
            "stage": stage,
            "graph_fingerprint": graph_fingerprint,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            **training,
        },
    )


def load_checkpoint(path, device="cpu", require_trained_quality=False, *, for_initialization=False):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if for_initialization and require_trained_quality:
        raise ValueError("Backbone initialization cannot be used for quality-aware inference")
    supported = (1, 2) if for_initialization else (2,)
    if data.get("schema_version") not in supported or "state_dict" not in data:
        raise ValueError("Unsupported checkpoint schema; use import-gfm or backbone initialization")
    if require_trained_quality and data.get("stage") not in TRAINED_STAGES:
        raise ValueError("QA requires a quality-supervised finetune checkpoint")
    if (
        data.get("stage") in TRAINED_STAGES
        and not for_initialization
        and data.get("quality_feature_version") != QUALITY_FEATURE_VERSION
    ):
        raise ValueError(
            "Checkpoint quality feature version mismatch; joint fine-tuning is required"
        )
    model = QualityGNNRetriever(data["model_config"], data["embedding_dim"])
    model.load_state_dict(data["state_dict"], strict=True)
    return model.to(device), data


def initialize_from_pretrain(model, checkpoint):
    pretrained, metadata = load_checkpoint(checkpoint, for_initialization=True)
    if (
        pretrained.model_config != model.model_config
        or pretrained.embedding_dim != model.embedding_dim
    ):
        raise ValueError("Pretraining/backbone configuration mismatch")
    initialize_question = metadata["stage"] in {"pretrain", "imported_gfm_pretrain"}
    excluded = (
        ("quality_encoder.", "question_mlp.") if initialize_question else ("quality_encoder.",)
    )
    keys = {k: v for k, v in pretrained.state_dict().items() if not k.startswith(excluded)}
    result = model.load_state_dict(keys, strict=False)
    allowed = {k for k in model.state_dict() if k.startswith(excluded)}
    if set(result.missing_keys) != allowed or result.unexpected_keys:
        raise ValueError(f"Invalid pretraining initialization: {result}")
    if initialize_question:
        model.question_mlp.load_state_dict(model.rel_mlp.state_dict())
    return metadata


def import_gfm(source: str | Path, output: str | Path, config: dict):
    source = Path(source)
    config_file = source / "config.json" if source.is_dir() else source.parent / "config.json"
    if not config_file.exists():
        raise ValueError("GFM import requires the original pretrained/config.json")
    original = read_json(config_file)
    architecture = original["model_config"]["entity_model"]
    if (
        architecture.get("message_func") != "distmult"
        or architecture.get("aggregate_func") != "sum"
    ):
        raise ValueError("Only GFM DistMult+sum checkpoints are supported")
    supported = {
        "_target_",
        "input_dim",
        "hidden_dims",
        "message_func",
        "aggregate_func",
        "short_cut",
        "layer_norm",
        "concat_hidden",
        "num_mlp_layer",
        "checkpoint_path",
        "activation",
        "num_relation",
    }
    if set(architecture) - supported or architecture.get("activation", "relu") != "relu":
        raise ValueError("Unsupported GFM backbone configuration")
    model_config = dict(config["model"])
    for key in ("input_dim", "hidden_dims", "short_cut", "layer_norm", "concat_hidden"):
        if key in architecture:
            model_config[key] = architecture[key]
    model_config["num_mlp_layers"] = architecture.get("num_mlp_layer", 2)
    weights = source
    if source.is_dir():
        options = [
            source / name
            for name in ("model.pth", "model.pt", "pytorch_model.bin", "model_best.pth")
        ]
        options = [p for p in options if p.exists()]
        if len(options) != 1:
            raise ValueError(
                "Specify the original weight file explicitly when discovery is ambiguous"
            )
        weights = options[0]
    data = torch.load(weights, map_location="cpu", weights_only=True)
    state = data.get("model", data.get("state_dict", data))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain GFM model state dictionary")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    dim = original["model_config"]["rel_emb_dim"]
    model = QualityGNNRetriever(model_config, dim)
    result = model.load_state_dict(state, strict=False)
    allowed = {k for k in model.state_dict() if k.startswith("quality_encoder.")}
    is_retrieval = "question_mlp.weight" in state
    if not is_retrieval:
        allowed.update(k for k in model.state_dict() if k.startswith("question_mlp."))
        model.question_mlp.load_state_dict(model.rel_mlp.state_dict())
    if set(result.missing_keys) != allowed or result.unexpected_keys:
        raise ValueError(f"GFM import is not structurally compatible: {result}")
    encoder = original["text_emb_model_config"]
    signature = {
        "model": encoder["text_emb_model_name"],
        "normalize": encoder.get("normalize", False),
        "dimension": dim,
    }
    save_checkpoint(
        output,
        model,
        signature,
        "imported_gfm_retrieval" if is_retrieval else "imported_gfm_pretrain",
        upstream_config=original,
    )
    return {
        "stage": "imported_gfm",
        "backbone_parameters_loaded": len(state),
        "quality_encoder_requires_finetune": True,
        "model_config": model_config,
    }
