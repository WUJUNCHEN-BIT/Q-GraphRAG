"""Configuration loading and default hyperparameters."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS = {
    "seed": 1024,
    "device": "cuda:0",
    "datasets": {"root": None, "train_names": [], "eval_name": "eval"},
    "paths": {
        "corpus": "data/dataset_corpus.json",
        "index": "artifacts/index",
        "train": "data/train.json",
        "valid": None,
        "output": "outputs/run",
    },
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen3-14B-AWQ",
        "api_key_env": "QGRAPHRAG_LLM_API_KEY",
        "timeout": 180.0,
        "retries": 3,
        "temperature": 0.0,
        "max_tokens": 4096,
        "json_mode": True,
        "disable_thinking": True,
        "cache_dir": "cache/llm",
    },
    "embedding": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "batch_size": 32,
        "normalize": False,
        "device": "cpu",
        "local_files_only": False,
    },
    "construction": {"chunk_chars": 2400, "overlap_chars": 200, "workers": 4, "aliases": None},
    "model": {
        "input_dim": 512,
        "hidden_dims": [512, 512, 512, 512, 512, 512],
        "short_cut": True,
        "layer_norm": True,
        "concat_hidden": False,
        "num_mlp_layers": 2,
        "quality_hidden_dim": 16,
        "edge_chunk_size": 8192,
    },
    "pretrain": {
        "epochs": 10,
        "batch_size": 4,
        "num_negative": 256,
        "adversarial_temperature": 1.0,
        "lr": 0.0005,
        "weight_decay": 0.01,
        "steps_per_epoch": None,
        "max_grad_norm": 1.0,
    },
    "finetune": {
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.0005,
        "weight_decay": 0.01,
        "adversarial_temperature": 0.2,
        "lambda_bce": 0.3,
        "lambda_rank": 0.7,
        "steps_per_epoch": None,
        "max_grad_norm": 1.0,
    },
    "retrieval": {
        "entity_top_k": 20,
        "graph_doc_top_k": 10,
        "bm25_top_k": 50,
        "vector_top_k": 50,
        "text_top_k": 10,
        "rerank_batch_size": 32,
        "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "reranker_device": "cpu",
        "final_top_k": 5,
        "linking_top_k": 1,
        "max_supplementation_rounds": 2,
    },
    "pcst": {
        "top_m": 5,
        "hops": 2,
        "alpha_s": 0.5,
        "c_min": 0.05,
        "eta": 0.5,
        "lambda_pcst": 0.5,
        "epsilon": 1e-8,
        "pruning": "gw",
    },
}


def _merge(target: dict, incoming: dict, prefix: str = "") -> None:
    for key, value in incoming.items():
        name = prefix + key
        if key not in target:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            _merge(target[key], value, name + ".")
        else:
            target[key] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if path:
        text = os.path.expandvars(Path(path).read_text(encoding="utf-8-sig"))
        if re.search(r"\$\{[^}]+\}", text):
            raise ValueError("Unresolved environment variable in configuration")
        incoming = yaml.safe_load(text) or {}
        if not isinstance(incoming, dict):
            raise ValueError("Configuration must be a YAML mapping")
        _merge(config, incoming)
    environment = {
        "QGRAPHRAG_LLM_BASE_URL": ("llm", "base_url"),
        "QGRAPHRAG_LLM_MODEL": ("llm", "model"),
        "QGRAPHRAG_EMBEDDING_MODEL": ("embedding", "model"),
        "QGRAPHRAG_RERANKER_MODEL": ("retrieval", "reranker"),
    }
    for variable, (section, key) in environment.items():
        if os.environ.get(variable):
            config[section][key] = os.environ[variable]
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Expected section.key=value: {override}")
        key, value = override.split("=", 1)
        item: Any = yaml.safe_load(value)
        for part in reversed(key.split(".")):
            item = {part: item}
        _merge(config, item)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    datasets = config["datasets"]
    names = datasets["train_names"]
    if (
        not isinstance(names, list)
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("datasets.train_names must contain unique graph names")
    evaluation = datasets["eval_name"]
    if not isinstance(evaluation, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", evaluation
    ):
        raise ValueError("Invalid datasets.eval_name")
    if evaluation in names:
        raise ValueError("The evaluation graph cannot be a training graph")
    if datasets["root"] is not None:
        if not isinstance(datasets["root"], str) or not datasets["root"].strip():
            raise ValueError("datasets.root must be a directory path")
        if config["paths"]["valid"] is not None:
            raise ValueError("paths.valid is only supported in single-graph configurations")
    elif names:
        raise ValueError("datasets.train_names requires datasets.root")
    chunk = config["construction"]
    if chunk["chunk_chars"] <= 0 or not 0 <= chunk["overlap_chars"] < chunk["chunk_chars"]:
        raise ValueError("Require 0 <= overlap_chars < chunk_chars")
    model = config["model"]
    if not model["hidden_dims"] or any(x != model["input_dim"] for x in model["hidden_dims"]):
        raise ValueError("GFM DistMult layers require equal input and hidden dimensions")
    if min(model["input_dim"], model["quality_hidden_dim"], model["edge_chunk_size"]) <= 0:
        raise ValueError("Model dimensions and edge_chunk_size must be positive")
    pcst = config["pcst"]
    if not 0 <= pcst["alpha_s"] <= 1 or pcst["hops"] < 0 or pcst["top_m"] <= 0:
        raise ValueError("Invalid PCST neighborhood or support parameters")
    if pcst["c_min"] <= 0 or pcst["eta"] < 0 or pcst["lambda_pcst"] < 0 or pcst["epsilon"] <= 0:
        raise ValueError("Invalid PCST costs or epsilon")
    for section in ("pretrain", "finetune"):
        item = config[section]
        if item["epochs"] < 1 or item["batch_size"] < 1 or item["lr"] <= 0:
            raise ValueError(f"Invalid {section} training configuration")
        if item["steps_per_epoch"] is not None and item["steps_per_epoch"] < 1:
            raise ValueError("steps_per_epoch must be positive or null")
    if config["pretrain"]["num_negative"] < 1:
        raise ValueError("num_negative must be positive")
    for key in (
        "entity_top_k",
        "graph_doc_top_k",
        "bm25_top_k",
        "vector_top_k",
        "text_top_k",
        "final_top_k",
        "linking_top_k",
    ):
        if config["retrieval"][key] < 1:
            raise ValueError(f"retrieval.{key} must be positive")
    rounds = config["retrieval"]["max_supplementation_rounds"]
    if type(rounds) is not int or rounds < 0:
        raise ValueError("retrieval.max_supplementation_rounds must be a nonnegative integer")
