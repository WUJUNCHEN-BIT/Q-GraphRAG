"""Shared frozen text encoder with serialized access for parallel graph/text retrieval."""

from __future__ import annotations

import threading

import torch


class SentenceEncoder:
    def __init__(self, config: dict):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(
            config["model"], device=config["device"], local_files_only=config["local_files_only"]
        )
        self.lock = threading.RLock()
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        self.dimension = (
            get_dimension() if get_dimension else self.model.get_sentence_embedding_dimension()
        )

    @property
    def signature(self) -> dict:
        return {
            "model": self.config["model"],
            "normalize": self.config["normalize"],
            "dimension": self.dimension,
        }

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.dimension, dtype=torch.float32)
        with self.lock, torch.inference_mode():
            embeddings = (
                self.model.encode(
                    texts,
                    batch_size=self.config["batch_size"],
                    normalize_embeddings=self.config["normalize"],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .detach()
                .cpu()
                .float()
            )
        return embeddings.clone()
