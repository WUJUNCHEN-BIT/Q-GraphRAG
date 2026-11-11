"""Question-entity linking without answer or supporting-document access."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .io import fingerprint
from .prompts import QUERY_ENTITIES
from .schema import GraphIndex, normalize_label


class EntityLinker:
    def __init__(self, graph: GraphIndex, encoder, backend, top_k=1):
        self.graph, self.encoder, self.backend = graph, encoder, backend
        self.top_k = top_k
        self.by_name = {normalize_label(name): i for i, name in enumerate(graph.entities)}
        for alias, canonical in graph.aliases.items():
            if canonical in self.by_name:
                self.by_name[normalize_label(alias)] = self.by_name[canonical]
        self._entity_vectors = None

    @property
    def signature(self):
        settings = getattr(self.backend, "config", {})
        return {
            "encoder": self.encoder.signature,
            "top_k": self.top_k,
            "ner_prompt": fingerprint(QUERY_ENTITIES),
            "ner": {
                key: settings.get(key)
                for key in ("model", "base_url", "temperature", "disable_thinking")
            },
        }

    def mentions(self, question: str) -> list[str]:
        def validate(value):
            if not isinstance(value.get("entities"), list) or any(
                not isinstance(x, str) or not x.strip() for x in value["entities"]
            ):
                raise ValueError("query_entities.entities must be a list of nonempty strings")

        result = self.backend.complete("query_entities", {"question": question}, validate)
        return result["entities"]

    def link(self, question: str, mentions: list[str] | None = None) -> list[int]:
        mentions = self.mentions(question) if mentions is None else mentions
        if not isinstance(mentions, list) or any(not isinstance(m, str) for m in mentions):
            raise ValueError("question_entities must be a string list")
        found, unresolved = set(), []
        for mention in mentions:
            key = normalize_label(mention)
            if key in self.by_name:
                found.add(self.by_name[key])
            elif key:
                unresolved.append(mention)
        if unresolved and self.graph.entities:
            if self._entity_vectors is None:
                self._entity_vectors = F.normalize(self.encoder.encode(self.graph.entities), dim=-1)
            vectors = F.normalize(self.encoder.encode(unresolved), dim=-1)
            similarities = vectors @ self._entity_vectors.t()
            for row in similarities:
                ids = torch.argsort(row, descending=True, stable=True)[: self.top_k]
                found.update(ids.tolist())
        return sorted(found)
