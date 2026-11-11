"""Quality modulation, query initialization, graph-support readout and gate caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..io import save_torch
from ..schema import QUALITY_FEATURE_VERSION
from ..tensors import GraphTensors
from .backbone import QueryNBFNet


@dataclass
class RetrievalOutput:
    logits: torch.Tensor
    gates: torch.Tensor
    edge_contributions: torch.Tensor | None = None
    total_support: torch.Tensor | None = None
    effective_support: torch.Tensor | None = None


class QualityEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, quality):
        # No query representation is accepted here. Coefficients are not confidence probabilities.
        return torch.sigmoid(self.net(quality).squeeze(-1))


class QualityGNNRetriever(nn.Module):
    def __init__(self, config: dict, embedding_dim: int):
        super().__init__()
        self.model_config = dict(config)
        self.embedding_dim = embedding_dim
        self.entity_model = QueryNBFNet(config)
        self.rel_mlp = nn.Linear(embedding_dim, config["input_dim"])
        self.question_mlp = nn.Linear(embedding_dim, config["input_dim"])
        self.quality_encoder = QualityEncoder(config["quality_hidden_dim"])
        self._gate_cache = None

    def train(self, mode: bool = True):
        self._gate_cache = None
        return super().train(mode)

    def quality_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for key, tensor in self.quality_encoder.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def gates(self, graph: GraphTensors, use_quality: bool = True):
        if graph.quality_feature_version != QUALITY_FEATURE_VERSION:
            raise ValueError("Model and graph quality feature version mismatch")
        if not use_quality:
            return graph.quality.new_ones(graph.num_edges)
        if self.training:
            return self.quality_encoder(graph.quality)
        key = (graph.graph_fingerprint, self.quality_fingerprint(), str(graph.quality.device))
        if self._gate_cache is None or self._gate_cache[0] != key:
            with torch.no_grad():
                self._gate_cache = (key, self.quality_encoder(graph.quality).detach())
        return self._gate_cache[1]

    @torch.no_grad()
    def cache_gates(self, graph: GraphTensors, path: str | Path):
        if self.training:
            raise ValueError("Gate caching requires model.eval()")
        values = self.gates(graph)
        save_torch(
            path,
            {
                "schema_version": 2,
                "quality_feature_version": graph.quality_feature_version,
                "graph_fingerprint": graph.graph_fingerprint,
                "quality_fingerprint": self.quality_fingerprint(),
                "gates": values.cpu(),
            },
        )
        return values

    def load_gate_cache(self, graph: GraphTensors, path: str | Path):
        if graph.quality_feature_version != QUALITY_FEATURE_VERSION:
            raise ValueError("Model and graph quality feature version mismatch")
        if self.training:
            raise ValueError("Cached gates cannot be used in training")
        data = torch.load(path, map_location=graph.quality.device, weights_only=True)
        if (
            data.get("schema_version") != 2
            or data.get("quality_feature_version") != graph.quality_feature_version
            or data["graph_fingerprint"] != graph.graph_fingerprint
            or data["quality_fingerprint"] != self.quality_fingerprint()
        ):
            raise ValueError("Gate cache does not match the graph and trained quality encoder")
        values = data["gates"]
        if (
            values.shape != (graph.num_edges,)
            or not torch.isfinite(values).all()
            or not ((values >= 0) & (values <= 1)).all()
        ):
            raise ValueError("Invalid cached gate values")
        key = (graph.graph_fingerprint, self.quality_fingerprint(), str(graph.quality.device))
        self._gate_cache = (key, values)

    def forward(
        self,
        graph: GraphTensors,
        question_embeddings: torch.Tensor,
        seed_mask: torch.Tensor,
        collect_support=False,
        use_quality=True,
    ) -> RetrievalOutput:
        if question_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("Question embedding dimension mismatch")
        if seed_mask.shape != (len(question_embeddings), graph.num_nodes):
            raise ValueError("Seed mask dimension mismatch")
        query = self.question_mlp(question_embeddings)
        boundary = seed_mask.unsqueeze(-1).to(query.dtype) * query.unsqueeze(1)
        relation = self.rel_mlp(graph.relation_embeddings)
        gates = self.gates(graph, use_quality)
        arc_gates = gates.index_select(0, graph.edge_to_final)
        logits, contributions = self.entity_model(
            graph, boundary, relation, query, arc_gates, collect_support
        )
        total, effective = None, None
        if collect_support:
            total = logits.new_zeros(logits.shape)
            effective = logits.new_zeros(logits.shape)
            # Incoming reverse arcs contribute to their receiving node just like forward arcs.
            total.index_add_(1, graph.edge_index[1], contributions)
            effective.index_add_(1, graph.edge_index[1], contributions * arc_gates.unsqueeze(0))
        return RetrievalOutput(logits, gates, contributions, total, effective)

    def completion(
        self,
        graph: GraphTensors,
        source: torch.Tensor,
        relation_ids: torch.Tensor,
        candidates: torch.Tensor,
        blocked_final_edges: torch.Tensor,
    ):
        relations = self.rel_mlp(graph.relation_embeddings)
        query = relations.index_select(0, relation_ids)
        boundary = query.new_zeros((len(source), graph.num_nodes, query.shape[-1]))
        boundary[torch.arange(len(source), device=source.device), source] = query
        # Completion pretraining fixes EVERY remaining edge coefficient to one.
        gates = graph.quality.new_ones(graph.edge_type.numel())
        if blocked_final_edges.numel():
            blocked = torch.zeros(graph.num_edges, device=gates.device, dtype=torch.bool)
            blocked[blocked_final_edges] = True
            gates = gates.masked_fill(blocked[graph.edge_to_final], 0)
        logits, _ = self.entity_model(graph, boundary, relations, query, gates)
        return logits.gather(1, candidates)
