"""Graph tensors: inverse arcs share the original edge's quality features and identity."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch

from .io import save_torch
from .schema import QUALITY_FEATURE_VERSION, GraphIndex


@dataclass
class GraphTensors:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_to_final: torch.Tensor
    quality: torch.Tensor
    relation_embeddings: torch.Tensor
    num_nodes: int
    num_direct_relations: int
    graph_fingerprint: str
    encoder_signature: dict
    relation_labels: list[str]
    quality_feature_version: int

    @property
    def num_edges(self) -> int:
        return self.quality.shape[0]

    def to(self, device: str | torch.device) -> GraphTensors:
        return GraphTensors(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    def validate(self) -> None:
        if self.quality_feature_version != QUALITY_FEATURE_VERSION:
            raise ValueError("Graph tensor quality feature version mismatch")
        count = self.num_edges
        if self.edge_index.shape != (2, 2 * count) or self.edge_type.shape != (2 * count,):
            raise ValueError("Invalid graph arc dimensions")
        if self.quality.shape != (count, 3):
            raise ValueError("Quality features must have shape [final edges,3]")
        if self.edge_to_final.shape != (2 * count,):
            raise ValueError("Missing inverse-arc identities")
        if count and (self.edge_index.min() < 0 or self.edge_index.max() >= self.num_nodes):
            raise ValueError("Invalid entity index")
        if (
            not torch.isfinite(self.quality).all()
            or not ((self.quality >= 0) & (self.quality <= 1)).all()
        ):
            raise ValueError("Invalid quality values")
        if not torch.isfinite(self.relation_embeddings).all():
            raise ValueError("Nonfinite relation embeddings")
        if self.relation_embeddings.shape[0] != 2 * self.num_direct_relations:
            raise ValueError("Invalid relation embedding dimensions")
        expected = torch.arange(count, device=self.edge_to_final.device).repeat(2)
        if not torch.equal(self.edge_to_final, expected):
            raise ValueError("Inverse arcs must preserve their final-edge identity")
        if not torch.equal(self.edge_index[:, count:], self.edge_index[:, :count].flip(0)):
            raise ValueError("Invalid inverse endpoints")
        if not torch.equal(
            self.edge_type[count:], self.edge_type[:count] + self.num_direct_relations
        ):
            raise ValueError("Invalid inverse relation IDs")


def relation_text(edge) -> str:
    if not edge.qualifiers:
        return edge.relation
    return edge.relation + " [qualifiers: " + "; ".join(edge.qualifiers) + "]"


def tensorize(graph: GraphIndex, encoder) -> GraphTensors:
    graph.validate()
    if not graph.entities or not graph.edges:
        raise ValueError("The constructed graph is empty; inspect construction_audit.json")
    entities = {name: i for i, name in enumerate(graph.entities)}
    relation_labels = sorted({relation_text(edge) for edge in graph.edges})
    relations = {name: i for i, name in enumerate(relation_labels)}
    count, relation_count = len(graph.edges), len(relations)
    edge_index = (
        torch.tensor([[entities[e.head], entities[e.tail]] for e in graph.edges], dtype=torch.long)
        .t()
        .contiguous()
    )
    edge_type = torch.tensor([relations[relation_text(e)] for e in graph.edges], dtype=torch.long)
    labels = relation_labels + ["inverse_" + x for x in relation_labels]
    tensors = GraphTensors(
        torch.cat((edge_index, edge_index.flip(0)), dim=1),
        torch.cat((edge_type, edge_type + relation_count)),
        torch.arange(count).repeat(2),
        torch.tensor([e.features for e in graph.edges], dtype=torch.float32),
        encoder.encode(labels),
        len(entities),
        relation_count,
        graph.fingerprint,
        encoder.signature,
        labels,
        graph.quality_feature_version,
    )
    tensors.validate()
    return tensors


def prepare_tensors(graph: GraphIndex, encoder, directory: str | Path) -> GraphTensors:
    path = Path(directory) / "graph_tensors.pt"
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (
            data["graph_fingerprint"] == graph.fingerprint
            and data["encoder_signature"] == encoder.signature
        ):
            if data.get("quality_feature_version") != graph.quality_feature_version:
                raise ValueError("Cached tensor quality feature version mismatch")
            tensors = GraphTensors(**data)
            tensors.validate()
            return tensors
    tensors = tensorize(graph, encoder)
    save_torch(path, {f.name: getattr(tensors, f.name) for f in fields(tensors)})
    return tensors
