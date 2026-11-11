"""NBFNet backbone adapted from GFM-RAG/ULTRA (Apache-2.0).

Modified for quality-weighted aggregation and propagation-support readout.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ops import distmult_sum, raw_message_norms


class GeneralizedRelationalConv(nn.Module):
    def __init__(self, dimension: int, layer_norm: bool, chunk_size: int):
        super().__init__()
        self.linear = nn.Linear(dimension * 2, dimension)
        self.layer_norm = nn.LayerNorm(dimension) if layer_norm else None
        self.relation_projection = nn.Sequential(
            nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension)
        )
        self.chunk_size = chunk_size

    def forward(
        self, nodes, boundary, relation_representations, graph, arc_gates, collect_support=False
    ):
        relation = self.relation_projection(relation_representations)
        contribution = (
            raw_message_norms(nodes, relation, graph.edge_index, graph.edge_type, self.chunk_size)
            if collect_support
            else None
        )
        aggregate = distmult_sum(
            nodes, relation, graph.edge_index, graph.edge_type, arc_gates, self.chunk_size
        )
        # Original NBFNet's boundary condition is reinjected at each layer; it is not a KG edge.
        updated = self.linear(torch.cat((nodes, aggregate + boundary), dim=-1))
        if self.layer_norm is not None:
            updated = self.layer_norm(updated)
        return F.relu(updated), contribution


class QueryNBFNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dimension = config["input_dim"]
        self.dims = [dimension] + config["hidden_dims"]
        self.short_cut = config["short_cut"]
        self.concat_hidden = config["concat_hidden"]
        self.layers = nn.ModuleList(
            [
                GeneralizedRelationalConv(
                    dimension, config["layer_norm"], config["edge_chunk_size"]
                )
                for _ in config["hidden_dims"]
            ]
        )
        feature_dim = (sum(config["hidden_dims"]) if self.concat_hidden else dimension) + dimension
        mlp = []
        for _ in range(config["num_mlp_layers"] - 1):
            mlp.extend((nn.Linear(feature_dim, feature_dim), nn.ReLU()))
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self, graph, boundary, relation_representations, query, arc_gates, collect_support=False
    ):
        nodes = boundary
        hiddens = []
        contribution = (
            boundary.new_zeros((boundary.shape[0], graph.edge_type.numel()))
            if collect_support
            else None
        )
        for layer in self.layers:
            hidden, raw = layer(
                nodes, boundary, relation_representations, graph, arc_gates, collect_support
            )
            if self.short_cut:
                hidden = hidden + nodes
            hiddens.append(hidden)
            nodes = hidden
            if collect_support:
                contribution = contribution + raw
        node_query = query.unsqueeze(1).expand(-1, graph.num_nodes, -1)
        features = torch.cat(
            hiddens + [node_query] if self.concat_hidden else [hiddens[-1], node_query], dim=-1
        )
        return self.mlp(features).squeeze(-1), contribution
