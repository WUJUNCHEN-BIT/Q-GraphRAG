"""Unrooted connected PCST on the induced undirected candidate topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReasoningSubgraph:
    node_ids: list[int]
    edge_ids: list[int]
    objective: float
    candidate_node_count: int
    candidate_edge_count: int
    node_prizes: dict[int, float]


def global_minmax(values: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("Normalization requires a finite, nonempty full-node vector")
    return (values - values.min()) / (values.max() - values.min() + epsilon)


def node_rewards(scores, total_support, effective_support, alpha_s=0.5, epsilon=1e-8):
    if scores.shape != total_support.shape or scores.shape != effective_support.shape:
        raise ValueError("Graph-support readout dimension mismatch")
    if (
        (total_support < 0).any()
        or (effective_support < 0).any()
        or not torch.isfinite(total_support).all()
        or not torch.isfinite(effective_support).all()
    ):
        raise ValueError("Invalid propagation support statistics")
    normalized_scores = global_minmax(scores, epsilon)
    normalized_log_support = global_minmax(torch.log1p(total_support), epsilon)
    ratio = effective_support / (total_support + epsilon)
    return normalized_scores * ((1 - alpha_s) + alpha_s * ratio * normalized_log_support)


class PCSTSelector:
    def __init__(self, graph_index, config):
        self.graph = graph_index
        self.config = config
        self.entity_ids = {name: i for i, name in enumerate(graph_index.entities)}
        self.endpoints = [
            (self.entity_ids[e.head], self.entity_ids[e.tail]) for e in graph_index.edges
        ]
        self.neighbors = [set() for _ in graph_index.entities]
        for source, target in self.endpoints:
            self.neighbors[source].add(target)
            self.neighbors[target].add(source)

    def candidate_nodes(self, scores):
        seeds = torch.argsort(scores, descending=True, stable=True)[: self.config["top_m"]].tolist()
        nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(self.config["hops"]):
            next_frontier = {n for v in frontier for n in self.neighbors[v]} - nodes
            nodes.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return nodes

    def select(self, scores, gates, total_support, effective_support):
        from pcst_fast import pcst_fast

        scores, gates = scores.detach().cpu(), gates.detach().cpu()
        total_support, effective_support = (
            total_support.detach().cpu(),
            effective_support.detach().cpu(),
        )
        if scores.shape != (len(self.graph.entities),) or gates.shape != (len(self.graph.edges),):
            raise ValueError("PCST input dimensions mismatch")
        if not torch.isfinite(gates).all() or not ((gates >= 0) & (gates <= 1)).all():
            raise ValueError("Invalid soft edge coefficients")
        settings = self.config
        prizes = node_rewards(
            scores, total_support, effective_support, settings["alpha_s"], settings["epsilon"]
        )
        costs = settings["lambda_pcst"] * (settings["c_min"] + settings["eta"] * (1 - gates))
        nodes = self.candidate_nodes(scores)
        candidate_edge_ids = [
            i for i, (u, v) in enumerate(self.endpoints) if u in nodes and v in nodes
        ]
        # Positive-cost self-loops/cycles cannot improve a node-prize-only PCST objective.
        # For parallel undirected edges only the cheapest can improve a tree; retain its original ID.
        cheapest = {}
        for index in candidate_edge_ids:
            u, v = self.endpoints[index]
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            old = cheapest.get(key)
            if old is None or float(costs[index]) < float(costs[old]):
                cheapest[key] = index
        topology = {v: set() for v in nodes}
        for u, v in cheapest:
            topology[u].add(v)
            topology[v].add(u)
        components, unvisited = [], set(nodes)
        while unvisited:
            start = min(unvisited)
            frontier, component = {start}, set()
            while frontier:
                component.update(frontier)
                frontier = {w for v in frontier for w in topology[v]} - component
            unvisited.difference_update(component)
            components.append(sorted(component))
        # A connected solution must lie within a single component. Evaluate each independently;
        # compare actual objectives rather than forcing terminals or returning a forest.
        best_nodes, best_edges, best_objective = [], [], -float("inf")
        component_ids = {v: i for i, component in enumerate(components) for v in component}
        component_edges = [[] for _ in components]
        for (u, v), edge_id in sorted(cheapest.items()):
            component_edges[component_ids[u]].append(edge_id)
        for component_index, component in enumerate(components):
            singleton = max(component, key=lambda v: (float(prizes[v]), -v))
            if float(prizes[singleton]) > best_objective:
                best_nodes, best_edges, best_objective = [singleton], [], float(prizes[singleton])
            if len(component) == 1:
                continue
            local = {v: i for i, v in enumerate(component)}
            original_edges = component_edges[component_index]
            edges = np.asarray(
                [
                    [local[self.endpoints[i][0]], local[self.endpoints[i][1]]]
                    for i in original_edges
                ],
                dtype=np.int64,
            )
            node_prizes = np.asarray([float(prizes[i]) for i in component], dtype=np.float64)
            edge_costs = np.asarray([float(costs[i]) for i in original_edges], dtype=np.float64)
            vertices, selected = pcst_fast(
                edges, node_prizes, edge_costs, -1, 1, settings["pruning"], 0
            )
            if not len(vertices):
                continue
            if (
                len(set(vertices.tolist())) != len(vertices)
                or any(int(i) < 0 or int(i) >= len(component) for i in vertices)
                or any(int(i) < 0 or int(i) >= len(original_edges) for i in selected)
            ):
                raise RuntimeError(
                    "pcst_fast returned invalid indices; verify the pinned NumPy/solver versions"
                )
            selected_nodes = sorted(component[int(i)] for i in vertices)
            selected_edges = sorted(original_edges[int(i)] for i in selected)
            if any(not set(self.endpoints[i]) <= set(selected_nodes) for i in selected_edges):
                raise RuntimeError("pcst_fast edge endpoints do not match its selected vertices")
            objective = sum(float(prizes[i]) for i in selected_nodes) - sum(
                float(costs[i]) for i in selected_edges
            )
            if objective > best_objective:
                best_nodes, best_edges, best_objective = selected_nodes, selected_edges, objective
        if not best_nodes:
            raise RuntimeError("PCST returned no connected solution")
        return ReasoningSubgraph(
            best_nodes,
            best_edges,
            best_objective,
            len(nodes),
            len(candidate_edge_ids),
            {i: float(prizes[i]) for i in best_nodes},
        )


def serialize_subgraph(subgraph, graph_index, gates):
    return {
        "nodes": [
            {"id": i, "name": graph_index.entities[i], "prize": subgraph.node_prizes[i]}
            for i in subgraph.node_ids
        ],
        "edges": [
            {
                "id": graph_index.edges[i].id,
                **graph_index.edges[i].triple(),
                "quality_features": graph_index.edges[i].features,
                "gate": float(gates[i]),
                "sources": [
                    {
                        "document_id": p.document_id,
                        "start": p.evidence_start,
                        "end": p.evidence_end,
                    }
                    for p in graph_index.edges[i].sources
                ],
            }
            for i in subgraph.edge_ids
        ],
        "objective": subgraph.objective,
        "candidate_nodes": subgraph.candidate_node_count,
        "candidate_edges": subgraph.candidate_edge_count,
    }
