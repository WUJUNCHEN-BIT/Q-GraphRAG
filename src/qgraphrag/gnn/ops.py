"""Chunked DistMult aggregation with message recomputation in backward."""

from __future__ import annotations

import torch


class _DistMultSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, relations, edge_index, edge_type, gates, chunk_size):
        output = torch.zeros_like(nodes)
        for left in range(0, edge_type.numel(), chunk_size):
            right = left + chunk_size
            source, target = edge_index[:, left:right]
            message = nodes.index_select(1, source) * relations.index_select(
                0, edge_type[left:right]
            ).unsqueeze(0)
            output.index_add_(1, target, message * gates[left:right].view(1, -1, 1))
        ctx.save_for_backward(nodes, relations, edge_index, edge_type, gates)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        nodes, relations, edge_index, edge_type, gates = ctx.saved_tensors
        grad_nodes = torch.zeros_like(nodes)
        grad_relations = torch.zeros_like(relations)
        grad_gates = torch.empty_like(gates)
        for left in range(0, edge_type.numel(), ctx.chunk_size):
            right = left + ctx.chunk_size
            source, target = edge_index[:, left:right]
            types = edge_type[left:right]
            origin = nodes.index_select(1, source)
            relation = relations.index_select(0, types).unsqueeze(0)
            gradient = grad_output.index_select(1, target)
            weighted_gradient = gradient * gates[left:right].view(1, -1, 1)
            grad_nodes.index_add_(1, source, weighted_gradient * relation)
            grad_relations.index_add_(0, types, (weighted_gradient * origin).sum(0))
            grad_gates[left:right] = (gradient * origin * relation).sum((0, 2))
        return grad_nodes, grad_relations, None, None, grad_gates, None


def distmult_sum(nodes, relations, edge_index, edge_type, gates, chunk_size=8192):
    return _DistMultSum.apply(nodes, relations, edge_index, edge_type, gates, chunk_size)


@torch.no_grad()
def raw_message_norms(nodes, relations, edge_index, edge_type, chunk_size=8192):
    result = nodes.new_empty((nodes.shape[0], edge_type.numel()))
    for left in range(0, edge_type.numel(), chunk_size):
        right = left + chunk_size
        raw = nodes.index_select(1, edge_index[0, left:right]) * relations.index_select(
            0, edge_type[left:right]
        ).unsqueeze(0)
        result[:, left:right] = raw.norm(p=2, dim=-1)
    return result
