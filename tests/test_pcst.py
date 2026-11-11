from types import SimpleNamespace

import pytest
import torch

from qgraphrag.pcst import PCSTSelector, global_minmax, node_rewards


def topology_graph(entities, edges):
    return SimpleNamespace(
        entities=entities, edges=[SimpleNamespace(head=u, tail=v) for u, v in edges]
    )


def assert_connected(result, graph):
    nodes = set(result.node_ids)
    assert len(nodes) == len(result.node_ids)
    reached = {next(iter(nodes))}
    while True:
        before = len(reached)
        for i in result.edge_ids:
            edge = graph.edges[i]
            u, v = graph.entities.index(edge.head), graph.entities.index(edge.tail)
            assert u in nodes and v in nodes
            if u in reached or v in reached:
                reached.update((u, v))
        if len(reached) == before:
            break
    assert reached == nodes


def test_global_normalization_and_support_reward_formula():
    scores = torch.tensor([-1.0, 0.0, 3.0, 100.0])
    total = torch.tensor([0.0, 1.0, 3.0, 1000.0])
    effective = total * 0.5
    rewards = node_rewards(scores, total, effective, 0.3)
    expected = global_minmax(scores) * (
        0.7 + 0.3 * effective / (total + 1e-8) * global_minmax(torch.log1p(total))
    )
    torch.testing.assert_close(rewards, expected)
    assert rewards[2] < 0.04  # The full-graph maximum remains in the normalization domain.
    assert torch.equal(global_minmax(torch.ones(4)), torch.zeros(4))


def test_low_gate_bridge_can_survive_pcst(small_config):
    graph = topology_graph(["a", "bridge", "c"], [("a", "bridge"), ("bridge", "c")])
    config = {**small_config["pcst"], "top_m": 2, "hops": 1, "alpha_s": 0.0, "lambda_pcst": 0.2}
    result = PCSTSelector(graph, config).select(
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([0.01, 0.01]),
        torch.ones(3),
        torch.ones(3) * 0.01,
    )
    assert set(result.node_ids) == {0, 1, 2}
    assert set(result.edge_ids) == {0, 1}
    assert_connected(result, graph)


def test_top_m_entities_are_not_mandatory_terminals(small_config):
    graph = topology_graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    config = {**small_config["pcst"], "top_m": 3, "alpha_s": 0.0, "lambda_pcst": 100.0}
    result = PCSTSelector(graph, config).select(
        torch.tensor([1.0, 0.5, 0.0]), torch.zeros(2), torch.zeros(3), torch.zeros(3)
    )
    assert result.node_ids == [0] and result.edge_ids == []


def test_disconnected_candidate_graph_returns_best_connected_component(small_config):
    graph = topology_graph(["a", "b", "c", "d", "isolated"], [("a", "b"), ("c", "d")])
    config = {**small_config["pcst"], "top_m": 5, "alpha_s": 0.0, "lambda_pcst": 0.1}
    result = PCSTSelector(graph, config).select(
        torch.tensor([1.0, 0.8, 0.2, 0.3, 0.0]), torch.ones(2), torch.ones(5), torch.ones(5)
    )
    assert set(result.node_ids) == {0, 1}
    assert_connected(result, graph)


def test_parallel_edges_select_cheapest_original_directed_relation(small_config):
    graph = topology_graph(["a", "b", "c"], [("b", "a"), ("a", "b"), ("b", "c")])
    config = {**small_config["pcst"], "top_m": 3, "alpha_s": 0.0, "lambda_pcst": 0.1}
    result = PCSTSelector(graph, config).select(
        torch.tensor([1.0, 0.9, 0.0]), torch.tensor([1.0, 0.01, 1.0]), torch.ones(3), torch.ones(3)
    )
    assert 0 in result.edge_ids and 1 not in result.edge_ids
    assert graph.edges[result.edge_ids[0]].head == "b"
    assert_connected(result, graph)


def test_single_node_and_zero_support_are_well_defined(small_config):
    graph = topology_graph(["a"], [])
    result = PCSTSelector(graph, small_config["pcst"]).select(
        torch.zeros(1), torch.zeros(0), torch.zeros(1), torch.zeros(1)
    )
    assert result.node_ids == [0] and result.edge_ids == [] and result.objective == pytest.approx(0)
