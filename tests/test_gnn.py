import copy
import threading

import pytest
import torch
from torch.nn import functional as F

from qgraphrag.gnn.losses import retrieval_losses
from qgraphrag.gnn.model import QualityGNNRetriever
from qgraphrag.gnn.ops import distmult_sum
from qgraphrag.tensors import tensorize
from qgraphrag.training import StrictNegativeSampler


def test_chunked_sum_gradcheck_including_learned_edge_gates():
    torch.manual_seed(4)
    nodes = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
    relations = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    types = torch.tensor([0, 1, 0, 1])
    gates = torch.rand(4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b, c: distmult_sum(a, b, edges, types, c, 2), (nodes, relations, gates), atol=1e-5
    )


def test_inverse_arcs_share_original_quality_and_relation_identity(toy_graph, tiny_encoder):
    tensors = tensorize(toy_graph, tiny_encoder)
    count = len(toy_graph.edges)
    assert torch.equal(tensors.edge_to_final[:count], tensors.edge_to_final[count:])
    assert torch.equal(tensors.edge_index[:, :count].flip(0), tensors.edge_index[:, count:])
    assert tensors.quality.shape == (count, 3)


def test_quality_supervision_has_nonzero_gradients(toy_graph, tiny_encoder, small_config):
    torch.manual_seed(15)
    graph = tensorize(toy_graph, tiny_encoder)
    model = QualityGNNRetriever(small_config["model"], tiny_encoder.dimension)
    questions = tiny_encoder.encode(["Where was Ada Stone born?"])
    seeds = torch.zeros(1, graph.num_nodes)
    seeds[0, toy_graph.entities.index("ada stone")] = 1
    output = model(graph, questions, seeds, collect_support=True)
    targets = torch.zeros_like(output.logits)
    targets[0, [toy_graph.entities.index("riverford"), toy_graph.entities.index("lumen")]] = 1
    bce, ranking = retrieval_losses(output.logits, targets)
    (bce + ranking).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.quality_encoder.parameters()
    )
    assert output.total_support.shape == output.logits.shape
    assert torch.all(output.effective_support <= output.total_support + 1e-6)
    assert torch.all((output.gates > 0) & (output.gates < 1))


def test_support_readout_uses_raw_messages_before_gating(toy_graph, tiny_encoder, small_config):
    config = copy.deepcopy(small_config["model"])
    config["hidden_dims"] = [4]
    graph = tensorize(toy_graph, tiny_encoder)
    model = QualityGNNRetriever(config, 8).eval()
    question = tiny_encoder.encode(["Ada Stone"])
    seeds = torch.zeros(1, graph.num_nodes)
    seeds[0, toy_graph.entities.index("ada stone")] = 1
    output = model(graph, question, seeds, collect_support=True)
    query = model.question_mlp(question)
    nodes = seeds.unsqueeze(-1) * query.unsqueeze(1)
    relation = model.entity_model.layers[0].relation_projection(
        model.rel_mlp(graph.relation_embeddings)
    )
    raw = (nodes[:, graph.edge_index[0]] * relation[graph.edge_type].unsqueeze(0)).norm(dim=-1)
    expected_u = torch.zeros_like(output.logits)
    expected_w = torch.zeros_like(output.logits)
    expected_u.index_add_(1, graph.edge_index[1], raw)
    expected_w.index_add_(1, graph.edge_index[1], raw * output.gates[graph.edge_to_final])
    torch.testing.assert_close(output.edge_contributions, raw)
    torch.testing.assert_close(output.total_support, expected_u)
    torch.testing.assert_close(output.effective_support, expected_w)


def test_query_independent_cache_invalidates_on_encoder_change(
    toy_graph, tiny_encoder, small_config, tmp_path
):
    graph = tensorize(toy_graph, tiny_encoder)
    model = QualityGNNRetriever(small_config["model"], 8).eval()
    path = tmp_path / "gates.pt"
    first = model.cache_gates(graph, path).clone()
    model.load_gate_cache(graph, path)
    torch.testing.assert_close(first, model.gates(graph))
    with torch.no_grad():
        model.quality_encoder.net[-1].bias.add_(0.5)
    assert not torch.allclose(first, model.gates(graph))
    with pytest.raises(ValueError, match="does not match"):
        model.load_gate_cache(graph, path)
    model.train()
    assert model.gates(graph).requires_grad
    assert torch.equal(model.gates(graph, use_quality=False), torch.ones(len(toy_graph.edges)))


def test_pretraining_negative_sampler_excludes_all_true_tails(toy_graph, tiny_encoder):
    graph = tensorize(toy_graph, tiny_encoder)
    sampler = StrictNegativeSampler(graph)
    generator = torch.Generator().manual_seed(123)
    for (source, relation), true_tails in sampler.truth.items():
        sampled = sampler.sample(source, relation, 50, generator)
        assert not set(sampled.tolist()) & true_tails


def test_losses_match_equations_and_detached_negative_weights():
    logits = torch.tensor([[0.2, -0.5, 1.3, -0.7]], dtype=torch.double, requires_grad=True)
    labels = torch.tensor([[1, 0, 1, 0]], dtype=torch.double)
    bce, ranking = retrieval_losses(logits, labels, 0.2)
    weights = torch.softmax(logits.detach()[0, [1, 3]] / 0.2, dim=0)
    elements = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    expected_bce = 0.5 * (elements[0, [0, 2]].mean() + (elements[0, [1, 3]] * weights).sum())
    probability = logits.sigmoid()
    expected_rank = -torch.log(probability[0, [0, 2]] / probability.sum()).mean()
    torch.testing.assert_close(bce, expected_bce)
    torch.testing.assert_close(ranking, expected_rank)
    actual_grad = torch.autograd.grad(bce, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_bce, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_ranking_loss_is_finite_for_extreme_logits():
    logits = torch.tensor([[-1000.0, 1000.0, -999.0]], requires_grad=True)
    bce, rank = retrieval_losses(logits, torch.tensor([[1, 0, 0]]))
    (bce + rank).backward()
    assert torch.isfinite(bce) and torch.isfinite(rank) and torch.isfinite(logits.grad).all()


def test_frozen_embeddings_can_feed_a_trainable_projection_on_cpu():
    from qgraphrag.embeddings import SentenceEncoder

    class Backend:
        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), 8)

    encoder = SentenceEncoder.__new__(SentenceEncoder)
    encoder.config = {"batch_size": 2, "normalize": False}
    encoder.model = Backend()
    encoder.lock = threading.RLock()
    encoder.dimension = 8
    embeddings = encoder.encode(["question"])
    projection = torch.nn.Linear(8, 4)
    projection(embeddings).sum().backward()
    assert not embeddings.requires_grad
    assert projection.weight.grad is not None
