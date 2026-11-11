import json
import threading
from types import SimpleNamespace

import pytest

from qgraphrag.pipeline import QGraphRAG
from qgraphrag.retrieval import GraphRetrieval, ScoredDocument, TextRetriever, graph_document_scores


def pipeline_fixture(toy_graph, recording_backend, sufficient):
    calls = []
    scaffold = {
        "nodes": [{"id": 0, "name": "ada stone"}, {"id": 1, "name": "riverford"}],
        "edges": [
            {
                "head": "ada stone",
                "relation": "born in",
                "tail": "riverford",
                "qualifiers": [],
                "sources": [{"document_id": "d4", "evidence": "UNSELECTED SOURCE QUOTE"}],
            }
        ],
    }
    graph_started, text_started = threading.Event(), threading.Event()

    class Graph:
        def retrieve(self, question, question_entities=None):
            graph_started.set()
            assert text_started.wait(timeout=5), "Text branch did not run in parallel"
            return GraphRetrieval([ScoredDocument("d1", 1, "graph")], scaffold, ["ada stone"], [])

    class Text:
        def retrieve(self, question):
            calls.append(question)
            text_started.set()
            assert graph_started.wait(timeout=5)
            return [ScoredDocument("d2" if question == "original question" else "d3", 1, "text")]

    def handler(task, payload):
        assert "UNSELECTED SOURCE QUOTE" not in json.dumps(payload)
        assert "sources" not in json.dumps(payload["subgraph"])
        if task == "sufficiency":
            assert payload["subgraph"]["edges"] == [
                {"head": "ada stone", "relation": "born in", "tail": "riverford", "qualifiers": []}
            ]
            ids = {d["id"] for d in payload["documents"]}
            assert ids == (
                {"d1", "d2"} if payload["supplementation_round"] == 0 else {"d1", "d2", "d3"}
            )
            current_sufficient = sufficient or "d3" in ids
            return {
                "sufficient": current_sufficient,
                "missing_information": "" if current_sufficient else "bridge",
                "supplement_query": "" if current_sufficient else "targeted missing fact",
            }
        if task == "select":
            candidates = {d["id"] for d in payload["documents"]}
            assert candidates == ({"d1", "d2"} if sufficient else {"d1", "d2", "d3"})
            return {"document_ids": ["d1", "d2"] if sufficient else ["d3", "d1"]}
        assert task == "answer"
        assert len(payload["documents"]) == 2
        assert payload["subgraph"]["edges"] == [
            {"head": "ada stone", "relation": "born in", "tail": "riverford", "qualifiers": []}
        ]
        assert {d["id"] for d in payload["documents"]} == (
            {"d1", "d2"} if sufficient else {"d3", "d1"}
        )
        return {"answer": "Lumen"}

    backend = recording_backend(handler)
    return QGraphRAG(toy_graph, Graph(), Text(), backend, final_top_k=2), calls, backend


def test_sufficient_evidence_skips_supplementation_and_retains_graph_context(
    toy_graph, recording_backend
):
    pipeline, queries, backend = pipeline_fixture(toy_graph, recording_backend, True)
    result = pipeline.answer("original question", question_id="q")
    assert queries == ["original question"]
    assert result["trace"]["supplementation_steps"] == 0
    assert result["retrieved_document_ids"] == ["d1", "d2"]
    assert [task for task, _ in backend.calls] == ["sufficiency", "select", "answer"]


def test_one_supplementation_is_reassessed_and_competes_for_final_k(toy_graph, recording_backend):
    pipeline, queries, backend = pipeline_fixture(toy_graph, recording_backend, False)
    result = pipeline.answer("original question")
    assert queries == ["original question", "targeted missing fact"]
    assert result["trace"]["supplementation_steps"] == 1
    assert result["retrieved_document_ids"] == ["d3", "d1"]
    assert len([task for task, _ in backend.calls if task == "sufficiency"]) == 2
    assert result["trace"]["sufficiency"]["sufficient"] is True
    assert result["trace"]["supplementation_stop_reason"] == "sufficient"


def test_invalid_document_ids_fail_instead_of_silently_filling(toy_graph, recording_backend):
    graph = SimpleNamespace(
        retrieve=lambda *args: GraphRetrieval(
            [ScoredDocument("d1", 1, "graph")], {"nodes": [], "edges": []}, [], []
        )
    )
    text = SimpleNamespace(retrieve=lambda question: [ScoredDocument("d2", 1, "text")])
    backend = recording_backend(
        lambda task, payload: (
            {"sufficient": True} if task == "sufficiency" else {"document_ids": ["d1", "invented"]}
        )
    )
    with pytest.raises(ValueError, match="distinct IDs"):
        QGraphRAG(toy_graph, graph, text, backend, 2).answer("Ada?")


def test_graph_document_retrieval_uses_top_entity_support_not_pcst(toy_graph):
    import torch

    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    docs, selected, normalized = graph_document_scores(toy_graph, logits, 2, 4)
    names = {toy_graph.entities[i] for i in selected}
    for row in docs:
        expected = sum(
            float(normalized[toy_graph.entities.index(e)])
            for e in set(toy_graph.document_entities[row.document_id]) & names
        )
        assert row.score == pytest.approx(expected)


def test_text_branch_deduplicates_bm25_vector_union_before_cross_encoder(
    toy_graph, tiny_encoder, small_config, tmp_path
):
    class Reranker:
        calls = []

        def score(self, question, docs):
            self.calls.append([d.id for d in docs])
            assert len({d.id for d in docs}) == len(docs)
            return [2.0 if d.id == "d2" else 0.0 for d in docs]

    reranker = Reranker()
    retriever = TextRetriever(
        toy_graph, tiny_encoder, reranker, small_config["retrieval"], tmp_path
    )
    results = retriever.retrieve("Riverford Lumen")
    assert results[0].document_id == "d2" and len(results) == 2
    assert len(reranker.calls) == 1


def iterative_fixture(toy_graph, recording_backend, assessments, retrievals, limit=2):
    graph_calls, queries, pools = [], [], []
    scaffold = {
        "nodes": [{"id": 0, "name": "ada stone"}, {"id": 1, "name": "riverford"}],
        "edges": [
            {
                "head": "ada stone",
                "relation": "born in",
                "tail": "riverford",
                "qualifiers": [],
                "sources": [{"evidence": "HIDDEN SOURCE QUOTE"}],
            }
        ],
    }
    reasoning = {
        "nodes": scaffold["nodes"],
        "edges": [
            {key: scaffold["edges"][0][key] for key in ("head", "relation", "tail", "qualifiers")}
        ],
    }

    class Graph:
        def retrieve(self, question, question_entities=None):
            graph_calls.append(question)
            return GraphRetrieval([ScoredDocument("d1", 1, "graph")], scaffold, [], [])

    class Text:
        def retrieve(self, query):
            queries.append(query)
            if query == "original question":
                return [ScoredDocument("d2", 1, "text")]
            round_number = len(queries) - 1
            assert query == f"gap query {round_number}"
            return [ScoredDocument(doc_id, 1, "text") for doc_id in retrievals[round_number - 1]]

    def handler(task, payload):
        assert payload["question"] == "original question"
        assert payload["subgraph"] == reasoning
        assert "HIDDEN SOURCE QUOTE" not in json.dumps(payload)
        document_ids = [doc["id"] for doc in payload["documents"]]
        assert len(document_ids) == len(set(document_ids))
        if task == "sufficiency":
            index = len(pools)
            assert payload["supplementation_round"] == index
            assert payload["remaining_rounds"] == limit - index
            pools.append(document_ids)
            sufficient = assessments[index]
            return {
                "sufficient": sufficient,
                "missing_information": "" if sufficient else "remaining gap",
                "supplement_query": f"gap query {index + 1}"
                if not sufficient and limit > index
                else "",
            }
        final_ids = pools[-1][::-1][:2]
        if task == "select":
            assert document_ids == pools[-1]
            return {"document_ids": final_ids}
        assert task == "answer" and document_ids == final_ids
        return {"answer": "Lumen"}

    backend = recording_backend(handler)
    pipeline = QGraphRAG(
        toy_graph, Graph(), Text(), backend, final_top_k=2, max_supplementation_rounds=limit
    )
    return pipeline, queries, pools, graph_calls, backend


def test_two_rounds_reassess_updated_pools_and_retain_fixed_graph(toy_graph, recording_backend):
    pipeline, queries, pools, graph_calls, backend = iterative_fixture(
        toy_graph, recording_backend, [False, False, True], [["d3"], ["d4"]]
    )
    result = pipeline.answer("original question")
    assert queries == ["original question", "gap query 1", "gap query 2"]
    assert pools == [["d1", "d2"], ["d1", "d2", "d3"], ["d1", "d2", "d3", "d4"]]
    assert graph_calls == ["original question"]
    assert result["retrieved_document_ids"] == ["d4", "d3"]
    trace = result["trace"]
    assert trace["initial_document_ids"] == ["d1", "d2"]
    assert trace["supplementation_steps"] == 2
    assert trace["supplementation_stop_reason"] == "sufficient"
    assert [item["document_ids"] for item in trace["sufficiency_history"]] == pools
    assert [item["added_document_ids"] for item in trace["supplementation_rounds"]] == [
        ["d3"],
        ["d4"],
    ]
    assert [task for task, _ in backend.calls] == ["sufficiency"] * 3 + ["select", "answer"]


def test_round_limit_still_performs_final_assessment(toy_graph, recording_backend):
    pipeline, queries, pools, _, _ = iterative_fixture(
        toy_graph, recording_backend, [False, False, False], [["d3"], ["d4"]]
    )
    result = pipeline.answer("original question")
    assert len(queries) == 3 and len(pools) == 3
    trace = result["trace"]
    assert trace["supplementation_steps"] == 2
    assert trace["supplementation_stop_reason"] == "max_rounds"
    assert trace["sufficiency"] == {
        "sufficient": False,
        "missing_information": "remaining gap",
        "supplement_query": "",
    }
    assert result["retrieved_document_ids"] == ["d4", "d3"]


@pytest.mark.parametrize("retrieved", [[], ["d1", "d2", "d1"]])
def test_no_new_documents_stops_after_reassessment(toy_graph, recording_backend, retrieved):
    pipeline, queries, pools, graph_calls, _ = iterative_fixture(
        toy_graph, recording_backend, [False, False], [retrieved]
    )
    result = pipeline.answer("original question")
    assert queries == ["original question", "gap query 1"]
    assert pools == [["d1", "d2"], ["d1", "d2"]]
    assert graph_calls == ["original question"]
    assert result["trace"]["supplementation_steps"] == 1
    assert result["trace"]["supplementation_stop_reason"] == "no_new_documents"
    assert result["trace"]["supplementation_rounds"][0]["added_document_ids"] == []


def test_partial_duplicates_are_merged_across_rounds(toy_graph, recording_backend):
    pipeline, _, pools, _, _ = iterative_fixture(
        toy_graph, recording_backend, [False, False, True], [["d2", "d3"], ["d1", "d3", "d4"]]
    )
    trace = pipeline.answer("original question")["trace"]
    assert pools[-1] == ["d1", "d2", "d3", "d4"]
    assert [step["added_document_ids"] for step in trace["supplementation_rounds"]] == [
        ["d3"],
        ["d4"],
    ]
    ids = [item["document_id"] for item in trace["supplementary_candidates"]]
    assert len(ids) == len(set(ids))


def test_second_round_without_new_documents_preserves_accumulated_pool(
    toy_graph, recording_backend
):
    pipeline, _, pools, _, _ = iterative_fixture(
        toy_graph, recording_backend, [False, False, False], [["d3"], ["d2", "d3"]]
    )
    trace = pipeline.answer("original question")["trace"]
    assert pools[-1] == ["d1", "d2", "d3"]
    assert trace["supplementation_stop_reason"] == "no_new_documents"
    assert trace["supplementation_steps"] == 2


def test_zero_round_budget_only_assesses_initial_pool(toy_graph, recording_backend):
    pipeline, queries, pools, _, _ = iterative_fixture(
        toy_graph, recording_backend, [False], [], limit=0
    )
    trace = pipeline.answer("original question")["trace"]
    assert queries == ["original question"] and pools == [["d1", "d2"]]
    assert trace["supplementation_steps"] == 0
    assert trace["supplementation_stop_reason"] == "max_rounds"


@pytest.mark.parametrize(
    "response",
    [
        {"sufficient": "false", "supplement_query": "gap"},
        {"sufficient": False, "supplement_query": ""},
        {"sufficient": False, "supplement_query": "   "},
    ],
)
def test_invalid_sufficiency_response_cannot_start_retrieval(
    toy_graph, recording_backend, response
):
    graph = SimpleNamespace(
        retrieve=lambda *args: GraphRetrieval(
            [ScoredDocument("d1", 1, "graph")], {"nodes": [], "edges": []}, [], []
        )
    )
    text = SimpleNamespace(retrieve=lambda question: [ScoredDocument("d2", 1, "text")])
    backend = recording_backend(lambda task, payload: response)
    with pytest.raises(ValueError, match="sufficient|supplement_query"):
        QGraphRAG(toy_graph, graph, text, backend).answer("original question")


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2"])
def test_invalid_round_limit_is_rejected(toy_graph, limit):
    with pytest.raises(ValueError, match="max_supplementation_rounds"):
        QGraphRAG(toy_graph, None, None, None, max_supplementation_rounds=limit)
