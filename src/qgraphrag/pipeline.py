"""Stage 3: parallel retrieval, bounded supplementation, evidence selection and QA."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from .schema import nonempty


class QGraphRAG:
    def __init__(
        self,
        graph_index,
        graph_retriever,
        text_retriever,
        backend,
        final_top_k=5,
        max_supplementation_rounds=2,
    ):
        if type(max_supplementation_rounds) is not int or max_supplementation_rounds < 0:
            raise ValueError("max_supplementation_rounds must be a nonnegative integer")
        self.graph = graph_index
        self.documents = {d.id: d for d in graph_index.documents}
        self.graph_retriever, self.text_retriever, self.backend = (
            graph_retriever,
            text_retriever,
            backend,
        )
        self.final_top_k = final_top_k
        self.max_supplementation_rounds = max_supplementation_rounds

    def _documents(self, document_ids):
        return [asdict(self.documents[doc_id]) for doc_id in document_ids]

    @staticmethod
    def _reasoning_context(subgraph):
        return {
            "nodes": [{"id": node["id"], "name": node["name"]} for node in subgraph["nodes"]],
            "edges": [
                {key: edge[key] for key in ("head", "relation", "tail", "qualifiers")}
                for edge in subgraph["edges"]
            ],
        }

    def answer(self, question, question_entities=None, question_id=None):
        nonempty(question, "question")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            graph_future = pool.submit(self.graph_retriever.retrieve, question, question_entities)
            text_future = pool.submit(self.text_retriever.retrieve, question)
            graph_result, text_result = graph_future.result(), text_future.result()
        retrieved_at = time.perf_counter()
        reasoning = self._reasoning_context(graph_result.subgraph)
        initial = list(
            dict.fromkeys(
                [d.document_id for d in graph_result.documents]
                + [d.document_id for d in text_result]
            )
        )
        if not initial:
            raise ValueError("Both retrieval branches returned no evidence documents")
        history = []

        def assess(document_ids, completed_rounds):
            remaining = self.max_supplementation_rounds - completed_rounds

            def validate(value):
                if type(value.get("sufficient")) is not bool:
                    raise ValueError("sufficient must be a boolean")
                if not value["sufficient"] and remaining > 0:
                    nonempty(value.get("supplement_query"), "supplement_query")

            assessment = self.backend.complete(
                "sufficiency",
                {
                    "question": question,
                    "subgraph": reasoning,
                    "documents": self._documents(document_ids),
                    "supplementation_round": completed_rounds,
                    "remaining_rounds": remaining,
                },
                validate,
            )
            history.append(
                {
                    "round": completed_rounds,
                    "document_ids": list(document_ids),
                    "assessment": assessment,
                }
            )
            return assessment

        augmented = list(initial)
        rounds = []
        supplementary = {}
        assessment = assess(augmented, 0)
        while not assessment["sufficient"] and len(rounds) < self.max_supplementation_rounds:
            query = assessment["supplement_query"]
            candidates = self.text_retriever.retrieve(query)
            previous = set(augmented)
            augmented = list(dict.fromkeys(augmented + [d.document_id for d in candidates]))
            added = [doc_id for doc_id in augmented if doc_id not in previous]
            for candidate in candidates:
                supplementary.setdefault(candidate.document_id, candidate)
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "query": query,
                    "candidates": [asdict(d) for d in candidates],
                    "added_document_ids": added,
                }
            )
            assessment = assess(augmented, len(rounds))
            if not added:
                break
        if assessment["sufficient"]:
            stop_reason = "sufficient"
        elif rounds and not rounds[-1]["added_document_ids"]:
            stop_reason = "no_new_documents"
        else:
            stop_reason = "max_rounds"
        expected = min(self.final_top_k, len(augmented))

        def validate_selection(value):
            ids = value.get("document_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != expected
                or any(not isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids)
                or not set(ids) <= set(augmented)
            ):
                raise ValueError(f"Select exactly {expected} distinct IDs from the supplied pool")

        selection = self.backend.complete(
            "select",
            {
                "question": question,
                "subgraph": reasoning,
                "documents": self._documents(augmented),
                "final_top_k": self.final_top_k,
            },
            validate_selection,
        )
        final = selection["document_ids"]

        def validate_answer(value):
            nonempty(value.get("answer"), "answer")

        response = self.backend.complete(
            "answer",
            {
                "question": question,
                "subgraph": reasoning,
                "documents": self._documents(final),
            },
            validate_answer,
        )
        end = time.perf_counter()
        return {
            "id": question_id,
            "question": question,
            "answer": response["answer"],
            "retrieved_document_ids": final,
            "documents": self._documents(final),
            "subgraph": graph_result.subgraph,
            "trace": {
                "graph_candidates": [asdict(d) for d in graph_result.documents],
                "text_candidates": [asdict(d) for d in text_result],
                "initial_document_ids": initial,
                "sufficiency": assessment,
                "sufficiency_history": history,
                "supplementary_candidates": [asdict(d) for d in supplementary.values()],
                "supplementation_rounds": rounds,
                "supplementation_stop_reason": stop_reason,
                "augmented_document_ids": augmented,
                "seed_entities": graph_result.seed_entities,
                "top_entities": graph_result.top_entities,
                "supplementation_steps": len(rounds),
                "retrieval_seconds": retrieved_at - start,
                "total_seconds": end - start,
            },
        }
