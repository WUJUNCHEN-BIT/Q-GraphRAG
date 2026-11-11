"""Parallel-ready graph and BM25+vector+Cross-Encoder retrieval branches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .io import fingerprint, save_torch
from .pcst import PCSTSelector, global_minmax, serialize_subgraph


@dataclass
class ScoredDocument:
    document_id: str
    score: float
    branch: str


@dataclass
class GraphRetrieval:
    documents: list[ScoredDocument]
    subgraph: dict
    seed_entities: list[str]
    top_entities: list[dict]


def graph_document_scores(graph_index, scores, entity_top_k, document_top_k, epsilon=1e-8):
    normalized = global_minmax(scores.detach().cpu(), epsilon)
    top_ids = torch.argsort(scores.detach().cpu(), descending=True, stable=True)[
        :entity_top_k
    ].tolist()
    selected = {graph_index.entities[i]: float(normalized[i]) for i in top_ids}
    candidates = []
    for doc in graph_index.documents:
        associated = set(graph_index.document_entities[doc.id]) & selected.keys()
        if associated:
            candidates.append(ScoredDocument(doc.id, sum(selected[e] for e in associated), "graph"))
    candidates.sort(key=lambda item: (-item.score, item.document_id))
    return candidates[:document_top_k], top_ids, normalized


class GraphRetriever:
    def __init__(self, graph_index, tensors, model, encoder, linker, config):
        self.graph_index, self.graph, self.model = graph_index, tensors, model
        self.encoder, self.linker, self.config = encoder, linker, config
        self.selector = PCSTSelector(graph_index, config["pcst"])
        self.model.eval()

    @torch.inference_mode()
    def retrieve(self, question, question_entities=None):
        seed_ids = self.linker.link(question, question_entities)
        question_embedding = self.encoder.encode([question]).to(self.graph.quality.device)
        seed_mask = question_embedding.new_zeros((1, self.graph.num_nodes))
        seed_mask[0, seed_ids] = 1
        result = self.model(self.graph, question_embedding, seed_mask, collect_support=True)
        settings = self.config["retrieval"]
        docs, top_ids, normalized = graph_document_scores(
            self.graph_index,
            result.logits[0],
            settings["entity_top_k"],
            settings["graph_doc_top_k"],
            self.config["pcst"]["epsilon"],
        )
        subgraph = self.selector.select(
            result.logits[0], result.gates, result.total_support[0], result.effective_support[0]
        )
        return GraphRetrieval(
            docs,
            serialize_subgraph(subgraph, self.graph_index, result.gates.cpu()),
            [self.graph_index.entities[i] for i in seed_ids],
            [
                {
                    "entity": self.graph_index.entities[i],
                    "logit": float(result.logits[0, i]),
                    "normalized_score": float(normalized[i]),
                }
                for i in top_ids
            ],
        )


def tokenize(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class CrossEncoderReranker:
    def __init__(self, config):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(config["reranker"], device=config["reranker_device"])
        self.batch_size = config["rerank_batch_size"]

    def score(self, question, documents):
        if not documents:
            return []
        pairs = [(question, doc.title + "\n" + doc.text) for doc in documents]
        scores = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        )
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise ValueError("Cross-Encoder must return one finite relevance score per document")
        return scores.tolist()


class TextRetriever:
    def __init__(self, graph_index, encoder, reranker, config, cache_dir):
        from rank_bm25 import BM25Okapi

        self.documents, self.encoder, self.reranker = graph_index.documents, encoder, reranker
        self.config = config
        texts = [doc.title + "\n" + doc.text for doc in self.documents]
        self.bm25 = BM25Okapi([tokenize(text) or ["__empty__"] for text in texts])
        key = fingerprint(
            {
                "documents": [[d.id, d.title, d.text] for d in self.documents],
                "encoder": encoder.signature,
                "schema": 1,
            }
        )
        path = Path(cache_dir) / ("documents_" + key + ".pt")
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data["fingerprint"] != key:
                raise ValueError("Document-embedding cache mismatch")
            embeddings = data["embeddings"]
        else:
            embeddings = encoder.encode(texts)
            save_torch(path, {"fingerprint": key, "embeddings": embeddings})
        if embeddings.shape != (len(self.documents), encoder.dimension):
            raise ValueError("Document-embedding cache dimensions mismatch")
        self.vectors = F.normalize(embeddings, dim=-1)

    def retrieve(self, question):
        lexical = np.asarray(self.bm25.get_scores(tokenize(question)))
        lexical_ids = np.argsort(-lexical, kind="stable")[: self.config["bm25_top_k"]].tolist()
        vector = F.normalize(self.encoder.encode([question]), dim=-1)
        semantic = (vector @ self.vectors.t())[0]
        vector_ids = torch.argsort(semantic, descending=True, stable=True)[
            : self.config["vector_top_k"]
        ].tolist()
        union = list(dict.fromkeys(lexical_ids + vector_ids))
        scores = self.reranker.score(question, [self.documents[i] for i in union])
        if len(scores) != len(union):
            raise ValueError("Cross-Encoder score count mismatch")
        result = [
            ScoredDocument(self.documents[i].id, float(score), "text")
            for i, score in zip(union, scores)
        ]
        result.sort(key=lambda item: (-item.score, item.document_id))
        return result[: self.config["text_top_k"]]
