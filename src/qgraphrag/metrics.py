"""Support-document Recall@K and standard normalized answer EM/token F1."""

from __future__ import annotations

import re
import string
from collections import Counter

from .training import support_document_ids


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, references):
    predicted = normalize_answer(prediction)
    exact, best_f1 = 0.0, 0.0
    for reference in references:
        gold = normalize_answer(reference)
        exact = max(exact, float(predicted == gold))
        if (
            predicted in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}
        ) and predicted != gold:
            continue
        pred_tokens, gold_tokens = predicted.split(), gold.split()
        if not pred_tokens or not gold_tokens:
            best_f1 = max(best_f1, float(pred_tokens == gold_tokens))
            continue
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap:
            precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return exact, best_f1


def evaluate_predictions(predictions, queries, graph, k=5):
    if not queries or k < 1:
        raise ValueError("Evaluation requires nonempty queries and positive K")
    by_id = {str(row["id"]): row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    ids = {str(row["id"]) for row in queries}
    if len(ids) != len(queries) or set(by_id) != ids:
        raise ValueError("Predictions and evaluation query IDs must match exactly")
    recall_sum, em_sum, f1_sum = 0.0, 0.0, 0.0
    for row in queries:
        prediction = by_id[str(row["id"])]
        support = set(support_document_ids(row, graph))
        retrieved = prediction["retrieved_document_ids"][:k]
        if len(retrieved) != len(set(retrieved)):
            raise ValueError("Duplicate final retrieved documents")
        recall_sum += len(support & set(retrieved)) / len(support)
        answer = row.get("answer")
        if not isinstance(answer, (str, int, float)):
            raise ValueError("Evaluation requires a reference answer")
        references = [str(answer)] + [str(x) for x in row.get("answer_aliases", [])]
        em, f1 = answer_scores(prediction["answer"], references)
        em_sum += em
        f1_sum += f1
    count = len(queries)
    return {
        "queries": count,
        f"recall@{k}": 100 * recall_sum / count,
        "em": 100 * em_sum / count,
        "f1": 100 * f1_sum / count,
        "units": "percent",
    }
