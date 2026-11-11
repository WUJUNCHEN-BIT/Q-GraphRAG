"""Balanced binary and listwise retrieval losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def retrieval_losses(logits, targets, temperature=0.2):
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("Expected matching [batch,entities] logits and targets")
    positive = targets.bool()
    negative = ~positive
    if not positive.any(dim=1).all() or not negative.any(dim=1).all():
        raise ValueError(
            "Each supervised query needs at least one positive and one negative entity"
        )
    pos_count = positive.sum(-1)
    element_loss = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    pos_loss = (element_loss * positive).sum(-1) / pos_count
    with torch.no_grad():
        if temperature > 0:
            weights = torch.softmax(
                (logits / temperature).masked_fill(positive, -torch.inf), dim=-1
            )
        else:
            weights = negative / negative.sum(-1, keepdim=True)
    neg_loss = (element_loss * weights).sum(-1)
    bce = (0.5 * (pos_loss + neg_loss)).mean()
    # log(sigmoid(logits)/sum(sigmoid(logits))) without small-probability underflow.
    log_probabilities = F.logsigmoid(logits)
    log_normalizer = torch.logsumexp(log_probabilities, dim=-1, keepdim=True)
    ranking = (-((log_probabilities - log_normalizer) * positive).sum(-1) / pos_count).mean()
    return bce, ranking


def retrieval_loss(logits, targets, config):
    bce, ranking = retrieval_losses(logits, targets, config["adversarial_temperature"])
    return config["lambda_bce"] * bce + config["lambda_rank"] * ranking


def completion_loss(logits, temperature=1.0):
    targets = torch.zeros_like(logits)
    targets[:, 0] = 1
    return retrieval_losses(logits, targets, temperature)[0]
