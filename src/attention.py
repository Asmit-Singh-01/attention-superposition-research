"""
Attention Analysis Utilities

Utilities for extracting and analyzing Query (Q) and Key (K)
representations from Transformer attention.
"""

import torch
import torch.nn.functional as F


def compute_attention_scores(Q, K):
    """
    Compute scaled dot-product attention scores.

    Args:
        Q: Query tensor [batch, sequence, d_k]
        K: Key tensor [batch, sequence, d_k]

    Returns:
        Attention score matrix.
    """

    d_k = Q.size(-1)

    scores = torch.matmul(
        Q,
        K.transpose(-2, -1)
    ) / torch.sqrt(
        torch.tensor(d_k, dtype=Q.dtype)
    )

    return scores


def compute_attention_weights(Q, K):
    """
    Compute normalized attention weights.
    """

    scores = compute_attention_scores(Q, K)

    weights = F.softmax(
        scores,
        dim=-1
    )

    return weights


def attention_statistics(attention_weights):
    """
    Calculate basic statistics of an attention matrix.
    """

    return {
        "mean": attention_weights.mean().item(),
        "std": attention_weights.std().item(),
        "min": attention_weights.min().item(),
        "max": attention_weights.max().item()
    }
