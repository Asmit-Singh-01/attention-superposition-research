"""
Attention Analysis Utilities
"""

import torch


def extract_qk_from_model(model):

    """
    Extract Q, K, attention scores and attention weights
    from every Transformer layer.
    """

    extracted = []

    for layer_index, layer in enumerate(
        model.layers
    ):

        attention = layer.attention

        extracted.append({

            "layer": layer_index,

            "Q": attention.last_q,

            "K": attention.last_k,

            "V": attention.last_v,

            "attention_scores":
                attention.last_attention_scores,

            "attention_weights":
                attention.last_attention_weights

        })

    return extracted


def compute_attention_scores(
    Q,
    K
):

    """
    Compute scaled dot-product attention scores.
    """

    d_k = Q.size(-1)

    return torch.matmul(
        Q,
        K.transpose(-2, -1)
    ) / (
        d_k ** 0.5
    )


def attention_statistics(
    attention_weights
):

    """
    Calculate basic statistics.
    """

    return {

        "mean":
            attention_weights.mean().item(),

        "std":
            attention_weights.std().item(),

        "min":
            attention_weights.min().item(),

        "max":
            attention_weights.max().item()

    }
