"""
Feature Decomposition Utilities

This module contains baseline methods for analyzing
representation spaces before introducing sparse decomposition.
"""

import torch


def reconstruction_error(
    original,
    reconstructed
):
    """
    Calculate mean squared reconstruction error.
    """

    error = torch.mean(
        (original - reconstructed) ** 2
    )

    return error.item()


def calculate_sparsity(
    activations,
    threshold=1e-5
):
    """
    Calculate the fraction of near-zero activations.

    Higher values indicate a sparser representation.
    """

    near_zero = (
        torch.abs(activations) < threshold
    ).float()

    return near_zero.mean().item()


def cosine_similarity(
    x,
    y
):
    """
    Calculate cosine similarity between two tensors.
    """

    x_flat = x.flatten()
    y_flat = y.flatten()

    similarity = torch.nn.functional.cosine_similarity(
        x_flat.unsqueeze(0),
        y_flat.unsqueeze(0)
    )

    return similarity.item()
