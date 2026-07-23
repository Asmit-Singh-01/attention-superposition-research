"""
Experiment 01: Baseline Attention Geometry

This experiment creates a small Transformer and generates
baseline representations for future Q/K geometry analysis.

IMPORTANT:
This script is currently an infrastructure prototype.
Final experimental conclusions require trained models
and controlled datasets.
"""

import torch

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.model import SmallTransformer


def main():

    print("=" * 50)
    print("EXPERIMENT 01: BASELINE TRANSFORMER")
    print("=" * 50)

    # Reproducibility
    torch.manual_seed(42)

    # Model configuration
    vocab_size = 100
    sequence_length = 16
    batch_size = 4

    # Create model
    model = SmallTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )

    # Generate synthetic token input
    x = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    # Forward pass
    with torch.no_grad():

        logits = model(x)

    print("\nModel created successfully.")

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        logits.shape
    )

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("\nExperiment infrastructure ready.")


if __name__ == "__main__":
    main()
