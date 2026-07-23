"""
Baseline Transformer Model

This module defines a small Transformer model for controlled
experiments on Query-Key representation geometry.
"""

import torch
import torch.nn as nn


class SmallTransformer(nn.Module):
    """
    Small Transformer model for research experiments.

    The model is intentionally lightweight so that experiments
    can be run in constrained environments.
    """

    def __init__(
        self,
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Token IDs [batch, sequence]

        Returns:
            logits: Model predictions
        """

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        x = self.embedding(x)

        x = x + self.position_embedding(positions)

        x = self.transformer(x)

        logits = self.output(x)

        return logits
