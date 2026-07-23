"""
Custom Transformer Model

This module defines a small Transformer with explicit
Query (Q), Key (K), and Value (V) extraction.

The model is designed for interpretability experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with explicit Q/K/V access.
    """

    def __init__(
        self,
        d_model=64,
        n_heads=4
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_projection = nn.Linear(
            d_model,
            d_model
        )

        self.k_projection = nn.Linear(
            d_model,
            d_model
        )

        self.v_projection = nn.Linear(
            d_model,
            d_model
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model
        )

        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_attention_scores = None
        self.last_attention_weights = None

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        Q = self.q_projection(x)
        K = self.k_projection(x)
        V = self.v_projection(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / (
            self.head_dim ** 0.5
        )

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.output_projection(
            output
        )

        self.last_q = Q.detach()
        self.last_k = K.detach()
        self.last_v = V.detach()

        self.last_attention_scores = (
            scores.detach()
        )

        self.last_attention_weights = (
            attention_weights.detach()
        )

        return output


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=64,
        n_heads=4
    ):
        super().__init__()

        self.attention = CustomMultiHeadAttention(
            d_model,
            n_heads
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.feed_forward = nn.Sequential(

            nn.Linear(
                d_model,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                d_model
            )
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        attention_output = self.attention(
            x
        )

        x = self.norm1(
            x + attention_output
        )

        ff_output = self.feed_forward(
            x
        )

        x = self.norm2(
            x + ff_output
        )

        return x


class SmallTransformer(nn.Module):

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

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.layers = nn.ModuleList([

            TransformerBlock(
                d_model,
                n_heads
            )

            for _ in range(n_layers)

        ])

        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        x = self.embedding(x)

        x = x + self.position_embedding(
            positions
        )

        for layer in self.layers:

            x = layer(x)

        logits = self.output(x)

        return logits
