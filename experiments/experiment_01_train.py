"""
Experiment 01: Controlled Multi-Task Transformer

Tasks:
1. Copy a selected token.
2. Determine parity of the number of odd tokens.

Purpose:
Train a small Transformer on two known computational tasks
before analysing its Query-Key representation space.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.model import SmallTransformer


# ============================================================
# Configuration
# ============================================================

SEED = 42

torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 20
SEQ_LEN = 8

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 1e-3


# ============================================================
# Dataset
# ============================================================

class MultiTaskDataset(Dataset):

    def __init__(self, size=2000):

        self.size = size

        self.inputs = torch.randint(
            0,
            10,
            (size, SEQ_LEN)
        )

        # Copy target:
        # copy the token at position 0
        self.copy_targets = self.inputs[:, 0]

        # Parity target:
        # 1 = odd number of odd tokens
        # 0 = even number of odd tokens
        odd_count = (
            self.inputs % 2
        ).sum(dim=1)

        self.parity_targets = (
            odd_count % 2
        ).long()

    def __len__(self):

        return self.size

    def __getitem__(self, index):

        return (
            self.inputs[index],
            self.copy_targets[index],
            self.parity_targets[index]
        )


# ============================================================
# Multi-task model
# ============================================================

class MultiTaskTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.transformer = SmallTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            max_seq_len=SEQ_LEN
        )

        self.copy_head = nn.Linear(
            D_MODEL,
            10
        )

        self.parity_head = nn.Linear(
            D_MODEL,
            2
        )

    def forward(self, x):

        # Reproduce embedding + transformer processing
        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        hidden = self.transformer.embedding(x)

        hidden = hidden + self.transformer.position_embedding(
            positions
        )

        for layer in self.transformer.layers:

            hidden = layer(hidden)

        # Mean pooling gives one representation
        # for the complete sequence.
        pooled = hidden.mean(dim=1)

        copy_logits = self.copy_head(
            pooled
        )

        parity_logits = self.parity_head(
            pooled
        )

        return (
            copy_logits,
            parity_logits
        )


# ============================================================
# Training
# ============================================================

def main():

    print("=" * 60)
    print("EXPERIMENT 01: CONTROLLED MULTI-TASK TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    dataset = MultiTaskDataset(
        size=2000
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MultiTaskTransformer().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        copy_correct = 0
        parity_correct = 0
        total = 0

        for (
            x,
            copy_target,
            parity_target
        ) in loader:

            x = x.to(DEVICE)

            copy_target = copy_target.to(
                DEVICE
            )

            parity_target = parity_target.to(
                DEVICE
            )

            optimizer.zero_grad()

            copy_logits, parity_logits = model(
                x
            )

            copy_loss = loss_function(
                copy_logits,
                copy_target
            )

            parity_loss = loss_function(
                parity_logits,
                parity_target
            )

            loss = (
                copy_loss +
                parity_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            copy_predictions = (
                copy_logits.argmax(dim=1)
            )

            parity_predictions = (
                parity_logits.argmax(dim=1)
            )

            copy_correct += (
                copy_predictions ==
                copy_target
            ).sum().item()

            parity_correct += (
                parity_predictions ==
                parity_target
            ).sum().item()

            total += x.size(0)

        copy_accuracy = (
            copy_correct / total
        )

        parity_accuracy = (
            parity_correct / total
        )

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Copy: {copy_accuracy:.3f} | "
            f"Parity: {parity_accuracy:.3f}"
        )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    os.makedirs(
        "models/experiment_01",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "models/experiment_01/"
        "multitask_transformer.pt"
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Model saved to:"
        " models/experiment_01/"
        "multitask_transformer.pt"
    )


if __name__ == "__main__":
    main()
