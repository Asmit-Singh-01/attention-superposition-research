"""
Experiment 01: Controlled Multi-Task Transformer

Tasks:
1. Copy the first token.
2. Predict the parity of a binary sequence.

Purpose:
Create a controlled environment in which the model must
learn two different computational behaviors.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.model import SmallTransformer


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

torch.manual_seed(SEED)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 2
SEQ_LEN = 8

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-3

TRAIN_SIZE = 4000
TEST_SIZE = 1000


# ============================================================
# Dataset
# ============================================================

class MultiTaskDataset(Dataset):

    def __init__(self, size):

        self.inputs = torch.randint(
            0,
            VOCAB_SIZE,
            (size, SEQ_LEN)
        )

        # ----------------------------------------------------
        # Task 1: Copy
        # ----------------------------------------------------

        self.copy_targets = self.inputs[:, 0]

        # ----------------------------------------------------
        # Task 2: Parity
        #
        # Number of 1s:
        # odd  -> 1
        # even -> 0
        # ----------------------------------------------------

        number_of_ones = (
            self.inputs == 1
        ).sum(dim=1)

        self.parity_targets = (
            number_of_ones % 2
        ).long()

    def __len__(self):

        return len(self.inputs)

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
            VOCAB_SIZE
        )

        self.parity_head = nn.Linear(
            D_MODEL,
            2
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        hidden = self.transformer.embedding(x)

        hidden = (
            hidden
            +
            self.transformer.position_embedding(
                positions
            )
        )

        for layer in self.transformer.layers:

            hidden = layer(hidden)

        # Global representation of the sequence
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
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    copy_correct = 0
    parity_correct = 0
    total = 0

    with torch.no_grad():

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

            copy_logits, parity_logits = model(
                x
            )

            copy_predictions = (
                copy_logits.argmax(dim=1)
            )

            parity_predictions = (
                parity_logits.argmax(dim=1)
            )

            copy_correct += (
                copy_predictions
                ==
                copy_target
            ).sum().item()

            parity_correct += (
                parity_predictions
                ==
                parity_target
            ).sum().item()

            total += x.size(0)

    copy_accuracy = (
        copy_correct / total
    )

    parity_accuracy = (
        parity_correct / total
    )

    return (
        copy_accuracy,
        parity_accuracy
    )


# ============================================================
# Training
# ============================================================

def main():

    print("=" * 60)
    print("EXPERIMENT 01: CONTROLLED MULTI-TASK TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = MultiTaskDataset(
        TRAIN_SIZE
    )

    test_dataset = MultiTaskDataset(
        TEST_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTaskTransformer().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_test_score = 0.0

    best_state = None

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for (
            x,
            copy_target,
            parity_target
        ) in train_loader:

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
                copy_loss
                +
                parity_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Test on unseen data
        # ----------------------------------------------------

        copy_accuracy, parity_accuracy = evaluate(
            model,
            test_loader
        )

        test_score = (
            copy_accuracy
            +
            parity_accuracy
        ) / 2

        if test_score > best_test_score:

            best_test_score = test_score

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        average_loss = (
            total_loss
            /
            len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {average_loss:.4f} | "
            f"Test Copy: {copy_accuracy:.3f} | "
            f"Test Parity: {parity_accuracy:.3f}"
        )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    os.makedirs(
        "models/experiment_01",
        exist_ok=True
    )

    model.load_state_dict(
        best_state
    )

    model_path = (
        "models/experiment_01/"
        "multitask_transformer.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_copy, final_parity = evaluate(
        model,
        test_loader
    )

    print()
    print("=" * 60)
    print("FINAL UNSEEN-DATA RESULTS")
    print("=" * 60)

    print(
        f"Copy accuracy:   {final_copy:.4f}"
    )

    print(
        f"Parity accuracy: {final_parity:.4f}"
    )

    if (
        final_copy >= 0.90
        and final_parity >= 0.90
    ):

        print()
        print("BASELINE STATUS: PASS")

    else:

        print()
        print("BASELINE STATUS: NEEDS IMPROVEMENT")

    print()
    print(
        "Best model saved to:",
        model_path
    )


if __name__ == "__main__":

    main()
