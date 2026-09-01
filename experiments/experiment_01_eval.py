"""
Experiment 01 Evaluation

Tests the trained model on NEW, unseen synthetic examples.

Purpose:
Verify that the model learned the computational tasks
rather than simply memorizing the training examples.
"""

import os
import sys
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from experiments.experiment_01_train import (
    MultiTaskTransformer,
    SEQ_LEN,
    DEVICE
)


MODEL_PATH = (
    "models/experiment_01/"
    "multitask_transformer.pt"
)

NUM_TEST_SAMPLES = 1000


def main():

    print("=" * 60)
    print("EXPERIMENT 01: UNSEEN-DATA EVALUATION")
    print("=" * 60)

    model = MultiTaskTransformer().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    # Completely new examples
    x = torch.randint(
        0,
        10,
        (NUM_TEST_SAMPLES, SEQ_LEN),
        device=DEVICE
    )

    copy_targets = x[:, 0]

    odd_count = (
        x % 2
    ).sum(dim=1)

    parity_targets = (
        odd_count % 2
    ).long()

    with torch.no_grad():

        copy_logits, parity_logits = model(x)

    copy_predictions = (
        copy_logits.argmax(dim=1)
    )

    parity_predictions = (
        parity_logits.argmax(dim=1)
    )

    copy_accuracy = (
        copy_predictions == copy_targets
    ).float().mean().item()

    parity_accuracy = (
        parity_predictions == parity_targets
    ).float().mean().item()

    print()
    print("UNSEEN DATA RESULTS")
    print("-------------------")

    print(
        f"Copy accuracy:   {copy_accuracy:.4f}"
    )

    print(
        f"Parity accuracy: {parity_accuracy:.4f}"
    )

    print()

    if (
        copy_accuracy >= 0.90
        and parity_accuracy >= 0.90
    ):

        print("BASELINE STATUS: PASS")

    else:

        print("BASELINE STATUS: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
