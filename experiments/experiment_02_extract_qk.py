"""
Experiment 02: Extract Q/K Representations

Purpose:
Load the trained multi-task Transformer and collect
Query/Key representations for a fixed evaluation dataset.

These representations will later be used for geometric
and sparse-decomposition analysis.
"""

import os
import sys
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from experiments.experiment_01_train import (
    MultiTaskTransformer,
    SEQ_LEN,
    VOCAB_SIZE
)


SEED = 123

torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = (
    "models/experiment_01/"
    "multitask_transformer.pt"
)

OUTPUT_DIR = (
    "results/experiment_01"
)


def main():

    print("=" * 60)
    print("EXPERIMENT 02: Q/K EXTRACTION")
    print("=" * 60)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = MultiTaskTransformer().to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Trained model loaded.")

    # --------------------------------------------------------
    # Fixed evaluation dataset
    # --------------------------------------------------------

    num_samples = 500

    x = torch.randint(
        0,
        10,
        (
            num_samples,
            SEQ_LEN
        ),
        device=DEVICE
    )

    copy_targets = x[:, 0]

    odd_count = (
        x % 2
    ).sum(dim=1)

    parity_targets = (
        odd_count % 2
    ).long()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        copy_logits, parity_logits = model(
            x
        )

    copy_predictions = (
        copy_logits.argmax(dim=1)
    )

    parity_predictions = (
        parity_logits.argmax(dim=1)
    )

    copy_accuracy = (
        copy_predictions ==
        copy_targets
    ).float().mean().item()

    parity_accuracy = (
        parity_predictions ==
        parity_targets
    ).float().mean().item()

    print()
    print("Evaluation results:")
    print(
        "Copy accuracy:",
        round(copy_accuracy, 4)
    )

    print(
        "Parity accuracy:",
        round(parity_accuracy, 4)
    )

    # --------------------------------------------------------
    # Extract Q/K from every Transformer layer
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for layer_index, layer in enumerate(
        model.transformer.layers
    ):

        attention = layer.attention

        Q = attention.last_q
        K = attention.last_k

        attention_scores = (
            attention.last_attention_scores
        )

        attention_weights = (
            attention.last_attention_weights
        )

        output = {

            "Q": Q.cpu(),

            "K": K.cpu(),

            "attention_scores":
                attention_scores.cpu(),

            "attention_weights":
                attention_weights.cpu(),

            "inputs":
                x.cpu(),

            "copy_targets":
                copy_targets.cpu(),

            "parity_targets":
                parity_targets.cpu(),

            "copy_predictions":
                copy_predictions.cpu(),

            "parity_predictions":
                parity_predictions.cpu(),

            "copy_accuracy":
                copy_accuracy,

            "parity_accuracy":
                parity_accuracy,

            "layer":
                layer_index

        }

        path = (
            f"{OUTPUT_DIR}/"
            f"layer_{layer_index}.pt"
        )

        torch.save(
            output,
            path
        )

        print(
            f"Saved layer {layer_index}: {path}"
        )

    print()
    print("=" * 60)
    print("Q/K EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
