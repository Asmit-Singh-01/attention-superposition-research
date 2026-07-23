"""
Experiment 01B: Save Baseline Q/K Data

This experiment saves extracted Q/K representations
and attention matrices for later analysis.
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

from src.attention import (
    extract_qk_from_model
)


def main():

    torch.manual_seed(42)


    # Configuration

    vocab_size = 100

    sequence_length = 16

    batch_size = 4


    # Model

    model = SmallTransformer(

        vocab_size=100,

        d_model=64,

        n_heads=4,

        n_layers=2,

        max_seq_len=32

    )


    # Input

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

        model(x)


    # Extract

    extracted = extract_qk_from_model(
        model
    )


    # Create results directory

    os.makedirs(

        "results/baseline",

        exist_ok=True

    )


    # Save each layer

    for item in extracted:

        layer = item["layer"]


        torch.save(

            item,

            f"results/baseline/"
            f"layer_{layer}.pt"

        )


    print(

        "Baseline Q/K data saved successfully."

    )


if __name__ == "__main__":

    main()
