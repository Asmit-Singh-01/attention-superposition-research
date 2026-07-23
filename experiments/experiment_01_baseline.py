"""
Experiment 01: Baseline Attention Geometry

This experiment verifies that Query and Key representations
can be extracted from the custom Transformer.
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
    extract_qk_from_model,
    attention_statistics
)


def main():

    print("=" * 60)

    print(
        "EXPERIMENT 01: Q/K EXTRACTION"
    )

    print("=" * 60)


    torch.manual_seed(42)


    # Configuration

    vocab_size = 100

    sequence_length = 16

    batch_size = 4

    d_model = 64

    n_heads = 4

    n_layers = 2


    # Create model

    model = SmallTransformer(

        vocab_size=vocab_size,

        d_model=d_model,

        n_heads=n_heads,

        n_layers=n_layers,

        max_seq_len=32

    )


    # Random input

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


    # Extract representations

    extracted = extract_qk_from_model(
        model
    )


    print("\nModel output shape:")

    print(
        logits.shape
    )


    print(

        "\nNumber of Transformer layers:",

        len(extracted)

    )


    for item in extracted:

        layer = item["layer"]

        Q = item["Q"]

        K = item["K"]

        attention = item[
            "attention_weights"
        ]


        print(

            f"\nLayer {layer}"

        )


        print(

            "Q shape:",

            Q.shape

        )


        print(

            "K shape:",

            K.shape

        )


        print(

            "Attention shape:",

            attention.shape

        )


        stats = attention_statistics(
            attention
        )


        print(

            "Attention statistics:",

            stats

        )


    print(

        "\nSUCCESS: Q/K extraction pipeline works."

    )


if __name__ == "__main__":

    main()
