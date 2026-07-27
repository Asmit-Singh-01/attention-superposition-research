# Research Notes

## Core Research Question

Can sparse feature decomposition reveal polysemantic computation
inside the Query-Key geometry of Transformer attention?

---

## Initial Hypothesis

Multiple computationally distinct features may coexist in superposition
within the Query-Key representation space of Transformer attention.

---

## Important Principle

The hypothesis must be experimentally tested.

We must not assume that polysemanticity exists before collecting evidence.

---

## Key Mathematical Object

Attention:

A(Q,K) = softmax(QK^T / sqrt(d_k))

---

## Research Focus

We investigate:

1. Q/K representation geometry.
2. Feature superposition.
3. Sparse feature decomposition.
4. Feature overlap.
5. Attention explanation.
6. Causal feature intervention.

---

## Potential Metrics

- Reconstruction Error
- Sparsity
- Feature Purity
- Feature Overlap
- Computational Selectivity
- Attention Explanation Score

---

## Open Questions

1. How should polysemanticity be mathematically defined?
2. How can feature overlap be quantified?
3. Can sparse decomposition recover known synthetic features?
4. Do recovered features explain attention behavior?
5. Does feature intervention causally affect computation?



---

## Experimental Architecture

The initial experimental pipeline consists of:

Input Tokens
↓
Small Transformer
↓
Query-Key Representations
↓
Attention Score Matrix
↓
Representation Geometry Analysis
↓
Sparse Feature Decomposition
↓
Feature-Level Evaluation
↓
Causal Intervention

The first stage focuses on establishing a reproducible baseline
before testing the central superposition hypothesis.


---

# Research Concept: Understanding the Core Idea

## The Problem

Transformer models can perform many different computational tasks.
However, it is not always clear how these computations are represented
inside the model's internal representation spaces.

A single representation may potentially encode multiple computational
features at the same time.

This phenomenon is related to the broader concept of superposition.

---

## The Core Idea of This Project

This project investigates whether multiple computational features can
coexist in overlapping regions or directions of the Query-Key (Q/K)
representation space of Transformer attention.

The project does not assume that such superposition exists.

Instead, it aims to experimentally test the hypothesis.

---

## Why Query-Key Geometry?

Transformer attention uses Query and Key representations to determine
relationships between tokens.

The attention score is computed as:

Attention Score = QK^T / sqrt(d_k)

Therefore, the geometry and interactions between Q and K representations
directly influence attention patterns.

This makes the Q/K representation space a potentially important place
to investigate computational feature interactions.

---

## Role of Sparse Feature Decomposition

Sparse feature decomposition is used as an analysis tool.

The goal is to investigate whether a dense Q/K representation can be
approximately represented using a smaller set of sparse, interpretable
latent features.

The project will investigate whether these recovered features
correspond to distinct computational behaviors.

---

## Central Research Question

Can sparse feature decomposition reveal polysemantic computation
inside the Query-Key geometry of Transformer attention?

---

## Research Pipeline

Input
↓
Transformer
↓
Query-Key Representations
↓
Q/K Geometry Analysis
↓
Feature Decomposition
↓
Feature Identification
↓
Computational Behavior Analysis
↓
Causal Intervention
↓
Conclusion

---

## Important Research Principle

The project will not assume that polysemanticity or superposition
exists in Q/K representations.

The existence, strength, and computational relevance of such phenomena
must be determined through controlled experiments.

Negative or inconclusive results will be considered valid research
outcomes if the experimental methodology is rigorous.
