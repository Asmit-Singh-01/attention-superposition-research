# Attentional Superposition

## Can sparse feature decomposition reveal polysemantic computation inside the Query-Key geometry of Transformer attention?

This project investigates whether multiple computational features can coexist in superposition within the Query-Key (Q/K) representation space of Transformer attention, and whether sparse feature decomposition can recover and characterize these features.

The project combines ideas from:

- Transformer interpretability
- Attention mechanisms
- Sparse feature decomposition
- Linear algebra
- Representation geometry
- Mechanistic interpretability
- Causal analysis

---


## 1. Research Question

> Can sparse feature decomposition reveal polysemantic computation inside the Query-Key geometry of Transformer attention?

More specifically, this research investigates whether multiple computationally distinct features are represented in overlapping directions within the Q/K representation space of Transformer attention.

The study further examines whether sparse decomposition methods can disentangle these features and whether the recovered features have measurable relationships with attention behavior and computational functions.

---

## 2. Research Hypothesis

### Primary Hypothesis

Multiple computational features may be represented in superposition within the Query-Key geometry of Transformer attention.

If this is the case, sparse feature decomposition may recover a set of latent features that are more interpretable and computationally meaningful than the original dense representation.

### Testable Prediction

If polysemantic computation exists within Q/K representations, then sparse decomposition should be able to:

1. Recover latent features associated with distinct computational behaviors.
2. Produce sparse representations with meaningful feature selectivity.
3. Explain a measurable portion of attention score variation.
4. Reveal feature overlap that is not easily observable in the original dense representation.
5. Allow specific feature interventions to produce measurable changes in attention behavior or task performance.

These predictions will be tested experimentally rather than assumed to be true.

---

## 3. Mathematical Background

Transformer attention is based on Query (Q), Key (K), and Value (V) representations.

The attention score matrix is given by:

A(Q,K) = softmax(QK^T / sqrt(d_k))

where:

- Q represents Query vectors.
- K represents Key vectors.
- d_k is the dimensionality of the Key vectors.

This project focuses primarily on the geometry of Q and K representations.

We investigate whether a dense representation X can be expressed approximately as a sparse combination of latent features:

X ≈ Σ(a_i f_i)

where:

- X is the original representation.
- f_i represents a latent feature.
- a_i represents the activation of that feature.

The central question is whether these latent features correspond to meaningful computational properties and whether multiple computational functions are encoded in overlapping feature representations.

---

## 4. Research Objectives

The project aims to:

1. Study the geometry of Query-Key representations in Transformer attention.
2. Investigate whether computationally distinct features coexist in superposition.
3. Apply sparse feature decomposition to Q/K representations.
4. Compare sparse decomposition against baseline dimensionality-reduction methods.
5. Develop quantitative measures of feature sparsity, purity, overlap, and interpretability.
6. Investigate whether recovered features can explain attention behavior.
7. Test whether feature-level interventions causally affect model computation.
8. Validate findings first in controlled synthetic settings and later in real Transformer models.

---

## 5. Experimental Strategy

The research will be conducted in multiple stages.

### Experiment 1 — Baseline Attention Geometry

Extract Query and Key representations from a small Transformer model.

Analyze:

- Q/K dimensionality
- Pairwise geometry
- Attention score distributions
- Representation structure

---

### Experiment 2 — Controlled Synthetic Environment

Train or construct a controlled model involving multiple computational tasks.

Potential tasks include:

- Number comparison
- Parity detection
- Token copying
- Pattern recognition

Because the underlying computational tasks are known, this environment provides a controlled setting for testing whether multiple features become superposed.

---

### Experiment 3 — Sparse Feature Decomposition

Apply sparse feature decomposition to Q/K representations.

Compare the resulting representations with baseline methods such as:

- PCA
- Random projection
- Sparse Autoencoder

The objective is to determine whether sparse decomposition provides more meaningful feature recovery.

---

### Experiment 4 — Feature Quality Analysis

Evaluate discovered features using quantitative metrics such as:

- Reconstruction Error
- Sparsity
- Feature Purity
- Feature Overlap
- Computational Selectivity

The exact definitions of these metrics will be developed and justified during the research process.

---

### Experiment 5 — Attention Explanation

Investigate whether recovered latent features can explain variations in attention scores.

The analysis will examine relationships between:

- Latent feature activations
- Query-Key interactions
- Attention scores
- Computational behaviors

---

### Experiment 6 — Causal Feature Intervention

Test whether individual latent features have a causal effect on model behavior.

A feature may be selectively suppressed or modified, after which changes in:

- Attention patterns
- Model representations
- Task performance

will be measured.

This experiment is intended to distinguish simple correlation from potentially causal computational roles.

---

### Experiment 7 — Real Transformer Validation

If the findings from controlled experiments are supported, the methodology will be evaluated on a real Transformer model.

The purpose is to determine whether the observed phenomenon generalizes beyond the controlled experimental environment.

---

## 6. Potential Contribution

The project aims to investigate a possible mathematical and experimental framework for studying polysemantic computation within Transformer attention.

Potential contributions include:

- A geometric analysis of Q/K representation superposition.
- A sparse decomposition framework for attention representations.
- Quantitative metrics for measuring feature overlap and polysemanticity.
- Experimental evidence regarding the relationship between latent features and attention computation.
- Causal analysis of recovered features through feature-level intervention.

These are research objectives and potential contributions. They will only be claimed as findings if supported by experimental evidence.

---

## 7. Project Structure

```text
attention-superposition-research/
│
├── data/
│   └── Dataset and preprocessing files
│
├── models/
│   └── Model configurations and saved model files
│
├── notebooks/
│   └── Exploratory analysis and visualization notebooks
│
├── src/
│   └── Reusable research code
│
├── experiments/
│   └── Experiment-specific scripts
│
├── results/
│   └── Experimental results and metrics
│
├── figures/
│   └── Research visualizations and plots
│
├── paper/
│   └── Research paper drafts and notes
│
└── README.md



### Phase 0 — Project Setup
- [x] Repository created
- [x] Project structure created
- [x] Research README created
- [x] Initial research notes created

### Phase 1 — Baseline Infrastructure
- [x] Small Transformer architecture defined
- [x] Attention analysis utilities created
- [x] Representation analysis utilities created
- [x] Baseline experiment script created
- [ ] Q/K extraction hooks
- [ ] Model training
- [ ] Baseline attention analysis

- [x] Custom Multi-Head Attention implemented
- [x] Explicit Q/K/V extraction implemented
- [x] Attention score computation implemented
- [x] Attention weight extraction implemented
- [x] Baseline Q/K data-saving pipeline created
