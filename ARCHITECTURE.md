# Architecture

This document describes the SPECTRA model and the design decisions behind it. It also records how the architecture evolved, since the exploration itself is part of the contribution.

## 1. Problem

Given a TCR (α and β chains, typically summarized by CDR3α/CDR3β) and a peptide-MHC, predict whether they bind. SPECTRA treats this as binary classification with a probability output, trained on modeled 3D complexes so it scales beyond the small set of experimental crystal structures.

## 2. The three modalities

### 2.1 Sequence — ESM-2 with LoRA
Each chain is embedded by a pretrained ESM-2 protein language model. Rather than treating ESM as a frozen feature extractor, SPECTRA supports **post-training** it:

- frozen (precomputed embeddings),
- partial unfreeze (tune the last *N* transformer layers),
- full fine-tune,
- **LoRA** — low-rank adapters, optionally *chain-specific* (a shared backbone with separate adapters for the highly diverse TCR vs. the more conserved pMHC).

Pooling is masked-mean or `[CLS]`, projected to a common model dimension.

### 2.2 Structure — EGNN or heterogeneous graph transformer
The complex is a graph over residues (Cα coordinates + node/edge features). Two interchangeable backbones:

- **EGNN** — an E(n)-equivariant GNN; rotation/translation-equivariant by construction. Node features may be one-hot amino acids, a learned embedding, or **ESM residue embeddings injected as node features** (coupling the sequence and structure views).
- **Pseudo-heterogeneous graph transformer** — maintains *separate* message-passing weights for each of the nine biological edge types across the interface (tcr↔tcr, tcr↔pep, tcr↔mhc, pep↔pep, pep↔mhc, mhc↔mhc, …), so the model can learn interface-specific interaction rules. Inspired by STAG-LLM's `psudo_hetero_transformer`, reimplemented for DGL graphs.

### 2.3 Energetics — Rosetta interface features
Twelve scalar interface descriptors — `sc_value`, `hbonds_int`, `dG_separated_per_dSASA`, `per_residue_energy_int`, `dSASA_int/hphobic/polar`, `fa_atr/sol/elec/rep`, `nres_int` — are encoded by a residual MLP. These give a cheap, physically grounded signal that complements the learned representations.

## 3. Fusion

- **Cross-attention.** Bidirectional multi-head attention lets peptide attend to MHC, TCRα to TCRβ, and TCR to pMHC; a structure↔sequence variant lets graph nodes attend to sequence residues and vice versa.
- **Gated fusion.** A learned **per-dimension sigmoid gate** blends the sequence, structure, and energetics representations, `fused = λ⊙a + (1−λ)⊙b`. A `*_available` mask makes the gate fall back to the present modalities when structure or Rosetta features are missing.
- **Head.** A residual-MLP classifier produces the binding logit.

## 4. Training

- **Loss:** class-weighted BCE (`pos_weight` for imbalance), logits clamped for stability.
- **Multi-task pretraining (two stages):** Stage 1 — Rosetta ΔG regression on binders + binding classification, ESM frozen, all structures; Stage 2 — refine on high-confidence (crystal) geometries, optionally unfreezing the last ESM layers with a lower LR.
- **Ensembling:** train N seeds; combine by probability averaging, logit averaging, majority vote, rank averaging, or learned stacking. Prediction variance across members is a natural uncertainty estimate.
- **Reproducibility:** global seeding, leak-free peptide/TCR-aware splits, config-driven runs.

## 5. Ablation matrix (modes A–H)

A single unified model exposes eight configurations so each component's value can be isolated:

| Mode | Sequence | Structure/Chains | Cross-attn | Rosetta |
|------|----------|------------------|-----------|---------|
| A | concat → `[CLS]` | 1 chain | – | – |
| B | per-chain pool | 4 chains | – | – |
| C | per-chain pool | 4 chains | – | ✓ |
| D | per-chain pool | 4 chains | ✓ | – |
| **E** | per-chain pool | 4 chains | ✓ | ✓ (**full**) |
| F | concat → `[CLS]` | 1 chain | – | ✓ |
| G | per-chain pool | 2 chains (pMHC / TCR) | – | – |
| H | per-chain pool | 2 chains | – | ✓ |

Controlled comparisons: A→B (positional-encoding effect), B→D (value of cross-attention), A→G→B (chain granularity vs. compute), and A→F / B→C / D→E / G→H (value of Rosetta features).

## 6. Design evolution

The final design converged through a deliberate sequence of experiments:

1. **v1** — GRU-VAE sequence encoders + EGNN structure + scalar gated fusion (the ImmunoStruct lineage, repointed at TCR-pMHC binding).
2. **v2** — replaced shallow GRUs with a Pre-LN Transformer + GRU summary; added a BatchNorm/residual classifier head; encoder-type flag for ablation.
3. **Model B** — pivoted the sequence branch to **ESM-2**, dropping the VAE reconstruction/KL; introduced **LoRA** with chain-specific adapters.
4. **Model B2** — swapped EGNN for the **pseudo-heterogeneous graph transformer** (per-edge-type message passing).
5. **Model B3** — **injected ESM embeddings as graph node features**, fully coupling the sequence and structure views.
6. **Single-modality controls** — sequence-only and structure-only models.
7. **Rosetta multi-task pretraining** — two-stage ΔG + binding objective.
8. **Flagship** — consolidated model: ESM+LoRA, EGNN with selectable node features, bidirectional structure↔sequence cross-attention, per-dimension `VectorGatedFusion`, residual classifier.
9. **Energetics pivot + ablation** — replaced the graph branch with 12 Rosetta interface features and unified everything into the A–H ablation model, plus multi-seed ensembling.

Two design principles recur throughout: **learned gated fusion with graceful missing-modality fallback**, and **controlled ablation treated as a first-class artifact**.

## 7. Original contributions vs. prior work

Reused foundations: the EGNN/VAE multimodal scaffold (ImmunoStruct) and the per-edge-type graph transformer idea (STAG-LLM), plus the ESM-2 backbone. SPECTRA's own contributions are the **integration** — ESM-2 + chain-specific LoRA post-training, the tri-modal gated cross-attention fusion (language + structure + energetics), the two-stage Rosetta multi-task pretraining, and the A–H ablation + ensembling methodology.
