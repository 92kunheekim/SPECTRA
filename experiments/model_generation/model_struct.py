"""
StructureOnlyModel — Graph-Only Ablation (No ESM Sequence Branch)
==================================================================

Ablation model that keeps ONLY the pseudo-heterogeneous graph transformer,
removing the entire ESM sequence branch. Node features are one-hot AA
encodings (20-dim) — no ESM token embeddings.

This isolates the contribution of the structure encoder and 3D contact
graph for TCR-pMHC binding prediction.

Architecture:
  1. HeteroStructureEncoder:
       node_proj(one-hot AA) → EdgeMLP → N × PseudoHeteroTransformerLayer
       → global mean pool → g_emb [B, struct_out_dim]
  2. Projection:  Linear(struct_out_dim, d_fused) + LayerNorm + GELU
  3. Classifier:  residual MLP → binding logit

Removed (vs model_llm2):
  - ESMSequenceEncoder (entire ESM-2 backbone)
  - LoRA adapters
  - pmhc_proj
  - CrossAttention (TCR ↔ pMHC sequence cross-attention)
  - StructureSequenceCrossAttention (graph ↔ sequence)
  - GatedFusion (replaced by simple Linear projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# Import shared structure components from model_llm2
from model_llm2 import (
    HeteroStructureEncoder,
    global_mean_pool,
    EDGE_TYPES,
    ETYPE_TO_IDX,
    NUM_EDGE_TYPES,
)


# ============================================================
# StructureOnlyModel
# ============================================================

class StructureOnlyModel(nn.Module):
    """
    Structure-only ablation model for TCR-pMHC binding prediction.

    Uses the pseudo-heterogeneous graph transformer with one-hot AA
    node features (20-dim), 3D contact edges, and 5 biological edge
    types. No ESM representations, no sequence information beyond
    amino acid identity at each node.
    """

    def __init__(
        self,
        # Structure encoder
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        struct_hidden_dim: int = 320,
        struct_edge_hidden: int = 32,
        struct_n_layers: int = 3,
        struct_n_heads: int = 4,
        struct_out_dim: int = 128,
        # Fusion / classifier
        d_fused: int = 256,
        clf_hidden: int = 256,
        dropout: float = 0.2,
        norm_type: str = "layernorm",
        # Loss
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ===== Structure branch: Pseudo-heterogeneous graph transformer =====
        self.struct_encoder = HeteroStructureEncoder(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            hidden_dim=struct_hidden_dim,
            edge_hidden_dim=struct_edge_hidden,
            n_layers=struct_n_layers,
            n_heads=struct_n_heads,
            d_out=struct_out_dim,
            dropout=dropout,
        )

        # ===== Projection (graph embedding → classifier input) =====
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_out_dim, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ===== Classifier (configurable norm + residual) =====
        self.norm_type = norm_type

        def _make_norm(dim):
            if norm_type == "batchnorm":
                return nn.BatchNorm1d(dim)
            return nn.LayerNorm(dim)

        self.clf_proj = nn.Linear(d_fused, clf_hidden)
        self.clf_fc1 = nn.Linear(d_fused, clf_hidden)
        self.clf_norm1 = _make_norm(clf_hidden)
        self.clf_fc2 = nn.Linear(clf_hidden, clf_hidden)
        self.clf_norm2 = _make_norm(clf_hidden)
        self.clf_out = nn.Linear(clf_hidden, 1)

    # ----------------------------------------------------------
    # Classifier
    # ----------------------------------------------------------

    def _classify(self, fused):
        res = self.clf_proj(fused)
        out = self.clf_fc1(fused)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_norm1(out)
        out = 0.5 * (res + out)
        res = out
        out = self.clf_fc2(out)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_norm2(out)
        out = 0.5 * (res + out)
        return self.clf_out(out)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        struct_graph,
        labels=None, compute_loss=False,
    ):
        """
        Args:
            struct_graph: DGL batched graph with ndata["x"], edata["feat"], edata["etype"]
            labels: [B] binding labels (optional)
            compute_loss: whether to compute BCE loss

        Returns:
            dict with logit, prob, g_emb, fused, and optionally loss/bind_loss
        """
        # 1. Structure encoding
        g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)

        # 2. Project graph embedding
        fused = self.struct_proj(g_emb)

        # 3. Classify
        logit = self._classify(fused)

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "g_emb": g_emb,
            "fused": fused,
        }

        if not compute_loss:
            return out

        # 4. Loss
        clamped_logit = logit.view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            clamped_logit, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )
        out.update({"loss": bind_loss, "bind_loss": bind_loss})
        return out

    # Utility
    @torch.no_grad()
    def predict(self, struct_graph, threshold=0.5):
        self.eval()
        out = self.forward(struct_graph, compute_loss=False)
        probs = out["prob"].squeeze(-1)
        return probs, (probs >= threshold).long()


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import random

    B = 4
    H_struct = 320

    # Build test graphs with edge types
    graphs = []
    for _ in range(B):
        n_mhc = random.randint(10, 20)
        n_pep = random.randint(5, 10)
        n_tra = random.randint(10, 20)
        n_trb = random.randint(10, 20)
        n = n_mhc + n_pep + n_tra + n_trb

        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        g = dgl.graph((src, dst))
        g.ndata["x"] = torch.randn(n, 20)         # one-hot AA
        g.ndata["coords"] = torch.randn(n, 3)      # Cα coords
        g.edata["feat"] = torch.randn(g.num_edges(), 7)  # edge features
        g.edata["etype"] = torch.randint(0, NUM_EDGE_TYPES, (g.num_edges(),))
        graphs.append(g)

    bg = dgl.batch(graphs)
    labels = torch.randint(0, 2, (B,)).float()

    # --- Test 1: Structure-only ---
    model = StructureOnlyModel(
        struct_hidden_dim=H_struct, struct_out_dim=128,
        d_fused=256, clf_hidden=128, pos_weight=2.0,
    )
    out = model(bg, labels=labels, compute_loss=True)
    print(f"=== Structure-Only ===")
    print(f"Loss: {out['loss']:.4f}, Probs: {out['prob'].squeeze(-1).tolist()}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")
    print(f"g_emb shape: {out['g_emb'].shape}")

    # --- Test 2: Backward ---
    out["loss"].backward()
    print(f"\nBackward pass OK")
    grad_proj = model.struct_proj[0].weight.grad
    print(f"Struct proj grad norm: {grad_proj.norm():.6f}")

    # --- Test 3: Inference ---
    model.eval()
    probs, preds = model.predict(bg)
    print(f"\nPredict probs: {probs.tolist()}")
    print(f"Predict preds: {preds.tolist()}")
