"""
ESMSequenceOnlyModel — Sequence-Only Ablation (No Structure Branch)
====================================================================

Ablation model that keeps ONLY the ESM sequence branch from model_llm2,
removing the entire graph/structure encoder. This isolates the contribution
of the ESM-2 sequence representations + cross-attention for TCR-pMHC
binding prediction.

Architecture:
  1. ESM-2 encoder (+ optional LoRA) → TCR projection  → h_tcr
  2. ESM-2 encoder (+ optional LoRA) → pMHC projection → h_mhc
  3. Bidirectional cross-attention: h_tcr ↔ h_mhc
  4. f_seq = [pool_tcr, pool_mhc, cross_tcr, cross_mhc]  (4 × d_model)
  5. Linear projection → d_fused
  6. Residual classifier → binding logit

Removed (vs model_llm2):
  - HeteroStructureEncoder (pseudo-heterogeneous graph transformer)
  - StructureSequenceCrossAttention
  - GatedFusion (replaced by simple Linear projection)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared components from model_llm2
from model_llm2 import (
    LoRALinear,
    LoRAAdapter,
    ESMSequenceEncoder,
    CrossAttention,
)


# ============================================================
# ESMSequenceOnlyModel
# ============================================================

class ESMSequenceOnlyModel(nn.Module):
    """
    Sequence-only ablation model for TCR-pMHC binding prediction.

    Uses ESM-2 representations + bidirectional cross-attention, without
    any structural information. This serves as a baseline to measure
    the contribution of the graph/structure encoder.
    """

    def __init__(
        self,
        # ESM
        esm_encoder: nn.Module,
        esm_embedding: nn.Module,
        esm_hidden_size: int = 320,
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        # LoRA configuration
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_n_layers: int = 4,
        lora_pmhc: bool = False,
        # Dimensions
        d_model: int = 256,
        n_cross_heads: int = 8,
        dropout: float = 0.2,
        # Fusion / classifier
        d_fused: int = 256,
        clf_hidden: int = 256,
        norm_type: str = "layernorm",
        # Loss
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout
        self.use_lora = use_lora
        self.lora_pmhc = lora_pmhc
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ===== Sequence branch (shared ESM backbone) =====
        self.esm_seq_encoder = ESMSequenceEncoder(
            esm_encoder=esm_encoder, esm_embedding=esm_embedding,
            esm_hidden_size=esm_hidden_size, d_model=d_model,
            freeze_esm=freeze_esm, n_tune_layers=n_tune_layers, dropout=dropout,
        )

        # Separate projection head for pMHC
        self.pmhc_proj = nn.Sequential(
            nn.Linear(esm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ===== LoRA adapters (optional) =====
        if use_lora:
            self.tcr_lora = LoRAAdapter(
                esm_encoder, n_layers=lora_n_layers,
                rank=lora_rank, alpha=lora_alpha,
            )
            self.pmhc_lora = LoRAAdapter(
                esm_encoder, n_layers=lora_n_layers,
                rank=lora_rank, alpha=lora_alpha,
            ) if lora_pmhc else None
        else:
            self.tcr_lora = None
            self.pmhc_lora = None

        # ===== Cross-attention =====
        self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)
        d_seq_feat = 4 * d_model  # [pool_tcr, pool_mhc, cross_tcr, cross_mhc]

        # ===== Projection (replaces GatedFusion) =====
        self.seq_proj = nn.Sequential(
            nn.Linear(d_seq_feat, d_fused),
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
        tcr_ids, mhc_ids,
        tcr_mask=None, mhc_mask=None,
        labels=None, compute_loss=False,
    ):
        # 1. TCR encoding via ESM (with optional LoRA)
        if self.tcr_lora is not None:
            self.esm_seq_encoder.apply_adapter(self.tcr_lora)
        h_tcr, pool_tcr = self.esm_seq_encoder(tcr_ids, tcr_mask)

        if self.tcr_lora is not None:
            self.esm_seq_encoder.remove_adapter()

        # 2. pMHC encoding via ESM (with optional LoRA) + separate projection
        if self.pmhc_lora is not None:
            self.esm_seq_encoder.apply_adapter(self.pmhc_lora)

        enc = self.esm_seq_encoder
        if enc.freeze_esm and enc._active_adapter is None:
            with torch.no_grad():
                emb = enc.esm_embedding(mhc_ids)
                esm_out_mhc = enc.esm_encoder(emb)[0]
            esm_out_mhc = esm_out_mhc.detach()
        else:
            emb = enc.esm_embedding(mhc_ids)
            esm_out_mhc = enc.esm_encoder(emb)[0]
        h_mhc = self.pmhc_proj(esm_out_mhc)
        pool_mhc = h_mhc[:, 0, :]

        if self.pmhc_lora is not None:
            self.esm_seq_encoder.remove_adapter()

        # 3. Bidirectional cross-attention
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)
        f_seq = torch.cat([pool_tcr, pool_mhc, cross_tcr, cross_mhc], dim=-1)

        # 4. Project (no structure fusion needed)
        fused = self.seq_proj(f_seq)

        # 5. Classify
        logit = self._classify(fused)

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "pool_tcr": pool_tcr, "pool_mhc": pool_mhc,
            "f_seq": f_seq, "fused": fused,
        }

        if not compute_loss:
            return out

        # 6. Loss
        clamped_logit = logit.view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            clamped_logit, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )
        out.update({"loss": bind_loss, "bind_loss": bind_loss})
        return out

    # Utility
    @torch.no_grad()
    def predict(self, tcr_ids, mhc_ids, tcr_mask=None, mhc_mask=None, threshold=0.5):
        self.eval()
        out = self.forward(tcr_ids, mhc_ids, tcr_mask, mhc_mask, compute_loss=False)
        probs = out["prob"].squeeze(-1)
        return probs, (probs >= threshold).long()

    def set_esm_tuning(self, freeze=True, n_tune_layers=0):
        self.esm_seq_encoder._configure_freezing(freeze, n_tune_layers)
        mode = "frozen" if freeze else f"tuning last {n_tune_layers}" if n_tune_layers else "full"
        lora_info = ""
        if self.tcr_lora is not None:
            tcr_params = sum(p.numel() for p in self.tcr_lora.parameters())
            lora_info += f" | TCR LoRA: {tcr_params:,} params"
        if self.pmhc_lora is not None:
            pmhc_params = sum(p.numel() for p in self.pmhc_lora.parameters())
            lora_info += f" | pMHC LoRA: {pmhc_params:,} params"
        print(f"[ESM seq-only] {mode}{lora_info}")


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    # --- Mock ESM components ---
    class MockSelfAttention(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.query = nn.Linear(h, h)
            self.key = nn.Linear(h, h)
            self.value = nn.Linear(h, h)

    class MockAttention(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.self = MockSelfAttention(h)

    class MockTransformerLayer(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.attention = MockAttention(h)
            self.ff = nn.Sequential(nn.Linear(h, h), nn.ReLU())
            self.norm = nn.LayerNorm(h)
        def forward(self, x):
            return self.norm(x + self.ff(x))

    class MockEnc(nn.Module):
        def __init__(self, h=320, n=6):
            super().__init__()
            self.layer = nn.ModuleList([MockTransformerLayer(h) for _ in range(n)])
        def forward(self, x):
            for layer in self.layer:
                x = layer(x)
            return (x,)

    class MockEmb(nn.Module):
        def __init__(self, v=33, h=320):
            super().__init__()
            self.emb = nn.Embedding(v, h)
        def forward(self, x): return self.emb(x)

    B, L_T, L_M = 4, 30, 34
    H, D = 320, 256

    tcr = torch.randint(1, 33, (B, L_T))
    mhc = torch.randint(1, 33, (B, L_M))
    tcr_mask = torch.ones(B, L_T, dtype=torch.bool)
    mhc_mask = torch.ones(B, L_M, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,)).float()

    # --- Test 1: Sequence-only (no LoRA) ---
    model = ESMSequenceOnlyModel(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D,
        d_fused=D, clf_hidden=D, pos_weight=2.0,
    )
    out = model(tcr, mhc, tcr_mask, mhc_mask, labels=labels, compute_loss=True)
    print(f"=== Sequence-Only (no LoRA) ===")
    print(f"Loss: {out['loss']:.4f}, Probs: {out['prob'].squeeze(-1).tolist()}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")

    # --- Test 2: Sequence-only + LoRA ---
    model_lora = ESMSequenceOnlyModel(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D,
        d_fused=D, clf_hidden=D, pos_weight=2.0,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
    )
    out2 = model_lora(tcr, mhc, tcr_mask, mhc_mask, labels=labels, compute_loss=True)
    print(f"\n=== Sequence-Only + TCR LoRA ===")
    print(f"Loss: {out2['loss']:.4f}, Probs: {out2['prob'].squeeze(-1).tolist()}")
    model_lora.set_esm_tuning(freeze=True, n_tune_layers=0)

    # --- Test 3: Backward ---
    out["loss"].backward()
    print(f"\nBackward pass OK")
    grad_proj = model.seq_proj[0].weight.grad
    print(f"Seq proj grad norm: {grad_proj.norm():.6f}")

    # --- Test 4: No struct_graph in forward (should work fine) ---
    out3 = model(tcr, mhc, tcr_mask, mhc_mask, compute_loss=False)
    assert "f_seq" in out3 and out3["fused"] is not None
    print(f"\nInference (no labels) OK, prob shape: {out3['prob'].shape}")
