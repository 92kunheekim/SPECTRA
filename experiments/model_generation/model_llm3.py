"""
ESMMultimodalBindingModel3 — ESM Node Features + Pseudo-Heterogeneous Graph Transformer
========================================================================================

Extends Model B2 (model_llm2) by injecting ESM token embeddings as additional
node features in the structure encoder, following the STAG-LLM approach.

Key difference from model_llm2:
  In model_llm2, the structure encoder uses only one-hot AA features (20-dim)
  as node features. In model_llm3, each graph node also receives the
  corresponding ESM encoder output embedding, providing rich pre-trained
  contextual representations as node features.

Node feature mapping (ESM → graph nodes):
  - Graph nodes store chain_id (A=0/MHC, C=1/pep, D=2/TRA, E=3/TRB)
    and chain_pos (0-indexed position within the chain).
  - ESM pMHC output: [BOS, MHC_1, ..., MHC_n, PEP_1, ..., PEP_k, EOS]
    → chain A nodes map to positions 1..n, chain C to positions n+1..n+k
  - ESM TCR output:  [BOS, TRA_1, ..., TRA_m, TRB_1, ..., TRB_p, EOS]
    → chain D nodes map to positions 1..m, chain E to positions m+1..m+p

Architecture:
  1. ESM-2 encoder → raw hidden states (for node features) + projected (for seq branch)
  2. Map raw ESM embeddings to graph nodes via chain_id/chain_pos → project → concat with one-hot
  3. Pseudo-heterogeneous graph transformer (structure encoder with augmented node features)
  4. Bidirectional cross-attention: h_tcr ↔ h_mhc
  5. Structure ↔ Sequence cross-attention
  6. Gated fusion → classifier → binding logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# Import shared components from model_llm2
from model_llm2 import (
    LoRALinear,
    LoRAAdapter,
    ESMSequenceEncoder,
    CrossAttention,
    StructureSequenceCrossAttention,
    GatedFusion,
    HeteroStructureEncoder,
    global_mean_pool,
    EDGE_TYPES,
    ETYPE_TO_IDX,
    NUM_EDGE_TYPES,
    CHAIN_TO_GROUP,
    classify_edge,
)


# ============================================================
# ESMMultimodalBindingModel3
# ============================================================

class ESMMultimodalBindingModel3(nn.Module):
    """
    ESM + pseudo-heterogeneous graph transformer with ESM node features.

    Compared to ESMMultimodalBindingModel2:
      - Raw ESM token embeddings are projected and concatenated with one-hot AA
        features before feeding into the structure encoder.
      - The structure encoder's node_proj input dim increases from node_feat_size
        to node_feat_size + esm_node_dim.
      - This provides the graph transformer with rich pre-trained residue
        representations (STAG-LLM style) in addition to structural features.
    """

    def __init__(
        self,
        # ESM
        esm_encoder: nn.Module,
        esm_embedding: nn.Module,
        esm_hidden_size: int = 320,
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        # LoRA
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_n_layers: int = 4,
        lora_pmhc: bool = False,
        # Dimensions
        d_model: int = 256,
        n_cross_heads: int = 8,
        dropout: float = 0.2,
        # ESM node feature projection
        esm_node_dim: int = 64,
        # Structure encoder
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        struct_hidden_dim: int = 320,
        struct_edge_hidden: int = 32,
        struct_n_layers: int = 3,
        struct_n_heads: int = 4,
        struct_out_dim: int = 128,
        # Cross-modal
        struct_seq_cross_heads: int = 4,
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
        self.esm_hidden_size = esm_hidden_size
        self.esm_node_dim = esm_node_dim
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ===== Sequence branch (shared ESM backbone) =====
        self.esm_seq_encoder = ESMSequenceEncoder(
            esm_encoder=esm_encoder, esm_embedding=esm_embedding,
            esm_hidden_size=esm_hidden_size, d_model=d_model,
            freeze_esm=freeze_esm, n_tune_layers=n_tune_layers, dropout=dropout,
        )

        # Separate projection for pMHC (sequence branch)
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

        self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)
        d_seq_feat = 4 * d_model

        # ===== ESM node feature projection =====
        # Projects raw ESM hidden states to a lower dim before concatenation
        self.esm_node_proj = nn.Sequential(
            nn.Linear(esm_hidden_size, esm_node_dim),
            nn.LayerNorm(esm_node_dim),
            nn.GELU(),
        )

        # ===== Structure branch =====
        # Input dim = one-hot AA (node_feat_size) + projected ESM (esm_node_dim)
        augmented_node_feat_size = node_feat_size + esm_node_dim

        self.struct_encoder = HeteroStructureEncoder(
            node_feat_size=augmented_node_feat_size,
            edge_feat_size=edge_feat_size,
            hidden_dim=struct_hidden_dim,
            edge_hidden_dim=struct_edge_hidden,
            n_layers=struct_n_layers,
            n_heads=struct_n_heads,
            d_out=struct_out_dim,
            dropout=dropout,
        )

        # Structure ↔ Sequence cross-attention
        self.struct_seq_cross = StructureSequenceCrossAttention(
            d_seq=d_model, d_struct=struct_hidden_dim,
            d_out=struct_out_dim, n_heads=struct_seq_cross_heads, dropout=dropout,
        )
        d_struct_feat = struct_out_dim + struct_out_dim

        # ===== Gated fusion =====
        self.fusion = GatedFusion(d_seq_feat, d_struct_feat, d_fused)

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
    # ESM encoding helpers
    # ----------------------------------------------------------

    def _esm_forward_raw(self, token_ids):
        """
        Run ESM backbone and return raw hidden states (before projection).

        When LoRA adapter is active, gradients flow through it.
        When frozen without adapter, uses torch.no_grad() for efficiency.
        """
        enc = self.esm_seq_encoder
        if enc.freeze_esm and enc._active_adapter is None:
            with torch.no_grad():
                emb = enc.esm_embedding(token_ids)
                esm_out = enc.esm_encoder(emb)[0]
            esm_out = esm_out.detach()
        else:
            emb = enc.esm_embedding(token_ids)
            esm_out = enc.esm_encoder(emb)[0]
        return esm_out  # [B, L, esm_hidden_size]

    # ----------------------------------------------------------
    # ESM → graph node feature mapping (vectorized)
    # ----------------------------------------------------------

    def _map_esm_to_nodes(self, bg, esm_tcr_raw, esm_mhc_raw):
        """
        Map ESM token embeddings to graph nodes using chain_id/chain_pos.

        Args:
            bg: batched DGL graph with ndata["chain_id"], ndata["chain_pos"]
            esm_tcr_raw: [B, L_tcr, H] raw ESM output for TCR
            esm_mhc_raw: [B, L_mhc, H] raw ESM output for pMHC

        Returns:
            node_esm: [N_total, esm_node_dim] projected ESM features per node
        """
        chain_id = bg.ndata["chain_id"]    # [N]
        chain_pos = bg.ndata["chain_pos"]  # [N]
        N = chain_id.shape[0]
        device = chain_id.device

        # Batch membership
        n_per_g = bg.batch_num_nodes()
        if not torch.is_tensor(n_per_g):
            n_per_g = torch.tensor(n_per_g, device=device)
        batch_ids = torch.repeat_interleave(
            torch.arange(len(n_per_g), device=device), n_per_g
        )

        B = esm_tcr_raw.shape[0]

        # Count chain A (MHC) and chain D (TRA) nodes per sample
        # to compute offset for chain C (peptide) and chain E (TRB)
        is_chain_a = (chain_id == 0).float()
        n_mhc_per_sample = torch.zeros(B, device=device)
        n_mhc_per_sample.scatter_add_(0, batch_ids, is_chain_a)

        is_chain_d = (chain_id == 2).float()
        n_tra_per_sample = torch.zeros(B, device=device)
        n_tra_per_sample.scatter_add_(0, batch_ids, is_chain_d)

        n_mhc = n_mhc_per_sample[batch_ids].long()  # [N]
        n_tra = n_tra_per_sample[batch_ids].long()    # [N]

        # Compute ESM token positions for pMHC and TCR outputs
        # Default to 0 (BOS/CLS token) as safe fallback
        esm_mhc_pos = torch.zeros(N, dtype=torch.long, device=device)
        esm_tcr_pos = torch.zeros(N, dtype=torch.long, device=device)

        # Chain A (MHC):    pMHC ESM position = 1 + chain_pos
        a_mask = (chain_id == 0)
        esm_mhc_pos[a_mask] = 1 + chain_pos[a_mask]

        # Chain C (peptide): pMHC ESM position = 1 + n_mhc + chain_pos
        c_mask = (chain_id == 1)
        esm_mhc_pos[c_mask] = 1 + n_mhc[c_mask] + chain_pos[c_mask]

        # Chain D (TRA):    TCR ESM position = 1 + chain_pos
        d_mask = (chain_id == 2)
        esm_tcr_pos[d_mask] = 1 + chain_pos[d_mask]

        # Chain E (TRB):    TCR ESM position = 1 + n_tra + chain_pos
        e_mask = (chain_id == 3)
        esm_tcr_pos[e_mask] = 1 + n_tra[e_mask] + chain_pos[e_mask]

        # Clamp to valid sequence length
        esm_mhc_pos = esm_mhc_pos.clamp(0, esm_mhc_raw.shape[1] - 1)
        esm_tcr_pos = esm_tcr_pos.clamp(0, esm_tcr_raw.shape[1] - 1)

        # Gather embeddings: index [batch_ids, position] from the ESM outputs
        node_esm_mhc = esm_mhc_raw[batch_ids, esm_mhc_pos]  # [N, H]
        node_esm_tcr = esm_tcr_raw[batch_ids, esm_tcr_pos]   # [N, H]

        # Select: pMHC embedding for chains A/C, TCR embedding for chains D/E
        is_pmhc = (chain_id == 0) | (chain_id == 1)
        node_esm_raw = torch.where(
            is_pmhc.unsqueeze(-1), node_esm_mhc, node_esm_tcr
        )  # [N, esm_hidden_size]

        # Project to lower dimension
        return self.esm_node_proj(node_esm_raw)  # [N, esm_node_dim]

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
        struct_graph=None, struct_available=None,
        labels=None, compute_loss=False,
    ):
        enc = self.esm_seq_encoder

        # 1. TCR: ESM encoding with optional LoRA
        if self.tcr_lora is not None:
            enc.apply_adapter(self.tcr_lora)

        esm_out_tcr_raw = self._esm_forward_raw(tcr_ids)       # [B, L_t, H_esm]
        h_tcr = enc.proj(esm_out_tcr_raw)                       # [B, L_t, d_model]
        pool_tcr = h_tcr[:, 0, :]                                # [B, d_model]

        if self.tcr_lora is not None:
            enc.remove_adapter()

        # 2. pMHC: ESM encoding with optional LoRA + separate projection
        if self.pmhc_lora is not None:
            enc.apply_adapter(self.pmhc_lora)

        esm_out_mhc_raw = self._esm_forward_raw(mhc_ids)       # [B, L_m, H_esm]
        h_mhc = self.pmhc_proj(esm_out_mhc_raw)                 # [B, L_m, d_model]
        pool_mhc = h_mhc[:, 0, :]                                # [B, d_model]

        if self.pmhc_lora is not None:
            enc.remove_adapter()

        # 3. Sequence cross-attention
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)
        f_seq = torch.cat([pool_tcr, pool_mhc, cross_tcr, cross_mhc], dim=-1)

        # 4. Structure encoding with ESM node features
        f_struct = None
        if struct_graph is not None:
            # Inject ESM embeddings as additional node features
            if "chain_id" in struct_graph.ndata:
                node_esm_feat = self._map_esm_to_nodes(
                    struct_graph, esm_out_tcr_raw, esm_out_mhc_raw
                )
                # Concatenate with one-hot AA features: [N, 20] + [N, esm_node_dim]
                struct_graph.ndata["x"] = torch.cat(
                    [struct_graph.ndata["x"], node_esm_feat], dim=-1
                )
            # else: fallback to one-hot only (old cache without chain metadata)

            g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)

            h_seq_concat = torch.cat([h_tcr, h_mhc], dim=1)
            seq_mask_concat = (
                torch.cat([tcr_mask, mhc_mask], dim=1)
                if tcr_mask is not None and mhc_mask is not None else None
            )
            struct_cross = self.struct_seq_cross(
                node_h, batch_ids, h_seq_concat, seq_mask_concat
            )
            f_struct = torch.cat([g_emb, struct_cross], dim=-1)

        # 5. Gated fusion
        fused = self.fusion(f_seq, f_struct, struct_available)

        # 6. Classify
        logit = self._classify(fused)

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "pool_tcr": pool_tcr, "pool_mhc": pool_mhc,
            "f_seq": f_seq, "f_struct": f_struct, "fused": fused,
        }

        if not compute_loss:
            return out

        # 7. Loss
        clamped_logit = logit.view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            clamped_logit, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )
        out.update({"loss": bind_loss, "bind_loss": bind_loss})
        return out

    # Utility
    @torch.no_grad()
    def predict(self, tcr_ids, mhc_ids, tcr_mask=None, mhc_mask=None,
                struct_graph=None, struct_available=None, threshold=0.5):
        self.eval()
        out = self.forward(tcr_ids, mhc_ids, tcr_mask, mhc_mask,
                           struct_graph, struct_available, compute_loss=False)
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
        esm_node_params = sum(p.numel() for p in self.esm_node_proj.parameters())
        print(f"[ESM] {mode}{lora_info} | ESM node proj: {esm_node_params:,} params")


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import random

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

    # Build test graphs with chain_id and chain_pos
    graphs = []
    for _ in range(B):
        # Simulate realistic chain distribution
        n_mhc = random.randint(10, 20)
        n_pep = random.randint(5, 10)
        n_tra = random.randint(10, 20)
        n_trb = random.randint(10, 20)
        n = n_mhc + n_pep + n_tra + n_trb

        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        g = dgl.graph((src, dst))
        g.ndata["x"] = torch.randn(n, 20)
        g.ndata["coords"] = torch.randn(n, 3)
        g.edata["feat"] = torch.randn(g.num_edges(), 7)
        g.edata["etype"] = torch.randint(0, NUM_EDGE_TYPES, (g.num_edges(),))

        # Chain metadata
        chain_ids = (
            [0] * n_mhc + [1] * n_pep + [2] * n_tra + [3] * n_trb
        )
        chain_pos = (
            list(range(n_mhc)) + list(range(n_pep))
            + list(range(n_tra)) + list(range(n_trb))
        )
        g.ndata["chain_id"] = torch.tensor(chain_ids, dtype=torch.long)
        g.ndata["chain_pos"] = torch.tensor(chain_pos, dtype=torch.long)
        graphs.append(g)

    bg = dgl.batch(graphs)
    struct_avail = torch.ones(B, dtype=torch.bool)

    # --- Test: ESM node features ---
    model = ESMMultimodalBindingModel3(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, esm_node_dim=64,
        struct_hidden_dim=H, struct_out_dim=128,
        d_fused=D, clf_hidden=D, pos_weight=2.0,
    )
    out = model(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"=== ESM Node Features (no LoRA) ===")
    print(f"Loss: {out['loss']:.4f}, Probs: {out['prob'].squeeze(-1).tolist()}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    esm_node_params = sum(p.numel() for p in model.esm_node_proj.parameters())
    print(f"ESM node proj params: {esm_node_params:,}")
    print(f"Trainable: {trainable:,} / {total:,}")

    # --- Test: ESM node features + LoRA ---
    model_lora = ESMMultimodalBindingModel3(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, esm_node_dim=64,
        struct_hidden_dim=H, struct_out_dim=128,
        d_fused=D, clf_hidden=D, pos_weight=2.0,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
    )
    out2 = model_lora(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                      struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"\n=== ESM Node Features + TCR LoRA ===")
    print(f"Loss: {out2['loss']:.4f}, Probs: {out2['prob'].squeeze(-1).tolist()}")
    model_lora.set_esm_tuning(freeze=True, n_tune_layers=0)

    # --- Backward test ---
    out["loss"].backward()
    print(f"\nBackward pass OK")
    grad_esm_node = model.esm_node_proj[0].weight.grad
    print(f"ESM node proj grad norm: {grad_esm_node.norm():.6f}")
