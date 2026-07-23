"""
Multimodal TCR–pMHC Binding Prediction Model (v2)
===================================================

Architecture improvements over v1:
  - Sequence encoder: Transformer (Pre-LN, GELU, 4-6 layers) + GRU summary
    replaces shallow 2-layer GRU — captures long-range dependencies in
    180-length MHC and TCR sequences
  - Classifier head: BatchNorm + residual connections (inspired by STAG-LLM)
    replaces plain MLP — stabilises training, improves gradient flow
  - Larger default latent_dim (128 vs 64) to reduce information bottleneck
  - encoder_type flag: "transformer" (default) or "gru" for ablation

Three modalities:
  1. TCR sequence  →  Transformer/GRU VAE encoder  →  z_tcr [B, Z]
  2. MHC sequence  →  Transformer/GRU VAE encoder  →  z_mhc [B, Z]
  3. TCR-pMHC structure graph  →  EGNN  →  g_struct [B, H_g]

Fusion strategy:
  - Sequence branch: z_tcr, z_mhc, cross-attention interaction vectors
  - Structure branch: EGNN graph-level embedding (with optional node→sequence
    cross-attention for structure-sequence alignment)
  - Gated fusion: learned gate decides per-sample how much to weight
    sequence vs structure representations before the classifier

The model gracefully handles missing structure via a binary flag.

Dependencies:
  - torch, torch_geometric (for global_mean_pool)
  - dgl + dgl.nn.EGNNConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from dgl.nn import EGNNConv


# ============================================================
# 1. Sequence components (from binding_model.py)
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer-based sequence encoder."""

    def __init__(self, d_model, max_len=600, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SeqEncoder(nn.Module):
    """
    Hybrid sequence encoder: Embedding → Transformer layers → GRU summary.

    The Transformer captures long-range dependencies (critical for 180-length
    MHC sequences), while the GRU provides a smooth sequential summary.
    LayerNorm on output stabilises downstream cross-attention and VAE heads.
    """

    def __init__(self, vocab_size, pad_id, d_model, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=600, dropout=dropout)
        self.pad_id = pad_id

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # GRU to produce a smooth sequential summary on top of Transformer output
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=1, batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_ids, mask=None):
        # mask: [B, L] True=valid, False=pad
        x = self.pos_enc(self.tok_emb(x_ids))

        # Transformer needs src_key_padding_mask where True=IGNORE
        if mask is not None:
            pad_mask = ~mask  # invert: True → ignore
        else:
            pad_mask = (x_ids == self.pad_id)

        h_seq = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, L, D]
        h_seq = self.out_norm(h_seq)

        # GRU for sequential pooling
        _, h_last = self.gru(h_seq)                # h_last: [1, B, D]
        h_pool = h_last.squeeze(0)                 # [B, D]

        return h_seq, h_pool


class GRUEncoder(nn.Module):
    """GRU encoder → sequence hidden states + masked-mean pooled vector.
    Kept for backward compatibility / lighter configs."""

    def __init__(self, vocab_size, pad_id, d_model, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x_ids, mask=None):
        x = self.tok_emb(x_ids)
        h_seq, _ = self.gru(x)
        if mask is None:
            h_pool = h_seq.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)
            h_pool = (h_seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return h_seq, h_pool


class CrossAttention(nn.Module):
    """Bidirectional multi-head cross-attention between two sequences."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _reshape(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def _attend(self, q_seq, ctx_seq, ctx_mask=None):
        Q = self._reshape(self.W_q(q_seq))
        K = self._reshape(self.W_k(ctx_seq))
        V = self._reshape(self.W_v(ctx_seq))
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if ctx_mask is not None:
            scores = scores.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(q_seq.shape)
        return self.norm(self.out_proj(out) + q_seq)

    def forward(self, h_a, h_b, mask_a=None, mask_b=None):
        a_attended = self._attend(h_a, h_b, mask_b)
        b_attended = self._attend(h_b, h_a, mask_a)

        def _pool(h, mask):
            if mask is None:
                return h.mean(dim=1)
            m = mask.float().unsqueeze(-1)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        return _pool(a_attended, mask_a), _pool(b_attended, mask_b)


# ============================================================
# 2. Structure component: EGNN with projection
# ============================================================

class StructureEGNN(nn.Module):
    """
    EGNN encoder for the TCR-pMHC complex graph.

    Input: DGL batched graph with:
        - ndata["x"]      : [N_total, node_feat_size]  (one-hot AA type, etc.)
        - ndata["coords"]  : [N_total, 3]               (Cα or backbone coords)
        - edata["feat"]    : [E_total, edge_feat_size]   (bond type, distance bins, etc.)
                             Optional — set edge_feat_size=0 if not used.

    Output: graph-level embedding [B, d_out]
    """

    def __init__(
        self,
        node_feat_size: int = 20,
        edge_feat_size: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 5,
        d_out: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # EGNN stack
        self.layers = nn.ModuleList()
        self.layers.append(EGNNConv(
            in_size=node_feat_size, hidden_size=hidden_dim,
            out_size=hidden_dim, edge_feat_size=edge_feat_size,
        ))
        for _ in range(n_layers - 1):
            self.layers.append(EGNNConv(
                in_size=hidden_dim, hidden_size=hidden_dim,
                out_size=hidden_dim, edge_feat_size=edge_feat_size,
            ))

        # Layer norm after message passing (stabilises training)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # Project to output dim
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    @staticmethod
    def _make_batch_ids(bg):
        n_per_g = bg.batch_num_nodes()
        if not torch.is_tensor(n_per_g):
            n_per_g = torch.tensor(n_per_g, device=bg.device)
        return torch.repeat_interleave(
            torch.arange(len(n_per_g), device=bg.device), n_per_g
        )

    def forward(self, bg):
        """
        Returns:
            g_emb       : [B, d_out]   graph-level embedding
            node_feat   : [N_total, hidden_dim]  node-level embeddings (for optional
                          cross-modal attention with sequence)
            batch_ids   : [N_total]    graph membership per node
        """
        h = bg.ndata["x"]
        coords = bg.ndata["coords"]

        edge_feat = bg.edata.get("feat", None)

        for layer in self.layers:
            if edge_feat is not None:
                h, coords = layer(bg, h, coords, edge_feat)
            else:
                h, coords = layer(bg, h, coords)

        h = self.node_norm(h)
        batch_ids = self._make_batch_ids(bg)
        g_emb = global_mean_pool(h, batch_ids)    # [B, hidden_dim]
        g_emb = self.out_proj(g_emb)              # [B, d_out]
        return g_emb, h, batch_ids


# ============================================================
# 3. Structure ↔ Sequence cross-attention (optional but powerful)
# ============================================================

class StructureSequenceCrossAttention(nn.Module):
    """
    Lets the structure graph nodes attend to sequence hidden states and
    vice versa.  This aligns residue-level info between the two modalities.

    Since graph node counts vary per sample, we process each sample in the
    batch independently (loop-based but correct for variable sizes).
    """

    def __init__(self, d_seq, d_struct, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        # Project both modalities to same dim
        self.proj_seq = nn.Linear(d_seq, d_out)
        self.proj_struct = nn.Linear(d_struct, d_out)

        self.d_k = d_out // n_heads
        self.n_heads = n_heads
        assert d_out % n_heads == 0

        # Struct queries → Seq keys/values  (struct attending to seq)
        self.W_q = nn.Linear(d_out, d_out)
        self.W_k = nn.Linear(d_out, d_out)
        self.W_v = nn.Linear(d_out, d_out)
        self.out_proj = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def _single_attend(self, q, kv, kv_mask=None):
        """
        q:  [1, Nq, D]
        kv: [1, Nkv, D]
        kv_mask: [1, Nkv] bool or None
        Returns: [1, Nq, D]
        """
        Q = self.W_q(q).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(q.shape)
        return self.norm(self.out_proj(out) + q)

    def forward(self, node_h, batch_ids, h_seq_concat, seq_mask_concat):
        """
        Args:
            node_h         : [N_total, d_struct]   EGNN node embeddings
            batch_ids      : [N_total]             graph membership
            h_seq_concat   : [B, L_tcr+L_mhc, d_seq]  concatenated TCR+MHC hidden states
            seq_mask_concat: [B, L_tcr+L_mhc] bool

        Returns:
            struct_enriched: [B, d_out]  structure pooled after attending to sequence
        """
        node_h = self.proj_struct(node_h)
        h_seq = self.proj_seq(h_seq_concat)

        B = batch_ids.max().item() + 1
        pooled = []

        for i in range(B):
            node_mask = (batch_ids == i)
            q = node_h[node_mask].unsqueeze(0)         # [1, Ni, D]
            kv = h_seq[i : i + 1]                       # [1, L, D]
            kv_mask = seq_mask_concat[i : i + 1] if seq_mask_concat is not None else None

            out = self._single_attend(q, kv, kv_mask)   # [1, Ni, D]
            pooled.append(out.mean(dim=1))               # [1, D]

        return torch.cat(pooled, dim=0)                  # [B, D]


# ============================================================
# 4. Gated multimodal fusion
# ============================================================

class GatedFusion(nn.Module):
    """
    Learns a per-sample gate λ ∈ [0,1] that blends two representation vectors:
        fused = λ · f_seq  +  (1-λ) · f_struct

    When structure is missing for a sample, the gate is forced to 1.0 (seq only).
    """

    def __init__(self, d_seq, d_struct, d_fused):
        super().__init__()
        self.proj_seq = nn.Linear(d_seq, d_fused)
        self.proj_struct = nn.Linear(d_struct, d_fused)

        # Gate network: takes concatenation of both → scalar per sample
        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.ReLU(),
            nn.Linear(d_fused, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_seq, f_struct=None, struct_available=None):
        """
        Args:
            f_seq    : [B, d_seq]
            f_struct : [B, d_struct] or None
            struct_available: [B] bool tensor — True if structure exists for that sample

        Returns:
            fused: [B, d_fused]
        """
        s = self.proj_seq(f_seq)             # [B, d_fused]

        if f_struct is None:
            return s  # pure sequence mode

        g = self.proj_struct(f_struct)       # [B, d_fused]
        lam = self.gate(torch.cat([s, g], dim=-1))   # [B, 1]

        # Force gate=1 (all seq) for samples without structure
        if struct_available is not None:
            mask = (~struct_available).float().unsqueeze(-1)  # 1 where missing
            lam = lam * (1 - mask) + mask  # push to 1.0 where structure missing

        fused = lam * s + (1 - lam) * g
        return fused


# ============================================================
# 5. Full multimodal model
# ============================================================

class MultimodalBindingModel(nn.Module):
    """
    Multimodal TCR-pMHC binding predictor.

    Modalities:
        (a) TCR sequence  — GRU VAE encoder
        (b) MHC sequence  — GRU VAE encoder
        (c) TCR-pMHC structure graph — EGNN

    Fusion:
        Sequence features:  [z_tcr, z_mhc, cross_tcr, cross_mhc]  → f_seq
        Structure features: [g_emb, struct_seq_cross]              → f_struct
        Gated fusion → fused → MLP classifier → binding logit

    Training objective:
        L = λ_bind * BCE  +  λ_recon * (recon_tcr + recon_mhc)
            + λ_kl * β * (KL_tcr + KL_mhc)  +  λ_struct * struct_contrastive (optional)
    """

    def __init__(
        self,
        # Vocabulary / tokenisation
        vocab_size: int,
        pad_id: int,
        # Sequence encoder config
        d_model: int = 256,
        latent_dim: int = 128,
        n_enc_layers: int = 4,
        n_enc_heads: int = 8,
        n_cross_heads: int = 4,
        dropout: float = 0.1,
        kl_anneal_steps: int = 5000,
        encoder_type: str = "transformer",   # "transformer" or "gru"
        # Structure encoder config
        node_feat_size: int = 20,
        edge_feat_size: int = 0,
        egnn_hidden: int = 128,
        egnn_layers: int = 5,
        egnn_out_dim: int = 128,
        # Cross-modal attention
        struct_seq_cross_heads: int = 4,
        # Fusion / classifier
        d_fused: int = 256,
        clf_hidden: int = 128,
        # Loss
        lambda_bind: float = 1.0,
        lambda_recon: float = 0.3,
        lambda_kl: float = 0.2,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.kl_anneal_steps = int(kl_anneal_steps)
        self._global_step = 0
        self.encoder_type = encoder_type

        self.lambda_bind = lambda_bind
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.register_buffer("pos_weight_buf", torch.tensor([pos_weight]))

        # ===== Sequence branch =====
        # Choose encoder type
        if encoder_type == "transformer":
            EncoderClass = lambda: SeqEncoder(vocab_size, pad_id, d_model, n_enc_layers, n_enc_heads, dropout)
        else:
            EncoderClass = lambda: GRUEncoder(vocab_size, pad_id, d_model, n_enc_layers, dropout)

        # TCR encoder / decoder
        self.tcr_enc = EncoderClass()
        self.tcr_to_mu = nn.Linear(d_model, latent_dim)
        self.tcr_to_logvar = nn.Linear(d_model, latent_dim)
        self.tcr_dec = nn.GRU(d_model, d_model, min(n_enc_layers, 2), batch_first=True,
                              dropout=dropout if n_enc_layers > 1 else 0.0)
        self.tcr_to_logits = nn.Linear(d_model, vocab_size)
        self.tcr_cond_proj = nn.Linear(latent_dim, d_model)

        # MHC encoder / decoder
        self.mhc_enc = EncoderClass()
        self.mhc_to_mu = nn.Linear(d_model, latent_dim)
        self.mhc_to_logvar = nn.Linear(d_model, latent_dim)
        self.mhc_dec = nn.GRU(d_model, d_model, min(n_enc_layers, 2), batch_first=True,
                              dropout=dropout if n_enc_layers > 1 else 0.0)
        self.mhc_to_logits = nn.Linear(d_model, vocab_size)
        self.mhc_cond_proj = nn.Linear(latent_dim, d_model)

        # Sequence cross-attention (TCR ↔ MHC)
        self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)

        # Sequence feature dim: z_tcr + z_mhc + cross_tcr + cross_mhc
        d_seq_feat = 2 * latent_dim + 2 * d_model

        # ===== Structure branch =====
        self.struct_encoder = StructureEGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            hidden_dim=egnn_hidden,
            n_layers=egnn_layers,
            d_out=egnn_out_dim,
            dropout=dropout,
        )

        # Structure ↔ Sequence cross-attention
        self.struct_seq_cross = StructureSequenceCrossAttention(
            d_seq=d_model,
            d_struct=egnn_hidden,   # node-level hidden before projection
            d_out=egnn_out_dim,
            n_heads=struct_seq_cross_heads,
            dropout=dropout,
        )

        # Structure feature dim: graph embedding + struct→seq cross-attended
        d_struct_feat = egnn_out_dim + egnn_out_dim

        # ===== Gated fusion =====
        self.fusion = GatedFusion(d_seq_feat, d_struct_feat, d_fused)

        # ===== Classifier (BatchNorm + residual, STAG-LLM style) =====
        self.dropout_p = dropout
        self.clf_proj = nn.Linear(d_fused, clf_hidden)
        self.clf_fc1 = nn.Linear(d_fused, clf_hidden)
        self.clf_bn1 = nn.BatchNorm1d(clf_hidden)
        self.clf_fc2 = nn.Linear(clf_hidden, clf_hidden)
        self.clf_bn2 = nn.BatchNorm1d(clf_hidden)
        self.clf_out = nn.Linear(clf_hidden, 1)

    # ---- Utilities ----
    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_divergence(mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1)

    def kl_beta(self):
        if self.kl_anneal_steps <= 0:
            return 1.0
        return min(1.0, self._global_step / self.kl_anneal_steps)

    def _encode_seq(self, encoder, to_mu, to_logvar, x_ids, mask):
        h_seq, h_pool = encoder(x_ids, mask)
        mu = to_mu(h_pool)
        logvar = to_logvar(h_pool)
        z = self.reparameterize(mu, logvar)
        return h_seq, mu, logvar, z

    def _decode_seq(self, decoder, to_logits, cond_proj, tok_emb, x_ids, z):
        ctx = cond_proj(z)                   # [B, D]
        x = tok_emb(x_ids) + ctx.unsqueeze(1)
        h, _ = decoder(x)
        return to_logits(h)

    # ---- Forward ----
    def forward(
        self,
        tcr_ids: torch.Tensor,               # [B, L_tcr]
        mhc_ids: torch.Tensor,               # [B, L_mhc]
        tcr_mask: torch.Tensor = None,        # [B, L_tcr] bool
        mhc_mask: torch.Tensor = None,        # [B, L_mhc] bool
        struct_graph=None,                     # DGL batched graph or None
        struct_available: torch.Tensor = None, # [B] bool — which samples have structure
        labels: torch.Tensor = None,           # [B] float {0,1}
        compute_loss: bool = False,
    ):
        self._global_step += 1

        # ===== 1. Sequence encoding =====
        h_tcr, mu_tcr, lv_tcr, z_tcr = self._encode_seq(
            self.tcr_enc, self.tcr_to_mu, self.tcr_to_logvar, tcr_ids, tcr_mask
        )
        h_mhc, mu_mhc, lv_mhc, z_mhc = self._encode_seq(
            self.mhc_enc, self.mhc_to_mu, self.mhc_to_logvar, mhc_ids, mhc_mask
        )

        # Sequence cross-attention (TCR ↔ MHC)
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)

        f_seq = torch.cat([z_tcr, z_mhc, cross_tcr, cross_mhc], dim=-1)  # [B, d_seq_feat]

        # ===== 2. Structure encoding (if available) =====
        f_struct = None
        if struct_graph is not None:
            g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)

            # Structure→Sequence cross-attention
            #   Concatenate TCR + MHC hidden states along sequence dim
            h_seq_concat = torch.cat([h_tcr, h_mhc], dim=1)  # [B, L_tcr+L_mhc, D]
            if tcr_mask is not None and mhc_mask is not None:
                seq_mask_concat = torch.cat([tcr_mask, mhc_mask], dim=1)
            else:
                seq_mask_concat = None

            struct_cross = self.struct_seq_cross(
                node_h, batch_ids, h_seq_concat, seq_mask_concat
            )  # [B, egnn_out_dim]

            f_struct = torch.cat([g_emb, struct_cross], dim=-1)  # [B, d_struct_feat]

        # ===== 3. Gated fusion =====
        fused = self.fusion(f_seq, f_struct, struct_available)  # [B, d_fused]

        # ===== 4. Classification (BatchNorm + residual) =====
        res = self.clf_proj(fused)                              # [B, clf_hidden]
        out = self.clf_fc1(fused)                               # [B, clf_hidden]
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_bn1(out)
        out = 0.5 * (res + out)                                 # residual
        res = out
        out = self.clf_fc2(out)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_bn2(out)
        out = 0.5 * (res + out)                                 # residual
        logit = self.clf_out(out)                               # [B, 1]

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "z_tcr": z_tcr, "z_mhc": z_mhc,
            "mu_tcr": mu_tcr, "logvar_tcr": lv_tcr,
            "mu_mhc": mu_mhc, "logvar_mhc": lv_mhc,
            "f_seq": f_seq,
            "f_struct": f_struct,
            "fused": fused,
        }

        if not compute_loss:
            return out

        # ===== 5. Losses =====

        # --- Binding ---
        bind_loss = F.binary_cross_entropy_with_logits(
            logit.view(-1), labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )

        # --- VAE reconstruction (teacher-forcing) ---
        tcr_logits = self._decode_seq(
            self.tcr_dec, self.tcr_to_logits, self.tcr_cond_proj,
            self.tcr_enc.tok_emb, tcr_ids, z_tcr,
        )
        mhc_logits = self._decode_seq(
            self.mhc_dec, self.mhc_to_logits, self.mhc_cond_proj,
            self.mhc_enc.tok_emb, mhc_ids, z_mhc,
        )

        def _recon(logits, target):
            B, L, V = logits.shape
            return F.cross_entropy(
                logits.view(B * L, V), target.view(B * L),
                ignore_index=self.pad_id, reduction="mean",
            )

        recon_tcr = _recon(tcr_logits, tcr_ids)
        recon_mhc = _recon(mhc_logits, mhc_ids)

        # --- KL ---
        kl_tcr = self.kl_divergence(mu_tcr, lv_tcr).mean()
        kl_mhc = self.kl_divergence(mu_mhc, lv_mhc).mean()
        beta = self.kl_beta()

        total = (
            self.lambda_bind * bind_loss
            + self.lambda_recon * (recon_tcr + recon_mhc)
            + self.lambda_kl * beta * (kl_tcr + kl_mhc)
        )

        out.update({
            "loss": total,
            "bind_loss": bind_loss,
            "recon_tcr": recon_tcr, "recon_mhc": recon_mhc,
            "kl_tcr": kl_tcr, "kl_mhc": kl_mhc,
            "beta": beta,
        })
        return out

    # ---- Inference ----
    @torch.no_grad()
    def predict(self, tcr_ids, mhc_ids, tcr_mask=None, mhc_mask=None,
                struct_graph=None, struct_available=None, threshold=0.5):
        self.eval()
        out = self.forward(
            tcr_ids, mhc_ids, tcr_mask, mhc_mask,
            struct_graph, struct_available, compute_loss=False,
        )
        probs = out["prob"].squeeze(-1)
        return probs, (probs >= threshold).long()


# ============================================================
# 6. Example usage / smoke test
# ============================================================

if __name__ == "__main__":
    import dgl

    B, L_tcr, L_mhc = 4, 30, 34
    V, PAD = 25, 0

    model = MultimodalBindingModel(
        vocab_size=V, pad_id=PAD,
        d_model=128, latent_dim=64,
        n_enc_layers=4, n_enc_heads=4, n_cross_heads=4,
        encoder_type="transformer",
        node_feat_size=20, edge_feat_size=0,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4,
        d_fused=128, clf_hidden=64,
        kl_anneal_steps=2000, pos_weight=2.0,
    )

    # --- Dummy sequence data ---
    tcr = torch.randint(1, V, (B, L_tcr))
    mhc = torch.randint(1, V, (B, L_mhc))
    tcr_mask = torch.ones(B, L_tcr, dtype=torch.bool)
    mhc_mask = torch.ones(B, L_mhc, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,)).float()

    # --- Dummy structure graphs (variable node counts) ---
    graphs = []
    for _ in range(B):
        n_nodes = torch.randint(40, 80, (1,)).item()
        src = torch.randint(0, n_nodes, (n_nodes * 3,))
        dst = torch.randint(0, n_nodes, (n_nodes * 3,))
        g = dgl.graph((src, dst))
        g.ndata["x"] = torch.randn(n_nodes, 20)
        g.ndata["coords"] = torch.randn(n_nodes, 3)
        graphs.append(g)
    bg = dgl.batch(graphs)

    struct_avail = torch.ones(B, dtype=torch.bool)

    # --- Forward with all modalities ---
    out = model(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )

    print("=== With structure ===")
    print(f"Total loss:   {out['loss']:.4f}")
    print(f"Binding loss: {out['bind_loss']:.4f}")
    print(f"Recon TCR:    {out['recon_tcr']:.4f}")
    print(f"Recon MHC:    {out['recon_mhc']:.4f}")
    print(f"Probs:        {out['prob'].squeeze(-1).tolist()}")

    # --- Forward without structure (sequence only) ---
    out2 = model(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=None, struct_available=None,
        labels=labels, compute_loss=True,
    )
    print("\n=== Sequence only ===")
    print(f"Total loss:   {out2['loss']:.4f}")
    print(f"Binding loss: {out2['bind_loss']:.4f}")
    print(f"Probs:        {out2['prob'].squeeze(-1).tolist()}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")