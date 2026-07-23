"""
ESM-Based Multimodal TCR–pMHC Binding Prediction Model (Model B)
=================================================================

Replaces the VAE sequence encoders in Model A with a pretrained ESM-2
protein language model. No reconstruction loss, no KL divergence —
just binding BCE + rich pretrained representations.

Architecture:
  1. TCR sequence  →  ESM-2 encoder (+ optional LoRA)  →  TCR projection  →  h_tcr
  2. pMHC sequence →  ESM-2 encoder (+ optional LoRA)  →  pMHC projection →  h_mhc
  3. Bidirectional cross-attention:  h_tcr ↔ h_mhc
  4. Structure graph →  EGNN  →  g_struct
  5. Structure ↔ Sequence cross-attention
  6. Gated fusion  →  BatchNorm + residual classifier  →  binding logit

ESM-2 can be:
  - Frozen (fast, memory-efficient, use precomputed embeddings)
  - Fine-tuned end-to-end (better performance, higher memory cost)
  - Partially fine-tuned (freeze early layers, tune last N layers)
  - LoRA-adapted with chain-specific adapters (shared backbone + separate
    low-rank adapters for TCR vs pMHC — memory-efficient specialization)

Dependencies:
  - torch, torch_geometric (for global_mean_pool)
  - dgl + dgl.nn.EGNNConv
  - transformers (for ESM-2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from dgl.nn import EGNNConv


# ============================================================
# 0. LoRA — Low-Rank Adaptation Modules
# ============================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.

    Adds a trainable low-rank bypass:  y = W_frozen·x + (B·A)·x · (α/r)
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, and the original W is frozen.

    This does NOT replace the original Linear — it sits alongside it and
    adds its output. The original weight remains frozen.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # A is initialized with Kaiming uniform, B with zeros
        # so the adapter output starts at zero (no perturbation at init)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward
        base_out = self.original(x)
        # LoRA bypass: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAAdapter(nn.Module):
    """
    A set of LoRA adapters for the last N layers of an ESM encoder.

    Each adapted layer gets LoRA on both the self-attention Q and V projections
    (the standard LoRA recipe from Hu et al. 2021).

    Usage:
        adapter = LoRAAdapter(esm_model, n_layers=4, rank=8)
        # adapter.apply_to(esm_model)   — patches in-place
        # adapter.remove_from(esm_model) — restores originals
    """

    def __init__(self, esm_model: nn.Module, n_layers: int = 4,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.alpha = alpha

        encoder_layers = list(esm_model.encoder.layer)
        target_layers = encoder_layers[-n_layers:] if n_layers > 0 else []

        # Store LoRA modules in a ModuleDict keyed by layer index
        self.lora_modules = nn.ModuleDict()
        self._original_modules = {}  # non-parameter storage for originals

        for layer in target_layers:
            layer_idx = encoder_layers.index(layer)
            attn = layer.attention.self

            # Create LoRA wrappers for Q and V projections
            lora_q = LoRALinear(attn.query, rank=rank, alpha=alpha)
            lora_v = LoRALinear(attn.value, rank=rank, alpha=alpha)

            self.lora_modules[f"layer_{layer_idx}_q"] = lora_q
            self.lora_modules[f"layer_{layer_idx}_v"] = lora_v

    def apply_to(self, esm_model: nn.Module):
        """Patch LoRA adapters into the ESM model (in-place)."""
        encoder_layers = list(esm_model.encoder.layer)
        for key, lora_mod in self.lora_modules.items():
            parts = key.split("_")  # "layer_3_q" -> idx=3, target=q
            layer_idx = int(parts[1])
            target = parts[2]
            attn = encoder_layers[layer_idx].attention.self
            # Save original for removal
            self._original_modules[key] = getattr(attn, "query" if target == "q" else "value")
            setattr(attn, "query" if target == "q" else "value", lora_mod)

    def remove_from(self, esm_model: nn.Module):
        """Remove LoRA adapters and restore original modules."""
        encoder_layers = list(esm_model.encoder.layer)
        for key, orig_mod in self._original_modules.items():
            parts = key.split("_")
            layer_idx = int(parts[1])
            target = parts[2]
            attn = encoder_layers[layer_idx].attention.self
            setattr(attn, "query" if target == "q" else "value", orig_mod)
        self._original_modules.clear()


# ============================================================
# 1. ESM Sequence Encoder Wrapper
# ============================================================

class ESMSequenceEncoder(nn.Module):
    """
    Wraps a full HuggingFace EsmModel to produce projected hidden states
    and a pooled sequence vector.

    Supports three fine-tuning modes:
      1. Frozen backbone (freeze_esm=True, no LoRA)
      2. Partial fine-tuning (freeze_esm=True, n_tune_layers>0)
      3. LoRA adaptation (use_lora=True) — see LoRAAdapter

    When using LoRA with chain-specific adapters, the parent model holds
    separate LoRAAdapter modules for TCR and pMHC. Before each forward pass,
    it patches in the correct adapter via apply_adapter() / remove_adapter().

    IMPORTANT: We use the full EsmModel (not just encoder+embedding) because
    EsmModel.forward() properly handles:
      - Position ID creation from input_ids (padding-aware)
      - Token dropout (ESM-specific masking strategy)
      - Attention mask conversion: [B, L] bool -> 4D float mask with -inf
        for padding positions (REQUIRED by transformer layers)

    Without proper attention masking, padding tokens attend to all positions,
    producing garbage hidden states that overflow in bf16.
    """

    def __init__(
        self,
        esm_model: nn.Module,
        esm_hidden_size: int = 480,
        d_model: int = 256,
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm_model = esm_model  # Full EsmModel (handles masks properly)
        self.esm_hidden_size = esm_hidden_size
        self.freeze_esm = freeze_esm
        self._active_adapter = None  # tracks which adapter is patched in

        # Projection: ESM hidden size -> working dimension
        self.proj = nn.Sequential(
            nn.Linear(esm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._configure_freezing(freeze_esm, n_tune_layers)

    def _configure_freezing(self, freeze_esm, n_tune_layers):
        """Freeze ESM parameters based on configuration."""
        if freeze_esm:
            for p in self.esm_model.parameters():
                p.requires_grad = False

            if n_tune_layers > 0:
                layers = list(self.esm_model.encoder.layer)
                for layer in layers[-n_tune_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def apply_adapter(self, adapter: "LoRAAdapter"):
        """Patch a LoRA adapter into the ESM model."""
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_model)
        adapter.apply_to(self.esm_model)
        self._active_adapter = adapter

    def remove_adapter(self):
        """Remove the currently active LoRA adapter."""
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_model)
            self._active_adapter = None

    def forward(self, token_ids, mask=None):
        """
        Args:
            token_ids: [B, L] ESM tokenized input (includes BOS/EOS)
            mask: [B, L] bool, True=valid token
        Returns:
            h_seq:  [B, L, d_model] projected hidden states
            h_pool: [B, d_model]    CLS token (index 0)
        """
        # Convert bool mask to int for HuggingFace (expects 1/0)
        attention_mask = mask.long() if mask is not None else None

        # When LoRA is active, adapters are patched into the model,
        # so we always run with gradients enabled for LoRA params.
        # When frozen without LoRA, use no_grad for efficiency.
        if self.freeze_esm and self._active_adapter is None:
            with torch.no_grad():
                esm_out = self.esm_model(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
            esm_out = esm_out.detach()
        else:
            esm_out = self.esm_model(
                input_ids=token_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        h_seq = self.proj(esm_out)       # [B, L, d_model]
        h_pool = h_seq[:, 0, :]          # [B, d_model]
        return h_seq, h_pool


# ============================================================
# 2. Cross-attention (reused from Model A)
# ============================================================

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
        # FIX: Clamp scores to prevent overflow in bf16/fp16
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.clamp(-50.0, 50.0)
        if ctx_mask is not None:
            scores = scores.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        # FIX: Sanitize NaN from softmax (when all scores are -inf)
        attn = torch.nan_to_num(attn, nan=0.0)
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
# 3. Structure EGNN (identical to Model A)
# ============================================================

class StructureEGNN(nn.Module):
    """EGNN encoder for TCR-pMHC complex graph."""

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

        self.node_norm = nn.LayerNorm(hidden_dim)
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
        # FIX: Force float32 for EGNN — bf16 causes NaN in distance
        # computations and coordinate updates over multiple layers.
        device = bg.device
        h = bg.ndata["x"].float()
        coords = bg.ndata["coords"].float()
        edge_feat = bg.edata.get("feat", None)
        if edge_feat is not None:
            edge_feat = edge_feat.float()

        with torch.autocast(device_type=device.type, enabled=False):
            for layer in self.layers:
                if edge_feat is not None:
                    h, coords = layer(bg, h, coords, edge_feat)
                else:
                    h, coords = layer(bg, h, coords)
                # Sanitize after each layer to stop NaN propagation
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                coords = torch.nan_to_num(coords, nan=0.0, posinf=1e4, neginf=-1e4)

        h = self.node_norm(h)
        batch_ids = self._make_batch_ids(bg)
        g_emb = global_mean_pool(h, batch_ids)
        g_emb = self.out_proj(g_emb)
        return g_emb, h, batch_ids


# ============================================================
# 4. Structure ↔ Sequence cross-attention (identical to Model A)
# ============================================================

class StructureSequenceCrossAttention(nn.Module):
    """Structure graph nodes attend to sequence hidden states."""

    def __init__(self, d_seq, d_struct, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_out = d_out
        self.proj_seq = nn.Linear(d_seq, d_out)
        self.proj_struct = nn.Linear(d_struct, d_out)

        self.d_k = d_out // n_heads
        self.n_heads = n_heads
        assert d_out % n_heads == 0

        self.W_q = nn.Linear(d_out, d_out)
        self.W_k = nn.Linear(d_out, d_out)
        self.W_v = nn.Linear(d_out, d_out)
        self.out_proj = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def _single_attend(self, q, kv, kv_mask=None):
        Q = self.W_q(q).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)

        # FIX: Clamp scores to prevent overflow in bf16/fp16
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.clamp(-50.0, 50.0)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        # FIX: Sanitize NaN from softmax (when all scores are -inf)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ V).transpose(1, 2).contiguous().view(q.shape)
        return self.norm(self.out_proj(out) + q)

    def forward(self, node_h, batch_ids, h_seq_concat, seq_mask_concat):
        node_h = self.proj_struct(node_h)
        h_seq = self.proj_seq(h_seq_concat)

        B = batch_ids.max().item() + 1
        pooled = []

        for i in range(B):
            node_mask = (batch_ids == i)
            n_nodes = node_mask.sum().item()
            # FIX: Handle empty graph (no nodes for this sample)
            if n_nodes == 0:
                pooled.append(torch.zeros(1, self.d_out,
                                          device=node_h.device, dtype=node_h.dtype))
                continue
            q = node_h[node_mask].unsqueeze(0)
            kv = h_seq[i : i + 1]
            kv_mask = seq_mask_concat[i : i + 1] if seq_mask_concat is not None else None

            out = self._single_attend(q, kv, kv_mask)
            pooled.append(out.mean(dim=1))

        return torch.cat(pooled, dim=0)


# ============================================================
# 5. Gated Fusion (identical to Model A)
# ============================================================

class GatedFusion(nn.Module):
    """Learns per-sample gate λ ∈ [0,1] blending sequence and structure."""

    def __init__(self, d_seq, d_struct, d_fused):
        super().__init__()
        self.proj_seq = nn.Linear(d_seq, d_fused)
        self.proj_struct = nn.Linear(d_struct, d_fused)

        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.ReLU(),
            nn.Linear(d_fused, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_seq, f_struct=None, struct_available=None):
        s = self.proj_seq(f_seq)

        if f_struct is None:
            return s

        g = self.proj_struct(f_struct)
        lam = self.gate(torch.cat([s, g], dim=-1))

        if struct_available is not None:
            mask = (~struct_available).float().unsqueeze(-1)
            lam = lam * (1 - mask) + mask

        return lam * s + (1 - lam) * g


# ============================================================
# 6. Full ESM-based Multimodal Binding Model
# ============================================================

class ESMMultimodalBindingModel(nn.Module):
    """
    ESM-based multimodal TCR-pMHC binding predictor.

    Compared to MultimodalBindingModel (Model A):
      - Replaces VAE encoders with pretrained ESM-2
      - No reconstruction loss, no KL divergence
      - Simpler training: single BCE loss
      - Much richer sequence representations from ESM-2

    Sequence features: [pool_tcr, pool_mhc, cross_tcr, cross_mhc]
    Structure features: [g_emb, struct_seq_cross]
    → Gated fusion → BatchNorm + residual classifier → binding logit
    """

    def __init__(
        self,
        # ESM configuration
        esm_model: nn.Module,
        esm_hidden_size: int = 480,
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        # LoRA configuration
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_n_layers: int = 4,
        lora_pmhc: bool = False,       # whether pMHC gets its own adapter
        # Working dimension (all cross-attention and fusion operates at this dim)
        d_model: int = 256,
        n_cross_heads: int = 8,
        dropout: float = 0.2,
        # Structure encoder config
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        egnn_hidden: int = 128,
        egnn_layers: int = 5,
        egnn_out_dim: int = 128,
        # Cross-modal attention
        struct_seq_cross_heads: int = 4,
        # Fusion / classifier
        d_fused: int = 256,
        clf_hidden: int = 256,
        # Loss
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout
        self.use_lora = use_lora
        self.lora_pmhc = lora_pmhc
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ===== Sequence branch: ESM encoder =====
        # Shared ESM backbone — always a single copy.
        # Specialization is achieved via:
        #   (a) chain-specific LoRA adapters (patched in/out before each forward)
        #   (b) separate projection heads for TCR vs pMHC
        self.esm_encoder = ESMSequenceEncoder(
            esm_model=esm_model,
            esm_hidden_size=esm_hidden_size,
            d_model=d_model,
            freeze_esm=freeze_esm,
            n_tune_layers=n_tune_layers,
            dropout=dropout,
        )

        # Separate projection head for pMHC (TCR uses the default self.esm_encoder.proj)
        self.pmhc_proj = nn.Sequential(
            nn.Linear(esm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ===== LoRA adapters (optional) =====
        if use_lora:
            # TCR always gets a LoRA adapter (high sequence diversity)
            self.tcr_lora = LoRAAdapter(
                esm_model, n_layers=lora_n_layers,
                rank=lora_rank, alpha=lora_alpha,
            )
            # pMHC adapter is optional — low diversity means it can overfit
            if lora_pmhc:
                self.pmhc_lora = LoRAAdapter(
                    esm_model, n_layers=lora_n_layers,
                    rank=lora_rank, alpha=lora_alpha,
                )
            else:
                self.pmhc_lora = None
        else:
            self.tcr_lora = None
            self.pmhc_lora = None

        # Sequence cross-attention (TCR ↔ pMHC)
        self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)

        # Sequence feature dim: pool_tcr + pool_mhc + cross_tcr + cross_mhc
        d_seq_feat = 4 * d_model

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
            d_struct=egnn_hidden,
            d_out=egnn_out_dim,
            n_heads=struct_seq_cross_heads,
            dropout=dropout,
        )

        d_struct_feat = egnn_out_dim + egnn_out_dim

        # ===== Gated fusion =====
        self.fusion = GatedFusion(d_seq_feat, d_struct_feat, d_fused)

        # ===== Classifier (LayerNorm + residual) =====
        # NOTE: Using LayerNorm instead of BatchNorm1d for bf16/mixed-precision
        # stability. BatchNorm running stats can overflow in bf16.
        self.clf_proj = nn.Linear(d_fused, clf_hidden)
        self.clf_fc1 = nn.Linear(d_fused, clf_hidden)
        self.clf_ln1 = nn.LayerNorm(clf_hidden)
        self.clf_fc2 = nn.Linear(clf_hidden, clf_hidden)
        self.clf_ln2 = nn.LayerNorm(clf_hidden)
        self.clf_out = nn.Linear(clf_hidden, 1)

    def _classify(self, fused):
        """LayerNorm + residual classifier head."""
        res = self.clf_proj(fused)
        out = self.clf_fc1(fused)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_ln1(out)
        out = 0.5 * (res + out)
        res = out
        out = self.clf_fc2(out)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.clf_ln2(out)
        out = 0.5 * (res + out)
        return self.clf_out(out)

    def forward(
        self,
        tcr_ids: torch.Tensor,               # [B, L_tcr] ESM tokenized
        mhc_ids: torch.Tensor,               # [B, L_mhc] ESM tokenized
        tcr_mask: torch.Tensor = None,        # [B, L_tcr] bool
        mhc_mask: torch.Tensor = None,        # [B, L_mhc] bool
        struct_graph=None,                     # DGL batched graph or None
        struct_available: torch.Tensor = None, # [B] bool
        labels: torch.Tensor = None,           # [B] float {0,1}
        compute_loss: bool = False,
    ):
        # ===== 1. Sequence encoding via ESM =====
        # TCR encoding — with LoRA adapter if enabled
        if self.tcr_lora is not None:
            self.esm_encoder.apply_adapter(self.tcr_lora)
        h_tcr, pool_tcr = self.esm_encoder(tcr_ids, tcr_mask)   # [B, L, D], [B, D]

        # pMHC encoding — swap adapter, use separate projection
        if self.tcr_lora is not None:
            self.esm_encoder.remove_adapter()
        if self.pmhc_lora is not None:
            self.esm_encoder.apply_adapter(self.pmhc_lora)

        # Run ESM backbone for pMHC, but use the separate pMHC projection
        attention_mask = mhc_mask.long() if mhc_mask is not None else None
        if self.esm_encoder.freeze_esm and self.esm_encoder._active_adapter is None:
            with torch.no_grad():
                esm_out_mhc = self.esm_encoder.esm_model(
                    input_ids=mhc_ids, attention_mask=attention_mask,
                ).last_hidden_state
            esm_out_mhc = esm_out_mhc.detach()
        else:
            esm_out_mhc = self.esm_encoder.esm_model(
                input_ids=mhc_ids, attention_mask=attention_mask,
            ).last_hidden_state
        h_mhc = self.pmhc_proj(esm_out_mhc)   # [B, L, d_model]
        pool_mhc = h_mhc[:, 0, :]              # [B, d_model]

        # Clean up adapter state
        if self.pmhc_lora is not None:
            self.esm_encoder.remove_adapter()

        # ===== 2. Sequence cross-attention (TCR ↔ pMHC) =====
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)

        f_seq = torch.cat([pool_tcr, pool_mhc, cross_tcr, cross_mhc], dim=-1)

        # ===== 3. Structure encoding (if available) =====
        f_struct = None
        if struct_graph is not None:
            g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)

            h_seq_concat = torch.cat([h_tcr, h_mhc], dim=1)
            if tcr_mask is not None and mhc_mask is not None:
                seq_mask_concat = torch.cat([tcr_mask, mhc_mask], dim=1)
            else:
                seq_mask_concat = None

            struct_cross = self.struct_seq_cross(
                node_h, batch_ids, h_seq_concat, seq_mask_concat
            )

            f_struct = torch.cat([g_emb, struct_cross], dim=-1)

        # ===== 4. Gated fusion =====
        fused = self.fusion(f_seq, f_struct, struct_available)

        # ===== 5. Classification =====
        logit = self._classify(fused)

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "pool_tcr": pool_tcr, "pool_mhc": pool_mhc,
            "cross_tcr": cross_tcr, "cross_mhc": cross_mhc,
            "f_seq": f_seq,
            "f_struct": f_struct,
            "fused": fused,
        }

        if not compute_loss:
            return out

        # ===== 6. Loss (binding only — no reconstruction, no KL) =====
        # Clamp logits to prevent overflow in bf16 mixed precision
        clamped_logit = logit.view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            clamped_logit, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )

        out.update({
            "loss": bind_loss,
            "bind_loss": bind_loss,
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

    # ---- Utility: switch ESM freezing mode ----
    def set_esm_tuning(self, freeze: bool = True, n_tune_layers: int = 0):
        """
        Switch between frozen and fine-tuned ESM at any point during training.
        Useful for phased training:
          Phase 1: freeze ESM, train projection + cross-attention + classifier (+ LoRA if enabled)
          Phase 2: unfreeze last N ESM layers for end-to-end fine-tuning
        Note: LoRA params are always trainable regardless of this setting.
        """
        self.esm_encoder._configure_freezing(freeze, n_tune_layers)
        mode = "frozen" if freeze else f"tuning last {n_tune_layers} layers" if n_tune_layers else "fully tunable"
        lora_info = ""
        if self.tcr_lora is not None:
            tcr_params = sum(p.numel() for p in self.tcr_lora.parameters())
            lora_info += f" | TCR LoRA: {tcr_params:,} params"
        if self.pmhc_lora is not None:
            pmhc_params = sum(p.numel() for p in self.pmhc_lora.parameters())
            lora_info += f" | pMHC LoRA: {pmhc_params:,} params"
        print(f"[ESM] {mode}{lora_info}")


# ============================================================
# 7. Smoke test
# ============================================================

if __name__ == "__main__":
    import dgl
    from types import SimpleNamespace

    # --- Mock ESM model that mimics HuggingFace EsmModel structure ---
    class MockSelfAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)

    class MockAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.self = MockSelfAttention(hidden_size)

    class MockTransformerLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = MockAttention(hidden_size)
            self.ff = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            return self.norm(x + self.ff(x))

    class MockESMModel(nn.Module):
        """Mimics transformers.EsmModel with encoder.layer structure."""
        def __init__(self, hidden_size=480, n_layers=6, vocab_size=33):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([
                MockTransformerLayer(hidden_size) for _ in range(n_layers)
            ])

        def forward(self, input_ids, attention_mask=None):
            x = self.embeddings(input_ids)
            for layer in self.encoder.layer:
                x = layer(x)
            return SimpleNamespace(last_hidden_state=x)

    ESM_HIDDEN = 480
    D_MODEL = 256
    B = 4
    L_TCR, L_MHC = 30, 34
    V_ESM = 33

    mock_esm = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)

    # --- Test 1: No LoRA (backward compatible) ---
    model = ESMMultimodalBindingModel(
        esm_model=mock_esm,
        esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True,
        d_model=D_MODEL,
        n_cross_heads=4,
        node_feat_size=20, edge_feat_size=0,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4,
        d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0,
    )

    # --- Dummy data ---
    tcr = torch.randint(1, V_ESM, (B, L_TCR))
    mhc = torch.randint(1, V_ESM, (B, L_MHC))
    tcr_mask = torch.ones(B, L_TCR, dtype=torch.bool)
    mhc_mask = torch.ones(B, L_MHC, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,)).float()

    graphs = []
    for _ in range(B):
        n = torch.randint(40, 80, (1,)).item()
        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        g = dgl.graph((src, dst))
        g.ndata["x"] = torch.randn(n, 20)
        g.ndata["coords"] = torch.randn(n, 3)
        graphs.append(g)
    bg = dgl.batch(graphs)
    struct_avail = torch.ones(B, dtype=torch.bool)

    # Forward with structure (no LoRA)
    out = model(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("=== No LoRA — With structure ===")
    print(f"Loss:  {out['loss']:.4f}")
    print(f"Probs: {out['prob'].squeeze(-1).tolist()}")

    # Forward without structure (no LoRA)
    out2 = model(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=None, struct_available=None,
        labels=labels, compute_loss=True,
    )
    print("\n=== No LoRA — Sequence only ===")
    print(f"Loss:  {out2['loss']:.4f}")
    print(f"Probs: {out2['prob'].squeeze(-1).tolist()}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nNo LoRA — Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # --- Test 2: With LoRA (TCR only) ---
    mock_esm2 = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_lora = ESMMultimodalBindingModel(
        esm_model=mock_esm2,
        esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
        lora_pmhc=False,
        d_model=D_MODEL,
        n_cross_heads=4,
        node_feat_size=20, edge_feat_size=0,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4,
        d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0,
    )

    out3 = model_lora(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("\n=== TCR-only LoRA — With structure ===")
    print(f"Loss:  {out3['loss']:.4f}")
    print(f"Probs: {out3['prob'].squeeze(-1).tolist()}")

    model_lora.set_esm_tuning(freeze=True, n_tune_layers=0)

    lora_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model_lora.parameters())
    tcr_lora_params = sum(p.numel() for p in model_lora.tcr_lora.parameters())
    print(f"\nTCR LoRA params: {tcr_lora_params:,}")
    print(f"LoRA model — Trainable: {lora_trainable:,} / {lora_total:,} ({100*lora_trainable/lora_total:.1f}%)")

    # --- Test 3: With LoRA (TCR + pMHC) ---
    mock_esm3 = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_both = ESMMultimodalBindingModel(
        esm_model=mock_esm3,
        esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
        lora_pmhc=True,
        d_model=D_MODEL,
        n_cross_heads=4,
        node_feat_size=20, edge_feat_size=0,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4,
        d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0,
    )

    out4 = model_both(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("\n=== TCR+pMHC LoRA — With structure ===")
    print(f"Loss:  {out4['loss']:.4f}")
    print(f"Probs: {out4['prob'].squeeze(-1).tolist()}")

    model_both.set_esm_tuning(freeze=True, n_tune_layers=0)

    both_trainable = sum(p.numel() for p in model_both.parameters() if p.requires_grad)
    both_total = sum(p.numel() for p in model_both.parameters())
    print(f"\nBoth LoRA model — Trainable: {both_trainable:,} / {both_total:,} ({100*both_trainable/both_total:.1f}%)")