"""
ESM-Based Multimodal TCR–pMHC Binding Prediction Model (Model B — v2)
======================================================================

Fixes over v1:
  [W1] Edge type features: etype one-hot (5-dim) appended to edge features
       so EGNN distinguishes interface vs intra-chain edges
  [W2] ESM pooling: masked mean pool over non-padding tokens instead of CLS
       (ESM-2 has no CLS training objective)
  [W3] EGNN improvements:
       - Pre-LN residual connections between layers
       - Dropout between layers
       - Enriched node features: one-hot AA (20) + hbond_acc (1) + hbond_don (1)
         + sidechain_vec (3) + chain_id one-hot (4) = 29 dims
  [W4] Bidirectional structure↔sequence cross-attention: sequence tokens are
       updated by structural context (not just structure by sequence)
  [W5] Vector-valued gated fusion: per-dimension gate instead of scalar

Architecture:
  1. TCR sequence  →  ESM-2 encoder (+ optional LoRA)  →  TCR projection  →  h_tcr
  2. pMHC sequence →  ESM-2 encoder (+ optional LoRA)  →  pMHC projection →  h_mhc
  3. Bidirectional cross-attention:  h_tcr ↔ h_mhc
  4. Structure graph →  EGNN (with residuals + enriched features)  →  g_struct
  5. Bidirectional Structure ↔ Sequence cross-attention
  6. Vector-gated fusion  →  LayerNorm + residual classifier  →  binding logit

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
# 0. LoRA — Low-Rank Adaptation Modules (unchanged from v1)
# ============================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.

    Adds a trainable low-rank bypass:  y = W_frozen·x + (B·A)·x · (α/r)
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, and the original W is frozen.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAAdapter(nn.Module):
    """
    A set of LoRA adapters for the last N layers of an ESM encoder.
    Each adapted layer gets LoRA on both the self-attention Q and V projections.
    """

    def __init__(self, esm_model: nn.Module, n_layers: int = 4,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.alpha = alpha

        encoder_layers = list(esm_model.encoder.layer)
        target_layers = encoder_layers[-n_layers:] if n_layers > 0 else []

        self.lora_modules = nn.ModuleDict()
        self._original_modules = {}

        for layer in target_layers:
            layer_idx = encoder_layers.index(layer)
            attn = layer.attention.self
            lora_q = LoRALinear(attn.query, rank=rank, alpha=alpha)
            lora_v = LoRALinear(attn.value, rank=rank, alpha=alpha)
            self.lora_modules[f"layer_{layer_idx}_q"] = lora_q
            self.lora_modules[f"layer_{layer_idx}_v"] = lora_v

    def apply_to(self, esm_model: nn.Module):
        """Patch LoRA adapters into the ESM model (in-place)."""
        encoder_layers = list(esm_model.encoder.layer)
        for key, lora_mod in self.lora_modules.items():
            parts = key.split("_")
            layer_idx = int(parts[1])
            target = parts[2]
            attn = encoder_layers[layer_idx].attention.self
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
#    [W2] FIX: Masked mean pooling instead of CLS token
# ============================================================

class ESMSequenceEncoder(nn.Module):
    """
    Wraps a full HuggingFace EsmModel to produce projected hidden states
    and a pooled sequence vector.

    [W2] Uses masked mean pooling over non-padding tokens instead of
    CLS token at index 0. ESM-2 was not trained with a CLS objective,
    so the BOS token representation is not a meaningful sequence summary.
    Mean pooling aggregates information from all residue positions.
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
        self.esm_model = esm_model
        self.esm_hidden_size = esm_hidden_size
        self.freeze_esm = freeze_esm
        self._active_adapter = None

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
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_model)
        adapter.apply_to(self.esm_model)
        self._active_adapter = adapter

    def remove_adapter(self):
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_model)
            self._active_adapter = None

    @staticmethod
    def _masked_mean_pool(h_seq, mask):
        """
        [W2] Mean pool over non-padding positions.

        Args:
            h_seq: [B, L, D]
            mask:  [B, L] bool, True = valid token
        Returns:
            [B, D]
        """
        if mask is None:
            return h_seq.mean(dim=1)
        m = mask.float().unsqueeze(-1)  # [B, L, 1]
        return (h_seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, token_ids, mask=None):
        """
        Args:
            token_ids: [B, L] ESM tokenized input (includes BOS/EOS)
            mask: [B, L] bool, True=valid token
        Returns:
            h_seq:  [B, L, d_model] projected hidden states
            h_pool: [B, d_model]    masked mean pool (NOT CLS)
        """
        attention_mask = mask.long() if mask is not None else None

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

        h_seq = self.proj(esm_out)                        # [B, L, d_model]
        h_pool = self._masked_mean_pool(h_seq, mask)      # [B, d_model]
        return h_seq, h_pool


# ============================================================
# 2. Cross-attention (unchanged from v1)
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
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.clamp(-50.0, 50.0)
        if ctx_mask is not None:
            scores = scores.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
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
# 3. Structure EGNN
#    [W1] FIX: Edge type one-hot appended to edge features
#    [W3] FIX: Pre-LN residual connections, inter-layer dropout,
#              enriched node features (29-dim)
# ============================================================

NUM_EDGE_TYPES = 5  # tcr2tcr, pep2pep, mhc2mhc, pep_mhc, tcr_pmhc
NUM_EXTRA_NODE_FEAT = 9  # hbond_acc(1) + hbond_don(1) + sidechain_vec(3) + chain_id_onehot(4)

# 1-letter AA → ESM-2 token index mapping
# ESM-2 vocabulary: 0=<cls>, 1=<pad>, 2=<eos>, 3=<unk>, then standard AAs
# Standard 20 AAs in ESM-2 vocab (indices 4-23):
_AA1_TO_ESM_IDX = {
    "L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11,
    "I": 12, "D": 13, "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18,
    "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23,
    "X": 3, "U": 3, "B": 3, "Z": 3, "O": 3, "J": 3,  # unknowns → <unk>
}


class StructureEGNN(nn.Module):
    """
    EGNN encoder for TCR-pMHC complex graph.

    Node feature source (controlled by `node_feat_source`):
      - "onehot":    20-dim AA one-hot (original v1 behavior)
      - "embedding": ESM-2 word embedding lookup (480-dim, frozen).
                     No transformer forward pass needed. Each node gets
                     the ESM learned embedding for its amino acid.
      - "encoder":   Full ESM encoder hidden states mapped to graph nodes
                     via chain_id + chain_pos. Requires ESM to run first
                     and hidden states passed in at forward time.

    Edge features: 7 bond types + 5 edge types = 12 dims.
    Extra node features: hbond_acc(1) + hbond_don(1) + sidechain_vec(3) + chain_onehot(4) = 9 dims.
    """

    VALID_NODE_FEAT_SOURCES = {"onehot", "embedding", "encoder"}

    def __init__(
        self,
        node_feat_source: str = "embedding",
        esm_embedding_layer: nn.Module = None,  # required for "embedding" mode
        esm_hidden_size: int = 480,              # ESM hidden dim (for embedding/encoder)
        edge_feat_size: int = 12,    # 7 bond types + 5 edge types
        hidden_dim: int = 128,
        n_layers: int = 5,
        d_out: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert node_feat_source in self.VALID_NODE_FEAT_SOURCES, \
            f"node_feat_source must be one of {self.VALID_NODE_FEAT_SOURCES}"
        self.node_feat_source = node_feat_source
        self.hidden_dim = hidden_dim
        self.n_layers_count = n_layers
        self.esm_hidden_size = esm_hidden_size

        # ---- Node feature input dimension depends on source ----
        if node_feat_source == "onehot":
            aa_dim = 20
        elif node_feat_source == "embedding":
            aa_dim = esm_hidden_size  # 480 for ESM-2 t12
            assert esm_embedding_layer is not None, \
                "esm_embedding_layer required for node_feat_source='embedding'"
            self.esm_word_emb = esm_embedding_layer
            # Freeze the embedding lookup — it's a shared reference
            for p in self.esm_word_emb.parameters():
                p.requires_grad = False
        elif node_feat_source == "encoder":
            aa_dim = esm_hidden_size  # will receive projected ESM hidden states

        node_feat_size = aa_dim + NUM_EXTRA_NODE_FEAT  # +9

        # ---- Input projection: (aa_dim + 9) → hidden_dim ----
        self.input_proj = nn.Linear(node_feat_size, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EGNNConv(
                in_size=hidden_dim, hidden_size=hidden_dim,
                out_size=hidden_dim, edge_feat_size=edge_feat_size,
            ))

        # Pre-LayerNorm + dropout for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_layers)
        ])

        # Final norm + projection
        self.final_norm = nn.LayerNorm(hidden_dim)
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

    def _get_aa_features_onehot(self, bg, N, device):
        """Original 20-dim one-hot AA encoding from ndata['x']."""
        return bg.ndata["x"].float()  # [N, 20]

    def _get_aa_features_embedding(self, bg, N, device):
        """
        Look up each node's amino acid in ESM's word embedding table.

        Uses ndata["x"] (20-dim one-hot) to recover the AA identity,
        maps to ESM token index, then does a frozen embedding lookup.
        Returns [N, esm_hidden_size] (e.g., [N, 480]).
        """
        # Recover AA index from one-hot (argmax). All-zeros → <unk>
        x_onehot = bg.ndata["x"].float()  # [N, 20]
        aa_idx = x_onehot.argmax(dim=-1)  # [N]
        is_unknown = (x_onehot.sum(dim=-1) < 0.5)  # all-zeros = MASK/unknown

        # Map AA index (0-19 in our one-hot) to ESM token index
        # Our one-hot order: ALA(0) CYS(1) ASP(2) ... TYR(19) — from enc_dict
        AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_esm = torch.zeros(20, dtype=torch.long, device=device)
        for i, aa in enumerate(AA_ORDER):
            aa_to_esm[i] = _AA1_TO_ESM_IDX.get(aa, 3)  # 3 = <unk>

        esm_token_ids = aa_to_esm[aa_idx.clamp(0, 19)]  # [N]
        esm_token_ids[is_unknown] = 3  # <unk>

        with torch.no_grad():
            aa_emb = self.esm_word_emb(esm_token_ids)  # [N, esm_hidden_size]

        return aa_emb

    def _get_aa_features_encoder(self, bg, N, device, esm_node_features):
        """
        Use pre-computed ESM encoder hidden states mapped to graph nodes.
        esm_node_features: [N_total, esm_hidden_size] — already aligned.
        """
        assert esm_node_features is not None, \
            "node_feat_source='encoder' requires esm_node_features in forward()"
        return esm_node_features.float()  # [N, esm_hidden_size]

    def forward(self, bg, esm_node_features=None):
        """
        Args:
            bg: DGL batched graph with ndata/edata
            esm_node_features: [N_total, esm_hidden_size] — only needed for
                               node_feat_source='encoder'. Pre-computed ESM
                               hidden states mapped to graph node positions.
        Returns:
            g_emb:     [B, d_out]      global graph embedding
            node_h:    [N, hidden_dim]  per-node hidden states
            batch_ids: [N]             graph index per node
        """
        device = bg.device
        N = bg.num_nodes()

        # ---- Get AA features based on source ----
        if self.node_feat_source == "onehot":
            x_aa = self._get_aa_features_onehot(bg, N, device)
        elif self.node_feat_source == "embedding":
            x_aa = self._get_aa_features_embedding(bg, N, device)
        elif self.node_feat_source == "encoder":
            x_aa = self._get_aa_features_encoder(bg, N, device, esm_node_features)

        coords = bg.ndata["coords"].float()  # [N, 3]

        # ---- Extra node features (same for all sources) ----
        hacc = bg.ndata.get("hbond_acceptors", torch.zeros(N, 1, device=device)).float()
        hdon = bg.ndata.get("hbond_donors", torch.zeros(N, 1, device=device)).float()
        scv = bg.ndata.get("sidechain_vector", torch.zeros(N, 3, device=device)).float()
        chain_ids = bg.ndata.get("chain_id", torch.zeros(N, dtype=torch.long, device=device))
        chain_onehot = F.one_hot(chain_ids.clamp(0, 3), num_classes=4).float()

        if hacc.dim() == 1: hacc = hacc.unsqueeze(1)
        if hdon.dim() == 1: hdon = hdon.unsqueeze(1)

        # Concatenate: [N, aa_dim + 9]
        h_input = torch.cat([x_aa, hacc, hdon, scv, chain_onehot], dim=-1)

        # ---- Edge features with etype ----
        edge_feat = bg.edata.get("feat", None)
        etype = bg.edata.get("etype", None)

        if edge_feat is not None:
            edge_feat = edge_feat.float()
        if etype is not None:
            etype_onehot = F.one_hot(etype.clamp(0, NUM_EDGE_TYPES - 1),
                                      num_classes=NUM_EDGE_TYPES).float()
            if edge_feat is not None:
                edge_feat = torch.cat([edge_feat, etype_onehot], dim=-1)
            else:
                edge_feat = etype_onehot

        # ---- Project to hidden dim ----
        h = self.input_proj(h_input)  # [N, hidden_dim]

        # ---- EGNN layers with pre-LN residual ----
        with torch.autocast(device_type=device.type, enabled=False):
            for i, layer in enumerate(self.layers):
                h_normed = self.layer_norms[i](h)
                if edge_feat is not None:
                    h_new, coords = layer(bg, h_normed, coords, edge_feat)
                else:
                    h_new, coords = layer(bg, h_normed, coords)

                h_new = torch.nan_to_num(h_new, nan=0.0, posinf=1e4, neginf=-1e4)
                coords = torch.nan_to_num(coords, nan=0.0, posinf=1e4, neginf=-1e4)
                h = h + self.layer_dropouts[i](h_new)

        h = self.final_norm(h)
        batch_ids = self._make_batch_ids(bg)
        g_emb = global_mean_pool(h, batch_ids)
        g_emb = self.out_proj(g_emb)
        return g_emb, h, batch_ids


# ============================================================
# 4. Bidirectional Structure ↔ Sequence Cross-Attention
#    [W4] FIX: Both directions — struct attends to seq AND seq
#         attends to struct. Returns both pooled representations.
# ============================================================

class BidirectionalStructureSequenceCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between structure graph nodes and
    sequence hidden states.

    [W4] Two directions:
      - struct_attended: structure nodes query, sequence tokens as KV
      - seq_attended:    sequence tokens query, structure nodes as KV
    Both are pooled and returned, so sequence representations are
    also updated by structural context.
    """

    def __init__(self, d_seq, d_struct, d_out, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_k = d_out // n_heads
        assert d_out % n_heads == 0

        # Projections to shared dimension
        self.proj_seq = nn.Linear(d_seq, d_out)
        self.proj_struct = nn.Linear(d_struct, d_out)

        # Direction 1: struct → seq (struct queries, seq KV)
        self.s2q_W_q = nn.Linear(d_out, d_out)
        self.s2q_W_k = nn.Linear(d_out, d_out)
        self.s2q_W_v = nn.Linear(d_out, d_out)
        self.s2q_out = nn.Linear(d_out, d_out)
        self.s2q_norm = nn.LayerNorm(d_out)

        # Direction 2: seq → struct (seq queries, struct KV)
        self.q2s_W_q = nn.Linear(d_out, d_out)
        self.q2s_W_k = nn.Linear(d_out, d_out)
        self.q2s_W_v = nn.Linear(d_out, d_out)
        self.q2s_out = nn.Linear(d_out, d_out)
        self.q2s_norm = nn.LayerNorm(d_out)

        self.dropout = nn.Dropout(dropout)

    def _attend(self, q, kv, W_q, W_k, W_v, out_proj, norm, kv_mask=None):
        """Single-sample cross-attention (unbatched: q=[1, Nq, D], kv=[1, Nk, D])."""
        Q = W_q(q).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = W_k(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = W_v(kv).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.clamp(-50.0, 50.0)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ V).transpose(1, 2).contiguous().view(q.shape)
        return norm(out_proj(out) + q)

    def forward(self, node_h, batch_ids, h_seq_concat, seq_mask_concat):
        """
        Args:
            node_h:          [N_total, d_struct] — all graph nodes (batched)
            batch_ids:       [N_total] — graph index per node
            h_seq_concat:    [B, L_seq, d_seq] — concatenated TCR+pMHC seq hidden states
            seq_mask_concat: [B, L_seq] bool — mask for sequence tokens

        Returns:
            struct_pooled: [B, d_out] — structure enriched by sequence context
            seq_pooled:    [B, d_out] — sequence enriched by structural context
        """
        node_h_proj = self.proj_struct(node_h)   # [N_total, d_out]
        h_seq_proj = self.proj_seq(h_seq_concat)  # [B, L, d_out]

        B = batch_ids.max().item() + 1
        struct_pooled = []
        seq_pooled = []

        for i in range(B):
            node_mask = (batch_ids == i)
            n_nodes = node_mask.sum().item()

            if n_nodes == 0:
                struct_pooled.append(torch.zeros(1, self.d_out,
                                                  device=node_h.device, dtype=node_h.dtype))
                seq_pooled.append(torch.zeros(1, self.d_out,
                                               device=node_h.device, dtype=node_h.dtype))
                continue

            struct_q = node_h_proj[node_mask].unsqueeze(0)   # [1, N_i, d_out]
            seq_kv = h_seq_proj[i:i+1]                       # [1, L, d_out]
            seq_m = seq_mask_concat[i:i+1] if seq_mask_concat is not None else None

            # Direction 1: structure attends to sequence
            s_attended = self._attend(
                struct_q, seq_kv,
                self.s2q_W_q, self.s2q_W_k, self.s2q_W_v,
                self.s2q_out, self.s2q_norm,
                kv_mask=seq_m,
            )
            struct_pooled.append(s_attended.mean(dim=1))  # [1, d_out]

            # Direction 2: sequence attends to structure
            # No mask on struct side (all nodes are valid)
            q_attended = self._attend(
                seq_kv, struct_q,
                self.q2s_W_q, self.q2s_W_k, self.q2s_W_v,
                self.q2s_out, self.q2s_norm,
                kv_mask=None,
            )
            # Masked mean pool over sequence positions
            if seq_m is not None:
                m = seq_m.float().unsqueeze(-1)  # [1, L, 1]
                seq_pooled.append((q_attended * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0))
            else:
                seq_pooled.append(q_attended.mean(dim=1))

        return torch.cat(struct_pooled, dim=0), torch.cat(seq_pooled, dim=0)


# ============================================================
# 5. Vector-Gated Fusion
#    [W5] FIX: Per-dimension gating instead of scalar gate
# ============================================================

class VectorGatedFusion(nn.Module):
    """
    Per-dimension gated fusion of sequence and structure features.

    [W5] Instead of a single scalar λ ∈ [0,1], produces a d_fused-dimensional
    gate vector. This allows the model to trust structure for some feature
    dimensions (e.g., interface geometry) and sequence for others (e.g.,
    evolutionary signals) on a per-sample, per-dimension basis.

    output = λ ⊙ s + (1 - λ) ⊙ g
    where λ ∈ R^{d_fused}, ⊙ is element-wise multiply.
    """

    def __init__(self, d_seq, d_struct, d_fused):
        super().__init__()
        self.proj_seq = nn.Linear(d_seq, d_fused)
        self.proj_struct = nn.Linear(d_struct, d_fused)

        # [W5] Gate outputs d_fused dimensions instead of 1
        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.ReLU(),
            nn.Linear(d_fused, d_fused),    # <-- d_fused, not 1
            nn.Sigmoid(),
        )

    def forward(self, f_seq, f_struct=None, struct_available=None):
        s = self.proj_seq(f_seq)            # [B, d_fused]

        if f_struct is None:
            return s

        g = self.proj_struct(f_struct)       # [B, d_fused]
        lam = self.gate(torch.cat([s, g], dim=-1))  # [B, d_fused]

        # When structure is unavailable for a sample, force gate to 1 (all seq)
        if struct_available is not None:
            mask = (~struct_available).float().unsqueeze(-1)  # [B, 1]
            lam = lam * (1 - mask) + mask  # lam → 1 where struct unavailable

        return lam * s + (1 - lam) * g


# ============================================================
# 6. Full ESM-based Multimodal Binding Model (v2)
# ============================================================

class ResidualClassifier(nn.Module):
    """
    Reusable LayerNorm + residual classifier head.
    Factored out so each mode (seq / struct / fusion) can have its own.
    """

    def __init__(self, d_in, d_hidden, dropout=0.2):
        super().__init__()
        self.dropout_p = dropout
        self.proj = nn.Linear(d_in, d_hidden)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, x):
        res = self.proj(x)
        out = self.fc1(x)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.ln1(out)
        out = 0.5 * (res + out)
        res = out
        out = self.fc2(out)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = F.leaky_relu(out)
        out = self.ln2(out)
        out = 0.5 * (res + out)
        return self.out(out)


class ESMMultimodalBindingModel(nn.Module):
    """
    ESM-based multimodal TCR-pMHC binding predictor (v2).

    Supports three ablation modes via the `mode` parameter:
      - "seq+struct" (default): Full multimodal model with gated fusion
      - "seq_only":             Sequence branch only (EGNN not executed)
      - "struct_only":          Structure branch only (ESM not executed)

    Each mode has its own classifier head with independent weights so
    ablation comparisons are not confounded by shared classifier capacity.

    In "struct_only" mode, the EGNN output is projected directly to the
    classifier without any cross-attention to sequence features. This
    makes it a pure graph model.

    Changes from v1:
      [W1] Edge type features in EGNN
      [W2] Mean pooling in ESM encoder
      [W3] EGNN residuals + enriched node features
      [W4] Bidirectional structure↔sequence cross-attention
      [W5] Vector-gated fusion
    """

    VALID_MODES = {"seq+struct", "seq_only", "struct_only"}
    VALID_NODE_FEAT_SOURCES = {"onehot", "embedding", "encoder"}

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
        lora_pmhc: bool = False,
        # Working dimension
        d_model: int = 256,
        n_cross_heads: int = 8,
        dropout: float = 0.2,
        # Structure encoder config
        edge_feat_size: int = 12,    # 7 bond types + 5 edge types
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
        # Ablation mode
        mode: str = "seq+struct",
        # EGNN node feature source
        node_feat_source: str = "embedding",
    ):
        super().__init__()
        assert mode in self.VALID_MODES, \
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"
        assert node_feat_source in self.VALID_NODE_FEAT_SOURCES, \
            f"node_feat_source must be one of {self.VALID_NODE_FEAT_SOURCES}"
        self.mode = mode
        self.node_feat_source = node_feat_source
        self.d_model = d_model
        self.dropout_p = dropout
        self.use_lora = use_lora
        self.lora_pmhc = lora_pmhc
        self.esm_hidden_size = esm_hidden_size
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # Store dims for external access (e.g., pretraining script)
        d_seq_feat = 4 * d_model
        d_struct_graph_only = egnn_out_dim
        d_struct_feat = egnn_out_dim * 3

        self._d_seq_feat = d_seq_feat
        self._d_struct_feat = d_struct_feat
        self._d_struct_graph_only = d_struct_graph_only
        self._d_fused = d_fused

        # ===== Sequence branch (built unless struct_only with non-encoder nodes) =====
        # Note: even struct_only needs ESM embeddings if node_feat_source="embedding"
        # But struct_only with encoder needs the full ESM encoder → validate
        if mode == "struct_only" and node_feat_source == "encoder":
            raise ValueError(
                "struct_only + encoder is contradictory: encoder mode requires "
                "running ESM forward, which makes it not struct_only. "
                "Use node_feat_source='embedding' for struct_only mode."
            )

        need_esm_seq_branch = (mode != "struct_only")
        if need_esm_seq_branch:
            self.esm_encoder = ESMSequenceEncoder(
                esm_model=esm_model,
                esm_hidden_size=esm_hidden_size,
                d_model=d_model,
                freeze_esm=freeze_esm,
                n_tune_layers=n_tune_layers,
                dropout=dropout,
            )
            self.pmhc_proj = nn.Sequential(
                nn.Linear(esm_hidden_size, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            if use_lora:
                self.tcr_lora = LoRAAdapter(
                    esm_model, n_layers=lora_n_layers,
                    rank=lora_rank, alpha=lora_alpha,
                )
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

            self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)
        else:
            self.esm_encoder = None
            self.pmhc_proj = None
            self.tcr_lora = None
            self.pmhc_lora = None
            self.seq_cross_attn = None

        # ===== Structure branch (built unless seq_only) =====
        if mode != "seq_only":
            # Get ESM embedding layer for "embedding" mode
            esm_emb_layer = None
            if node_feat_source == "embedding":
                esm_emb_layer = esm_model.embeddings.word_embeddings

            self.struct_encoder = StructureEGNN(
                node_feat_source=node_feat_source,
                esm_embedding_layer=esm_emb_layer,
                esm_hidden_size=esm_hidden_size,
                edge_feat_size=edge_feat_size,
                hidden_dim=egnn_hidden,
                n_layers=egnn_layers,
                d_out=egnn_out_dim,
                dropout=dropout,
            )
        else:
            self.struct_encoder = None

        # ===== Cross-modal attention (only for seq+struct) =====
        if mode == "seq+struct":
            self.struct_seq_cross = BidirectionalStructureSequenceCrossAttention(
                d_seq=d_model,
                d_struct=egnn_hidden,
                d_out=egnn_out_dim,
                n_heads=struct_seq_cross_heads,
                dropout=dropout,
            )
            self.fusion = VectorGatedFusion(d_seq_feat, d_struct_feat, d_fused)
        else:
            self.struct_seq_cross = None
            self.fusion = None

        # ===== Mode-specific classifiers =====
        if mode == "seq+struct":
            self.classifier = ResidualClassifier(d_fused, clf_hidden, dropout)
        elif mode == "seq_only":
            self.classifier = ResidualClassifier(d_seq_feat, clf_hidden, dropout)
        elif mode == "struct_only":
            self.classifier = ResidualClassifier(d_struct_graph_only, clf_hidden, dropout)

    # ------------------------------------------------------------------
    # Helper: Map ESM hidden states to graph nodes (encoder mode)
    # ------------------------------------------------------------------
    def _map_esm_to_graph_nodes(self, esm_out_mhc_raw, esm_out_tcr_raw,
                                  mhc_mask, tcr_mask, struct_graph):
        """
        Map ESM encoder hidden states back to graph nodes using chain_id
        and chain_pos stored in ndata.

        chain_id mapping: 0=MHC(A), 1=peptide(C), 2=TRA(D), 3=TRB(E)

        ESM tokenization layout:
          pMHC: [BOS, mhc_1..mhc_180, pep_1..pep_14, EOS, PAD...]
                 ^0    ^1..180          ^181..194
          TCR:  [BOS, tra_1..tra_180, trb_1..trb_180, EOS, PAD...]
                 ^0    ^1..180         ^181..360

        Note: The | separator between chains is STRIPPED during tokenization
        (dataset.py line 541: s.replace("|", "")), so the two chains are
        concatenated contiguously after BOS.

        Graph nodes have chain_pos = 0-indexed position within the chain's
        sorted residue list, directly corresponding to ESM token position
        (offset by 1 for BOS).

        Returns: [N_total, esm_hidden_size] tensor aligned to graph nodes.
        """
        N = struct_graph.num_nodes()
        device = esm_out_mhc_raw.device
        esm_d = esm_out_mhc_raw.size(-1)

        chain_ids = struct_graph.ndata["chain_id"]     # [N] long: 0,1,2,3
        chain_pos = struct_graph.ndata["chain_pos"]    # [N] long: 0-indexed in chain

        node_features = torch.zeros(N, esm_d, device=device, dtype=esm_out_mhc_raw.dtype)

        # Get batch assignment for graph nodes
        batch_ids = StructureEGNN._make_batch_ids(struct_graph)

        B = esm_out_mhc_raw.size(0)
        # Chain offset in ESM token sequence (after BOS at position 0):
        # pMHC: MHC residues at positions 1..180, peptide at 181..194
        # TCR:  TRA residues at positions 1..180, TRB at 181..360
        MHC_OFFSET = 1      # skip BOS
        PEP_OFFSET = 181    # 1 (BOS) + 180 (MHC)
        TRA_OFFSET = 1      # skip BOS
        TRB_OFFSET = 181    # 1 (BOS) + 180 (TRA)

        for b in range(B):
            mask = (batch_ids == b)
            cids = chain_ids[mask]    # chain IDs for this sample's nodes
            cpos = chain_pos[mask]    # chain positions for this sample's nodes

            sample_feats = torch.zeros(mask.sum(), esm_d, device=device,
                                        dtype=esm_out_mhc_raw.dtype)

            for local_idx in range(mask.sum()):
                cid = cids[local_idx].item()
                pos = cpos[local_idx].item()

                if cid == 0:  # MHC
                    tok_idx = MHC_OFFSET + pos
                    if tok_idx < esm_out_mhc_raw.size(1):
                        sample_feats[local_idx] = esm_out_mhc_raw[b, tok_idx]
                elif cid == 1:  # peptide
                    tok_idx = PEP_OFFSET + pos
                    if tok_idx < esm_out_mhc_raw.size(1):
                        sample_feats[local_idx] = esm_out_mhc_raw[b, tok_idx]
                elif cid == 2:  # TRA
                    tok_idx = TRA_OFFSET + pos
                    if tok_idx < esm_out_tcr_raw.size(1):
                        sample_feats[local_idx] = esm_out_tcr_raw[b, tok_idx]
                elif cid == 3:  # TRB
                    tok_idx = TRB_OFFSET + pos
                    if tok_idx < esm_out_tcr_raw.size(1):
                        sample_feats[local_idx] = esm_out_tcr_raw[b, tok_idx]

            node_features[mask] = sample_feats

        return node_features  # [N_total, esm_hidden_size]

    # ------------------------------------------------------------------
    # Sequence-only forward path
    # ------------------------------------------------------------------
    def _forward_seq_only(self, tcr_ids, mhc_ids, tcr_mask, mhc_mask):
        """Run sequence branch only. Returns f_seq [B, 4*d_model]."""
        # TCR
        if self.tcr_lora is not None:
            self.esm_encoder.apply_adapter(self.tcr_lora)
        h_tcr, pool_tcr = self.esm_encoder(tcr_ids, tcr_mask)

        # pMHC
        if self.tcr_lora is not None:
            self.esm_encoder.remove_adapter()
        if self.pmhc_lora is not None:
            self.esm_encoder.apply_adapter(self.pmhc_lora)

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
        h_mhc = self.pmhc_proj(esm_out_mhc)
        pool_mhc = ESMSequenceEncoder._masked_mean_pool(h_mhc, mhc_mask)

        if self.pmhc_lora is not None:
            self.esm_encoder.remove_adapter()

        # Cross-attention
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)
        f_seq = torch.cat([pool_tcr, pool_mhc, cross_tcr, cross_mhc], dim=-1)

        # Also return raw ESM outputs for encoder mode mapping
        return f_seq, h_tcr, h_mhc, tcr_mask, mhc_mask, esm_out_mhc, {
            "pool_tcr": pool_tcr, "pool_mhc": pool_mhc,
            "cross_tcr": cross_tcr, "cross_mhc": cross_mhc,
        }

    # ------------------------------------------------------------------
    # Structure-only forward path
    # ------------------------------------------------------------------
    def _forward_struct_only(self, struct_graph):
        """Run structure branch only. Returns g_emb [B, egnn_out_dim]."""
        # struct_only cannot use encoder mode (validated in __init__)
        g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)
        return g_emb

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        tcr_ids: torch.Tensor = None,        # [B, L_tcr] ESM tokenized
        mhc_ids: torch.Tensor = None,        # [B, L_mhc] ESM tokenized
        tcr_mask: torch.Tensor = None,        # [B, L_tcr] bool
        mhc_mask: torch.Tensor = None,        # [B, L_mhc] bool
        struct_graph=None,                     # DGL batched graph or None
        struct_available: torch.Tensor = None, # [B] bool
        labels: torch.Tensor = None,           # [B] float {0,1}
        compute_loss: bool = False,
    ):
        out = {}

        # ==========================================
        # MODE: seq_only
        # ==========================================
        if self.mode == "seq_only":
            f_seq, _, _, _, _, _, seq_out = self._forward_seq_only(
                tcr_ids, mhc_ids, tcr_mask, mhc_mask
            )
            logit = self.classifier(f_seq)
            out.update(seq_out)
            out["f_seq"] = f_seq
            out["f_struct"] = None
            out["fused"] = f_seq

        # ==========================================
        # MODE: struct_only
        # ==========================================
        elif self.mode == "struct_only":
            assert struct_graph is not None, "struct_only mode requires struct_graph"
            g_emb = self._forward_struct_only(struct_graph)
            logit = self.classifier(g_emb)
            out["f_seq"] = None
            out["f_struct"] = g_emb
            out["fused"] = g_emb

        # ==========================================
        # MODE: seq+struct (full multimodal)
        # ==========================================
        elif self.mode == "seq+struct":
            # Sequence branch
            f_seq, h_tcr, h_mhc, tcr_mask, mhc_mask, esm_out_mhc_raw, seq_out = \
                self._forward_seq_only(tcr_ids, mhc_ids, tcr_mask, mhc_mask)
            out.update(seq_out)
            out["f_seq"] = f_seq

            # Structure branch + cross-modal attention
            f_struct = None
            if struct_graph is not None:
                # For encoder mode: map ESM hidden states to graph nodes
                esm_node_feats = None
                if self.node_feat_source == "encoder":
                    # Get raw TCR ESM output (before projection)
                    # h_tcr is projected [B, L, d_model], we need raw [B, L, esm_hidden]
                    # Re-run or cache? We cached esm_out_mhc_raw already.
                    # For TCR: run ESM again or use the raw output.
                    # The ESM encoder returns projected h_tcr — we need pre-projection.
                    # Run ESM for TCR to get raw hidden states:
                    attn_mask_tcr = tcr_mask.long() if tcr_mask is not None else None
                    if self.esm_encoder.freeze_esm and self.esm_encoder._active_adapter is None:
                        with torch.no_grad():
                            esm_out_tcr_raw = self.esm_encoder.esm_model(
                                input_ids=tcr_ids, attention_mask=attn_mask_tcr,
                            ).last_hidden_state
                        esm_out_tcr_raw = esm_out_tcr_raw.detach()
                    else:
                        esm_out_tcr_raw = self.esm_encoder.esm_model(
                            input_ids=tcr_ids, attention_mask=attn_mask_tcr,
                        ).last_hidden_state

                    esm_node_feats = self._map_esm_to_graph_nodes(
                        esm_out_mhc_raw, esm_out_tcr_raw,
                        mhc_mask, tcr_mask, struct_graph)

                g_emb, node_h, batch_ids = self.struct_encoder(
                    struct_graph, esm_node_features=esm_node_feats)

                h_seq_concat = torch.cat([h_tcr, h_mhc], dim=1)
                if tcr_mask is not None and mhc_mask is not None:
                    seq_mask_concat = torch.cat([tcr_mask, mhc_mask], dim=1)
                else:
                    seq_mask_concat = None

                struct_cross, seq_struct_cross = self.struct_seq_cross(
                    node_h, batch_ids, h_seq_concat, seq_mask_concat
                )
                f_struct = torch.cat([g_emb, struct_cross, seq_struct_cross], dim=-1)

            out["f_struct"] = f_struct

            # Gated fusion
            fused = self.fusion(f_seq, f_struct, struct_available)
            out["fused"] = fused

            logit = self.classifier(fused)

        # ==========================================
        # Common: logit → prob → loss
        # ==========================================
        out["logit"] = logit
        out["prob"] = torch.sigmoid(logit)

        if not compute_loss:
            return out

        clamped_logit = logit.view(-1).clamp(-10.0, 10.0)
        bind_loss = F.binary_cross_entropy_with_logits(
            clamped_logit, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )
        out["loss"] = bind_loss
        out["bind_loss"] = bind_loss
        return out

    # ---- Inference ----
    @torch.no_grad()
    def predict(self, tcr_ids=None, mhc_ids=None, tcr_mask=None, mhc_mask=None,
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
        """Switch between frozen and fine-tuned ESM at any point during training."""
        if self.esm_encoder is None:
            print("[ESM] struct_only mode — no ESM encoder to configure")
            return
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
# 7. Smoke test — all three ablation modes
# ============================================================

if __name__ == "__main__":
    import sys
    import dgl
    from types import SimpleNamespace

    # --- Mock ESM model ---
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

    # --- Dummy data ---
    tcr = torch.randint(1, V_ESM, (B, L_TCR))
    mhc = torch.randint(1, V_ESM, (B, L_MHC))
    tcr_mask = torch.ones(B, L_TCR, dtype=torch.bool)
    mhc_mask = torch.ones(B, L_MHC, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,)).float()

    def make_graph_batch():
        graphs = []
        for _ in range(B):
            n = torch.randint(40, 80, (1,)).item()
            src = torch.randint(0, n, (n * 3,))
            dst = torch.randint(0, n, (n * 3,))
            g = dgl.graph((src, dst))
            g.ndata["x"] = torch.randn(n, 20)
            g.ndata["coords"] = torch.randn(n, 3)
            g.ndata["hbond_acceptors"] = torch.rand(n, 1)
            g.ndata["hbond_donors"] = torch.rand(n, 1)
            g.ndata["sidechain_vector"] = torch.randn(n, 3)
            g.ndata["chain_id"] = torch.randint(0, 4, (n,))
            g.edata["feat"] = torch.randn(n * 3, 7)
            g.edata["etype"] = torch.randint(0, 5, (n * 3,))
            graphs.append(g)
        return dgl.batch(graphs)

    bg = make_graph_batch()
    struct_avail = torch.ones(B, dtype=torch.bool)

    # =========================================================
    # Test 1: seq+struct with embedding node features
    # =========================================================
    mock_esm = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_full = ESMMultimodalBindingModel(
        esm_model=mock_esm, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="seq+struct", node_feat_source="embedding",
    )

    out = model_full(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("=== seq+struct (embedding) — With structure ===")
    print(f"  Loss:     {out['loss']:.4f}")
    print(f"  Probs:    {out['prob'].squeeze(-1).tolist()}")
    print(f"  f_struct: {out['f_struct'].shape}")

    # Also test without struct
    out_ns = model_full(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=None, struct_available=None,
        labels=labels, compute_loss=True,
    )
    assert out_ns['f_struct'] is None
    print(f"  (no struct) Loss: {out_ns['loss']:.4f}")

    tp = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model_full.parameters())
    print(f"  Params: {tp:,} trainable / {tt:,} total")

    # =========================================================
    # Test 2: seq+struct with onehot node features (backward compat)
    # =========================================================
    mock_esm1b = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_oh = ESMMultimodalBindingModel(
        esm_model=mock_esm1b, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="seq+struct", node_feat_source="onehot",
    )
    out_oh = model_oh(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("\n=== seq+struct (onehot) ===")
    print(f"  Loss: {out_oh['loss']:.4f}")

    # =========================================================
    # Test 3: seq+struct with encoder node features
    # =========================================================
    mock_esm_enc = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_enc = ESMMultimodalBindingModel(
        esm_model=mock_esm_enc, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="seq+struct", node_feat_source="encoder",
    )

    # Need chain_pos in graph for encoder mode
    bg2 = make_graph_batch()
    for g_sub in dgl.unbatch(bg2):
        n = g_sub.num_nodes()
        g_sub.ndata["chain_pos"] = torch.arange(n) % 50  # mock positions

    bg2 = dgl.batch(dgl.unbatch(bg2))
    out_enc = model_enc(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg2, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print("\n=== seq+struct (encoder) ===")
    print(f"  Loss: {out_enc['loss']:.4f}")
    print(f"  f_struct: {out_enc['f_struct'].shape}")

    # =========================================================
    # Test 4: seq_only
    # =========================================================
    mock_esm2 = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_seq = ESMMultimodalBindingModel(
        esm_model=mock_esm2, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="seq_only",
    )

    out2 = model_seq(
        tcr, mhc, tcr_mask, mhc_mask,
        labels=labels, compute_loss=True,
    )
    print("\n=== seq_only ===")
    print(f"  Loss:  {out2['loss']:.4f}")
    assert out2['f_struct'] is None
    assert model_seq.struct_encoder is None

    # =========================================================
    # Test 5: struct_only with embedding
    # =========================================================
    mock_esm3 = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_struct = ESMMultimodalBindingModel(
        esm_model=mock_esm3, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="struct_only", node_feat_source="embedding",
    )

    out3 = model_struct(
        struct_graph=bg,
        labels=labels, compute_loss=True,
    )
    print("\n=== struct_only (embedding) ===")
    print(f"  Loss: {out3['loss']:.4f}")
    assert out3['f_seq'] is None
    assert model_struct.esm_encoder is None

    # =========================================================
    # Test 6: struct_only with onehot
    # =========================================================
    mock_esm3b = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_struct_oh = ESMMultimodalBindingModel(
        esm_model=mock_esm3b, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True, d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="struct_only", node_feat_source="onehot",
    )
    out3b = model_struct_oh(struct_graph=bg, labels=labels, compute_loss=True)
    print("\n=== struct_only (onehot) ===")
    print(f"  Loss: {out3b['loss']:.4f}")

    # =========================================================
    # Test 7: struct_only + encoder should RAISE
    # =========================================================
    try:
        mock_esm_bad = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
        ESMMultimodalBindingModel(
            esm_model=mock_esm_bad, esm_hidden_size=ESM_HIDDEN,
            mode="struct_only", node_feat_source="encoder",
            d_model=D_MODEL, egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        )
        print("\n=== struct_only + encoder: SHOULD HAVE RAISED ===")
        sys.exit(1)
    except ValueError as e:
        print(f"\n=== struct_only + encoder correctly raised: {e} ===")

    # =========================================================
    # Test 8: LoRA with seq+struct + embedding
    # =========================================================
    mock_esm4 = MockESMModel(ESM_HIDDEN, n_layers=6, vocab_size=V_ESM)
    model_lora = ESMMultimodalBindingModel(
        esm_model=mock_esm4, esm_hidden_size=ESM_HIDDEN,
        freeze_esm=True,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
        lora_pmhc=True,
        d_model=D_MODEL, n_cross_heads=4,
        edge_feat_size=12,
        egnn_hidden=64, egnn_layers=3, egnn_out_dim=64,
        struct_seq_cross_heads=4, d_fused=D_MODEL, clf_hidden=D_MODEL,
        pos_weight=2.0, mode="seq+struct", node_feat_source="embedding",
    )

    out4 = model_lora(
        tcr, mhc, tcr_mask, mhc_mask,
        struct_graph=bg, struct_available=struct_avail,
        labels=labels, compute_loss=True,
    )
    print(f"\n=== seq+struct LoRA (embedding) ===")
    print(f"  Loss: {out4['loss']:.4f}")

    print("\n=== All smoke tests passed ===")