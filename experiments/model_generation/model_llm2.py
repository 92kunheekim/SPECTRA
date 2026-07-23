"""
ESMMultimodalBindingModel2 — ESM + Pseudo-Heterogeneous Graph Transformer
===========================================================================

Key difference from ESMModel (Model B):
  Replaces the EGNN structure encoder with a STAG-LLM-inspired pseudo-
  heterogeneous graph transformer that maintains SEPARATE message-passing
  weights for each biological edge type:

  Chain mapping:  A=MHC, C=peptide, D=TCR-α, E=TCR-β
  Logical groups: TCR = {D, E},  pMHC = {A, C}

  Edge types (9 interaction classes):
    tcr↔tcr   tcr↔pep   tcr↔mhc
    pep↔tcr   pep↔pep   pep↔mhc
    mhc↔tcr   mhc↔pep   mhc↔mhc

  This mirrors STAG-LLM's `psudo_hetero_transformer` which uses 9 separate
  TransformerConv layers, but adapted for DGL graphs.

Architecture:
  1. ESM-2 encoder (+ optional LoRA) → TCR projection  → h_tcr
  2. ESM-2 encoder (+ optional LoRA) → pMHC projection → h_mhc
  3. Bidirectional cross-attention: h_tcr ↔ h_mhc
  4. Pseudo-heterogeneous graph transformer → g_struct
  5. Structure ↔ Sequence cross-attention
  6. Gated fusion → LayerNorm + residual classifier → binding logit

LoRA (optional):
  Chain-specific low-rank adapters on ESM attention layers.
  TCR gets its own adapter (high diversity); pMHC adapter is optional.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


# ============================================================
# 0. LoRA — Low-Rank Adaptation Modules
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

    Targets Q and V projections in self-attention (standard LoRA recipe).

    NOTE: This version works with the split encoder (esm.encoder, NOT full
    EsmModel). The layer list is accessed via `esm_encoder.layer`.
    """

    def __init__(self, esm_encoder: nn.Module, n_layers: int = 4,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.alpha = alpha

        encoder_layers = list(esm_encoder.layer)
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

    def apply_to(self, esm_encoder: nn.Module):
        """Patch LoRA adapters into the ESM encoder (in-place)."""
        encoder_layers = list(esm_encoder.layer)
        for key, lora_mod in self.lora_modules.items():
            parts = key.split("_")
            layer_idx = int(parts[1])
            target = parts[2]
            attn = encoder_layers[layer_idx].attention.self
            self._original_modules[key] = getattr(attn, "query" if target == "q" else "value")
            setattr(attn, "query" if target == "q" else "value", lora_mod)

    def remove_from(self, esm_encoder: nn.Module):
        """Remove LoRA adapters and restore original modules."""
        encoder_layers = list(esm_encoder.layer)
        for key, orig_mod in self._original_modules.items():
            parts = key.split("_")
            layer_idx = int(parts[1])
            target = parts[2]
            attn = encoder_layers[layer_idx].attention.self
            setattr(attn, "query" if target == "q" else "value", orig_mod)
        self._original_modules.clear()


def global_mean_pool(h, batch_ids):
    """DGL-compatible global mean pooling: [N_total, D] → [B, D]."""
    B = batch_ids.max().item() + 1
    out = torch.zeros(B, h.shape[1], device=h.device, dtype=h.dtype)
    out.index_add_(0, batch_ids, h)
    counts = torch.zeros(B, device=h.device, dtype=h.dtype)
    counts.index_add_(0, batch_ids, torch.ones(h.shape[0], device=h.device, dtype=h.dtype))
    return out / counts.unsqueeze(1).clamp(min=1.0)


# ============================================================
# Edge-type constants (dataset must store these in edata["etype"])
# ============================================================

# Chain ID → group mapping
CHAIN_TO_GROUP = {"A": "mhc", "C": "pep", "D": "tcr", "E": "tcr"}

# All 5 directed edge types (reduced from 9)
# Rationale: collapse tcr↔pep and tcr↔mhc into tcr↔pmhc,
# keep pMHC internals resolved (pep vs mhc are distinct chains)
EDGE_TYPES = [
    "tcr2tcr",    # 0: intra-TCR (TRA + TRB)
    "pep2pep",    # 1: intra-peptide
    "mhc2mhc",    # 2: intra-MHC
    "pep_mhc",    # 3: peptide ↔ MHC groove (bidirectional)
    "tcr_pmhc",   # 4: TCR ↔ pMHC binding interface (bidirectional)
]
ETYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)


def classify_edge(src_chain: str, dst_chain: str) -> int:
    """Return integer edge-type index given source and destination chain IDs."""
    sg = CHAIN_TO_GROUP.get(src_chain, "unk")
    dg = CHAIN_TO_GROUP.get(dst_chain, "unk")
    # Same group → intra-group
    if sg == dg:
        if sg == "tcr": return 0   # tcr2tcr
        if sg == "pep": return 1   # pep2pep
        if sg == "mhc": return 2   # mhc2mhc
    # pep ↔ mhc
    if {sg, dg} == {"pep", "mhc"}: return 3  # pep_mhc
    # tcr ↔ pep or tcr ↔ mhc → tcr_pmhc
    if "tcr" in {sg, dg}: return 4  # tcr_pmhc
    return 0  # fallback


# ============================================================
# 1. ESM Sequence Encoder (identical to Model B)
# ============================================================

class ESMSequenceEncoder(nn.Module):
    """
    Wraps ESM-2 encoder → projected hidden states + CLS pooled vector.

    Supports LoRA adapter patching: the parent model can call
    apply_adapter() / remove_adapter() to swap chain-specific
    LoRA adapters before each forward pass.
    """

    def __init__(self, esm_encoder, esm_embedding, esm_hidden_size=480,
                 d_model=256, freeze_esm=True, n_tune_layers=0, dropout=0.1):
        super().__init__()
        self.esm_embedding = esm_embedding
        self.esm_encoder = esm_encoder
        self.esm_hidden_size = esm_hidden_size
        self.freeze_esm = freeze_esm
        self._active_adapter = None  # tracks which adapter is patched in

        self.proj = nn.Sequential(
            nn.Linear(esm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._configure_freezing(freeze_esm, n_tune_layers)

    def _configure_freezing(self, freeze_esm, n_tune_layers):
        if freeze_esm:
            for p in self.esm_embedding.parameters():
                p.requires_grad = False
            for p in self.esm_encoder.parameters():
                p.requires_grad = False
            if n_tune_layers > 0:
                for layer in list(self.esm_encoder.layer)[-n_tune_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def apply_adapter(self, adapter: "LoRAAdapter"):
        """Patch a LoRA adapter into the ESM encoder."""
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_encoder)
        adapter.apply_to(self.esm_encoder)
        self._active_adapter = adapter

    def remove_adapter(self):
        """Remove the currently active LoRA adapter."""
        if self._active_adapter is not None:
            self._active_adapter.remove_from(self.esm_encoder)
            self._active_adapter = None

    def forward(self, token_ids, mask=None):
        # When LoRA is active, run with gradients for adapter params.
        # When frozen without LoRA, use no_grad for efficiency.
        if self.freeze_esm and self._active_adapter is None:
            with torch.no_grad():
                emb = self.esm_embedding(token_ids)
                esm_out = self.esm_encoder(emb)[0]
            esm_out = esm_out.detach()
        else:
            emb = self.esm_embedding(token_ids)
            esm_out = self.esm_encoder(emb)[0]

        h_seq = self.proj(esm_out)
        h_pool = h_seq[:, 0, :]
        return h_seq, h_pool


# ============================================================
# 2. DGL TransformerConv (single edge-type message passing)
# ============================================================

class DGLTransformerConv(nn.Module):
    """
    Graph Transformer convolution for a single edge type, implemented in DGL.

    Equivalent to PyG's TransformerConv:
      - Multi-head attention: Q from dst, K/V from src
      - Edge features are added to attention scores
      - Output: projected attended values + residual

    Args:
        in_dim:  input node feature dimension
        out_dim: output node feature dimension (per head)
        edge_dim: edge feature dimension
        n_heads: number of attention heads
        dropout: attention dropout
    """

    def __init__(self, in_dim, out_dim, edge_dim=0, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_e = nn.Linear(edge_dim, n_heads) if edge_dim > 0 else None
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, edge_feat=None):
        """
        Args:
            g: DGL (sub)graph for this edge type
            h: [N_total, in_dim] node features (full graph nodes)
            edge_feat: [E, edge_dim] edge features for this edge type

        Returns:
            out: [N_total, out_dim] — only destination nodes are updated,
                 other nodes get zeros (caller handles aggregation)
        """
        N = h.shape[0]
        out_dim = self.n_heads * self.d_k

        with g.local_scope():
            Q = self.W_q(h).view(N, self.n_heads, self.d_k)
            K = self.W_k(h).view(N, self.n_heads, self.d_k)
            V = self.W_v(h).view(N, self.n_heads, self.d_k)

            g.ndata['q'] = Q
            g.ndata['k'] = K
            g.ndata['v'] = V

            # Compute attention scores: q_dst · k_src / sqrt(d_k)
            g.apply_edges(dgl.function.u_dot_v('k', 'q', 'score'))  # [E, n_heads, 1]
            scores = g.edata['score'] / (self.d_k ** 0.5)           # [E, n_heads, 1]

            # Add edge feature bias to attention
            if edge_feat is not None and self.W_e is not None:
                e_bias = self.W_e(edge_feat)  # [E, n_heads]
                scores = scores.squeeze(-1) + e_bias  # [E, n_heads]
                scores = scores.unsqueeze(-1)

            # Edge-wise softmax → attention weights
            try:
                from dgl.nn.functional import edge_softmax
            except ImportError:
                from dgl.ops import edge_softmax
            g.edata['a'] = edge_softmax(g, scores)  # [E, n_heads, 1]
            g.edata['a'] = self.dropout(g.edata['a'])

            # Weighted aggregation of values
            g.ndata['v'] = V
            g.update_all(
                dgl.function.u_mul_e('v', 'a', 'm'),
                dgl.function.sum('m', 'out'),
            )

            out = g.ndata['out'].view(N, out_dim)  # [N, out_dim]
            return self.out_proj(out)


# ============================================================
# 3. Pseudo-Heterogeneous Graph Transformer Layer
# ============================================================

class PseudoHeteroTransformerLayer(nn.Module):
    """
    STAG-LLM-inspired pseudo-heterogeneous transformer layer for DGL.

    Maintains 9 separate DGLTransformerConv modules, one per edge type.
    For each edge type, extracts the relevant subgraph, runs message passing,
    and averages all 9 outputs.

    Includes:
      - Per-edge-type attention-based message passing
      - Residual connection (0.5 * (x + x'))
      - LayerNorm (graph-aware, per batch)
      - Dropout
    """

    def __init__(self, hidden_dim, edge_dim, n_heads=4, dropout=0.125):
        super().__init__()
        self.convs = nn.ModuleDict({
            etype: DGLTransformerConv(hidden_dim, hidden_dim, edge_dim, n_heads, dropout)
            for etype in EDGE_TYPES
        })
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, edge_feats_by_type, batch_ids):
        """
        Args:
            g: full DGL batched graph
            h: [N, hidden_dim] node features
            edge_feats_by_type: dict {etype_str: [E_type, edge_dim]} edge features
            batch_ids: [N] graph membership

        Returns:
            h_out: [N, hidden_dim] updated node features
        """
        etype_ids = g.edata["etype"]  # [E_total] integer edge type labels
        agg = torch.zeros_like(h)
        n_active = 0

        for etype_str, conv in self.convs.items():
            etype_idx = ETYPE_TO_IDX[etype_str]
            mask = (etype_ids == etype_idx)

            if mask.sum() == 0:
                continue

            # Extract subgraph for this edge type
            edge_ids = mask.nonzero(as_tuple=True)[0]
            sub_g = dgl.edge_subgraph(g, edge_ids, relabel_nodes=False)

            ef = edge_feats_by_type.get(etype_str)
            out = conv(sub_g, h, ef)
            agg = agg + out
            n_active += 1

        if n_active > 0:
            agg = agg / n_active

        agg = self.dropout(F.leaky_relu(agg))
        agg = self.norm(agg)
        h_out = 0.5 * (h + agg)  # Residual

        return h_out


# ============================================================
# 4. Edge Feature MLP (shared across edge types, like STAG-LLM)
# ============================================================

class EdgeMLP(nn.Module):
    """
    STAG-LLM-style edge feature MLP.

    Concatenates [src_feat, dst_feat, edge_attr] → hidden → edge_hidden.
    Applied per edge type to produce enriched edge features that encode
    both structural (RBF distances, bond types) and node context.
    """

    def __init__(self, node_dim, edge_in_dim, edge_hidden_dim, dropout=0.125):
        super().__init__()
        self.lin1 = nn.Linear(node_dim * 2 + edge_in_dim, edge_hidden_dim)
        self.lin2 = nn.Linear(edge_hidden_dim, edge_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, src, dst, edge_attr):
        """
        Args:
            h: [N, node_dim] node features
            src: [E, ] source node indices
            dst: [E, ] destination node indices
            edge_attr: [E, edge_in_dim] raw edge features

        Returns:
            [E, edge_hidden_dim] enriched edge features
        """
        x = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


# ============================================================
# 5. Full Pseudo-Heterogeneous Structure Encoder
# ============================================================

class HeteroStructureEncoder(nn.Module):
    """
    Replaces StructureEGNN with a STAG-LLM-inspired pseudo-heterogeneous
    graph transformer stack.

    Pipeline:
      1. Project input node features to hidden_dim
      2. Edge MLP: enrich edge features with node context (per edge type)
      3. N layers of PseudoHeteroTransformerLayer
      4. Global max pool → output projection

    Input: DGL batched graph with:
      - ndata["x"]:      [N, node_feat_size]  one-hot AA
      - ndata["coords"]:  [N, 3]              Cα coordinates
      - edata["feat"]:    [E, edge_feat_size]  bond type encoding
      - edata["etype"]:   [E]                 integer edge type (0-8)

    Output: (g_emb [B, d_out], node_h [N, hidden_dim], batch_ids [N])
    """

    def __init__(
        self,
        node_feat_size: int = 20,
        edge_feat_size: int = 7,
        hidden_dim: int = 320,
        edge_hidden_dim: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        d_out: int = 128,
        dropout: float = 0.125,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project node features
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Edge MLP (STAG-LLM style: concat [src, dst, edge_attr] → edge_hidden)
        self.edge_mlp = EdgeMLP(hidden_dim, edge_feat_size, edge_hidden_dim, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            PseudoHeteroTransformerLayer(hidden_dim, edge_hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Pool + project
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
            g_emb:     [B, d_out]       graph-level embedding
            node_h:    [N_total, hidden_dim]  node-level embeddings
            batch_ids: [N_total]        graph membership
        """
        h = self.node_proj(bg.ndata["x"])           # [N, hidden_dim]
        raw_edge_feat = bg.edata["feat"]             # [E, edge_feat_size]
        etype_ids = bg.edata["etype"]                # [E] integer
        src, dst = bg.edges()

        # --- Edge MLP: enrich edge features per edge type ---
        enriched_edge_feat = self.edge_mlp(h, src, dst, raw_edge_feat)  # [E, edge_hidden]

        # Split edge features by type for the transformer layers
        edge_feats_by_type = {}
        for etype_str in EDGE_TYPES:
            etype_idx = ETYPE_TO_IDX[etype_str]
            mask = (etype_ids == etype_idx)
            if mask.sum() > 0:
                edge_feats_by_type[etype_str] = enriched_edge_feat[mask]

        # --- Transformer layers ---
        batch_ids = self._make_batch_ids(bg)
        for layer in self.layers:
            h = layer(bg, h, edge_feats_by_type, batch_ids)

        # --- Global max pool (STAG-LLM uses max pool) ---
        g_emb = global_mean_pool(h, batch_ids)  # can switch to max_pool
        g_emb = self.out_proj(g_emb)

        return g_emb, h, batch_ids


# ============================================================
# 6. Cross-attention (reused from Model B)
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
        if ctx_mask is not None:
            scores = scores.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(q_seq.shape)
        return self.norm(self.out_proj(out) + q_seq)

    def forward(self, h_a, h_b, mask_a=None, mask_b=None):
        a_att = self._attend(h_a, h_b, mask_b)
        b_att = self._attend(h_b, h_a, mask_a)

        def _pool(h, mask):
            if mask is None: return h.mean(dim=1)
            m = mask.float().unsqueeze(-1)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        return _pool(a_att, mask_a), _pool(b_att, mask_b)


# ============================================================
# 7. Structure ↔ Sequence Cross-attention (from Model B)
# ============================================================

class StructureSequenceCrossAttention(nn.Module):
    """Structure graph nodes attend to sequence hidden states."""

    def __init__(self, d_seq, d_struct, d_out, n_heads=4, dropout=0.1):
        super().__init__()
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
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(q.shape)
        return self.norm(self.out_proj(out) + q)

    def forward(self, node_h, batch_ids, h_seq_concat, seq_mask_concat):
        node_h = self.proj_struct(node_h)
        h_seq = self.proj_seq(h_seq_concat)
        B = batch_ids.max().item() + 1
        pooled = []
        for i in range(B):
            node_mask = (batch_ids == i)
            q = node_h[node_mask].unsqueeze(0)
            kv = h_seq[i:i+1]
            kv_mask = seq_mask_concat[i:i+1] if seq_mask_concat is not None else None
            out = self._single_attend(q, kv, kv_mask)
            pooled.append(out.mean(dim=1))
        return torch.cat(pooled, dim=0)


# ============================================================
# 8. Gated Fusion (from Model B)
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self, d_seq, d_struct, d_fused):
        super().__init__()
        self.proj_seq = nn.Linear(d_seq, d_fused)
        self.proj_struct = nn.Linear(d_struct, d_fused)
        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused), nn.ReLU(),
            nn.Linear(d_fused, 1), nn.Sigmoid(),
        )

    def forward(self, f_seq, f_struct=None, struct_available=None):
        s = self.proj_seq(f_seq)
        if f_struct is None: return s
        g = self.proj_struct(f_struct)
        lam = self.gate(torch.cat([s, g], dim=-1))
        if struct_available is not None:
            mask = (~struct_available).float().unsqueeze(-1)
            lam = lam * (1 - mask) + mask
        return lam * s + (1 - lam) * g


# ============================================================
# 9. Full ESMMultimodalBindingModel2
# ============================================================

class ESMMultimodalBindingModel2(nn.Module):
    """
    ESM + pseudo-heterogeneous graph transformer for TCR-pMHC binding.

    Key improvements over ESMMultimodalBindingModel (Model B):
      - Structure encoder uses edge-type-aware message passing (STAG-LLM style)
        with 9 separate TransformerConv per layer instead of type-agnostic EGNN
      - Edge MLP enriches edge features with node context before convolution
      - LayerNorm everywhere (no BatchNorm) for bf16 stability
      - Logit clamping for numerical safety
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
        norm_type: str = "layernorm",  # "layernorm" or "batchnorm"
        # Loss
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout
        self.use_lora = use_lora
        self.lora_pmhc = lora_pmhc
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ===== Sequence branch =====
        # Shared ESM backbone — single copy. Specialization via:
        #   (a) chain-specific LoRA adapters (patched in/out before each forward)
        #   (b) separate projection heads for TCR vs pMHC
        self.esm_seq_encoder = ESMSequenceEncoder(
            esm_encoder=esm_encoder, esm_embedding=esm_embedding,
            esm_hidden_size=esm_hidden_size, d_model=d_model,
            freeze_esm=freeze_esm, n_tune_layers=n_tune_layers, dropout=dropout,
        )

        # Separate projection head for pMHC (TCR uses the default self.esm_seq_encoder.proj)
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
            if lora_pmhc:
                self.pmhc_lora = LoRAAdapter(
                    esm_encoder, n_layers=lora_n_layers,
                    rank=lora_rank, alpha=lora_alpha,
                )
            else:
                self.pmhc_lora = None
        else:
            self.tcr_lora = None
            self.pmhc_lora = None

        self.seq_cross_attn = CrossAttention(d_model, n_cross_heads, dropout)
        d_seq_feat = 4 * d_model

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
            return nn.LayerNorm(dim)  # default

        self.clf_proj = nn.Linear(d_fused, clf_hidden)
        self.clf_fc1 = nn.Linear(d_fused, clf_hidden)
        self.clf_norm1 = _make_norm(clf_hidden)
        self.clf_fc2 = nn.Linear(clf_hidden, clf_hidden)
        self.clf_norm2 = _make_norm(clf_hidden)
        self.clf_out = nn.Linear(clf_hidden, 1)

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

    def forward(
        self,
        tcr_ids, mhc_ids,
        tcr_mask=None, mhc_mask=None,
        struct_graph=None, struct_available=None,
        labels=None, compute_loss=False,
    ):
        # 1. Sequence encoding via ESM
        # TCR encoding — with LoRA adapter if enabled
        if self.tcr_lora is not None:
            self.esm_seq_encoder.apply_adapter(self.tcr_lora)
        h_tcr, pool_tcr = self.esm_seq_encoder(tcr_ids, tcr_mask)

        # pMHC encoding — swap adapter, use separate projection
        if self.tcr_lora is not None:
            self.esm_seq_encoder.remove_adapter()
        if self.pmhc_lora is not None:
            self.esm_seq_encoder.apply_adapter(self.pmhc_lora)

        # Run ESM backbone for pMHC, but use the separate pMHC projection
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

        # Clean up adapter state
        if self.pmhc_lora is not None:
            self.esm_seq_encoder.remove_adapter()

        # 2. Sequence cross-attention
        cross_tcr, cross_mhc = self.seq_cross_attn(h_tcr, h_mhc, tcr_mask, mhc_mask)
        f_seq = torch.cat([pool_tcr, pool_mhc, cross_tcr, cross_mhc], dim=-1)

        # 3. Structure encoding
        f_struct = None
        if struct_graph is not None:
            g_emb, node_h, batch_ids = self.struct_encoder(struct_graph)

            h_seq_concat = torch.cat([h_tcr, h_mhc], dim=1)
            seq_mask_concat = (
                torch.cat([tcr_mask, mhc_mask], dim=1)
                if tcr_mask is not None and mhc_mask is not None else None
            )
            struct_cross = self.struct_seq_cross(node_h, batch_ids, h_seq_concat, seq_mask_concat)
            f_struct = torch.cat([g_emb, struct_cross], dim=-1)

        # 4. Gated fusion
        fused = self.fusion(f_seq, f_struct, struct_available)

        # 5. Classify
        logit = self._classify(fused)

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "pool_tcr": pool_tcr, "pool_mhc": pool_mhc,
            "f_seq": f_seq, "f_struct": f_struct, "fused": fused,
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
        print(f"[ESM] {mode}{lora_info}")


# ============================================================
# 9b. BatchNorm variant — ESMMultimodalBindingModel2BN
# ============================================================

class ESMMultimodalBindingModel2BN(ESMMultimodalBindingModel2):
    """
    Identical to ESMMultimodalBindingModel2 but uses BatchNorm1d in the
    classifier head instead of LayerNorm.

    Use this for A/B comparison:
      - ESMMultimodalBindingModel2    → LayerNorm  (--model_type esm2)
      - ESMMultimodalBindingModel2BN  → BatchNorm  (--model_type esm2_bn)

    BatchNorm can provide stronger per-feature normalization across the batch
    and implicit regularization via running statistics, but:
      - Needs batch size ≥ 16 to get stable statistics
      - Running mean/var can overflow in fp16/bf16 (use fp32)
      - Behaviour differs between train/eval modes
    """

    def __init__(self, **kwargs):
        kwargs["norm_type"] = "batchnorm"
        super().__init__(**kwargs)


# ============================================================
# 10. Smoke test
# ============================================================

if __name__ == "__main__":
    import random

    # --- Mock ESM components that mimic HuggingFace ESM split architecture ---
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

    class MockEnc(nn.Module):
        """Mimics esm.esm.encoder with .layer attribute."""
        def __init__(self, h=320, n=6):
            super().__init__()
            self.layer = nn.ModuleList([
                MockTransformerLayer(h) for _ in range(n)
            ])
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

    # --- Dummy graph data ---
    tcr = torch.randint(1, 33, (B, L_T))
    mhc = torch.randint(1, 33, (B, L_M))
    tcr_mask = torch.ones(B, L_T, dtype=torch.bool)
    mhc_mask = torch.ones(B, L_M, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,)).float()

    graphs = []
    for _ in range(B):
        n = random.randint(40, 80)
        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        g = dgl.graph((src, dst))
        g.ndata["x"] = torch.randn(n, 20)
        g.ndata["coords"] = torch.randn(n, 3)
        g.edata["feat"] = torch.randn(g.num_edges(), 7)
        g.edata["etype"] = torch.randint(0, NUM_EDGE_TYPES, (g.num_edges(),))
        graphs.append(g)
    bg = dgl.batch(graphs)
    struct_avail = torch.ones(B, dtype=torch.bool)

    # --- Test 1: No LoRA (backward compatible) ---
    model = ESMMultimodalBindingModel2(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, struct_hidden_dim=H,
        struct_out_dim=128, d_fused=D, clf_hidden=D, pos_weight=2.0,
    )
    out = model(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"=== No LoRA ===")
    print(f"Loss: {out['loss']:.4f}, Probs: {out['prob'].squeeze(-1).tolist()}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")

    # --- Test 2: TCR-only LoRA ---
    model_lora = ESMMultimodalBindingModel2(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, struct_hidden_dim=H,
        struct_out_dim=128, d_fused=D, clf_hidden=D, pos_weight=2.0,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
        lora_pmhc=False,
    )
    out2 = model_lora(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                      struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"\n=== TCR-only LoRA ===")
    print(f"Loss: {out2['loss']:.4f}, Probs: {out2['prob'].squeeze(-1).tolist()}")
    model_lora.set_esm_tuning(freeze=True, n_tune_layers=0)
    lora_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in model_lora.parameters())
    tcr_lora_params = sum(p.numel() for p in model_lora.tcr_lora.parameters())
    print(f"TCR LoRA params: {tcr_lora_params:,}")
    print(f"Trainable: {lora_trainable:,} / {lora_total:,}")

    # --- Test 3: TCR + pMHC LoRA ---
    model_both = ESMMultimodalBindingModel2(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, struct_hidden_dim=H,
        struct_out_dim=128, d_fused=D, clf_hidden=D, pos_weight=2.0,
        use_lora=True, lora_rank=8, lora_alpha=16.0, lora_n_layers=4,
        lora_pmhc=True,
    )
    out3 = model_both(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                      struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"\n=== TCR+pMHC LoRA ===")
    print(f"Loss: {out3['loss']:.4f}, Probs: {out3['prob'].squeeze(-1).tolist()}")
    model_both.set_esm_tuning(freeze=True, n_tune_layers=0)
    both_trainable = sum(p.numel() for p in model_both.parameters() if p.requires_grad)
    both_total = sum(p.numel() for p in model_both.parameters())
    print(f"Trainable: {both_trainable:,} / {both_total:,}")

    # --- Test 4: BatchNorm classifier head ---
    model_bn = ESMMultimodalBindingModel2(
        esm_encoder=MockEnc(H), esm_embedding=MockEmb(33, H),
        esm_hidden_size=H, d_model=D, struct_hidden_dim=H,
        struct_out_dim=128, d_fused=D, clf_hidden=D, pos_weight=2.0,
        norm_type="batchnorm",
    )
    out4 = model_bn(tcr, mhc, tcr_mask, mhc_mask, struct_graph=bg,
                    struct_available=struct_avail, labels=labels, compute_loss=True)
    print(f"\n=== BatchNorm classifier ===")
    print(f"Loss: {out4['loss']:.4f}, Probs: {out4['prob'].squeeze(-1).tolist()}")
    print(f"norm_type: {model_bn.norm_type}")
    print(f"clf_norm1: {model_bn.clf_norm1}")
    print(f"clf_norm2: {model_bn.clf_norm2}")