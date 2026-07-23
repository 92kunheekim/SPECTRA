"""
model_ablation.py — Unified Model for Architecture Ablation
=============================================================

Seven configurations testing what each component contributes:

  A: concat_cls            "MHC.pep.TRA.TRB" → ESM → [CLS] → classifier
  B: 4chain_pool           4 chains → ESM×4 → pool → concat → classifier
  C: 4chain_pool_rosetta   4 chains → ESM×4 → pool → concat + Rosetta → classifier
  D: 4chain_crossattn      4 chains → ESM×4 → 3 cross-attn → classifier
  E: 4chain_crossattn_rosetta   (D) + Rosetta gated fusion  [FULL MODEL]
  F: concat_cls_rosetta    (A) + Rosetta features
  G: 2chain_pool           "MHC.pep" + "TRA.TRB" → ESM×2 → pool → concat → classifier
  H: 2chain_pool_rosetta   (G) + Rosetta features

Controls:
  A→B: positional encoding fix (1 chain vs 4 independent)
  B→D: cross-attention value (pool-only vs interaction-aware)
  A→G→B: 1-chain vs 2-chain vs 4-chain (compute scaling)
  A→F, B→C, D→E, G→H: Rosetta feature value
  F vs E: is cross-attention worth 4× compute over simple + Rosetta?

All models share:
  - Same ESM-2 backbone (shared weights for multi-chain)
  - Same RosettaEncoder (when used)
  - Same ResidualClassifier head
  - Same forward signature (unified collate handles all modes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Constants
# ============================================================

ROSETTA_FEATURE_NAMES = [
    "sc_value", "hbonds_int", "dG_separated_per_dSASA",
    "per_residue_energy_int", "dSASA_int", "dSASA_hphobic", "dSASA_polar",
    "fa_atr", "fa_sol", "fa_elec",
    "fa_rep", "nres_int",
]
N_ROSETTA_FEATURES = len(ROSETTA_FEATURE_NAMES)

ABLATION_MODES = {
    "A": "concat_cls",
    "B": "4chain_pool",
    "C": "4chain_pool_rosetta",
    "D": "4chain_crossattn",
    "E": "4chain_crossattn_rosetta",
    "F": "concat_cls_rosetta",
    "G": "2chain_pool",
    "H": "2chain_pool_rosetta",
}

# Config per mode: (n_esm_passes, use_crossattn, use_rosetta, chain_mode)
MODE_CONFIG = {
    "A": {"esm_passes": 1, "crossattn": False, "rosetta": False, "chain": "1chain"},
    "B": {"esm_passes": 4, "crossattn": False, "rosetta": False, "chain": "4chain"},
    "C": {"esm_passes": 4, "crossattn": False, "rosetta": True,  "chain": "4chain"},
    "D": {"esm_passes": 4, "crossattn": True,  "rosetta": False, "chain": "4chain"},
    "E": {"esm_passes": 4, "crossattn": True,  "rosetta": True,  "chain": "4chain"},
    "F": {"esm_passes": 1, "crossattn": False, "rosetta": True,  "chain": "1chain"},
    "G": {"esm_passes": 2, "crossattn": False, "rosetta": False, "chain": "2chain"},
    "H": {"esm_passes": 2, "crossattn": False, "rosetta": True,  "chain": "2chain"},
}


# ============================================================
# 1. ESM Chain Encoder
# ============================================================

class ESMChainEncoder(nn.Module):
    """Encodes one chain (or concatenated multi-chain) with ESM-2."""

    def __init__(self, esm_model, esm_hidden, d_model, freeze=True, n_tune_layers=0):
        super().__init__()
        self.esm = esm_model
        self.esm_hidden = esm_hidden
        self.d_model = d_model
        self.freeze = freeze

        if freeze:
            for p in self.esm.parameters():
                p.requires_grad = False

        if n_tune_layers > 0 and hasattr(self.esm, 'encoder') and hasattr(self.esm.encoder, 'layer'):
            for layer in self.esm.encoder.layer[-n_tune_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(esm_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, input_ids, attention_mask=None):
        """Returns h [B, L, d_model], pooled [B, d_model]."""
        attn = attention_mask.long() if attention_mask is not None else None
        if self.freeze:
            with torch.no_grad():
                out = self.esm(input_ids=input_ids, attention_mask=attn).last_hidden_state
            out = out.detach()
        else:
            out = self.esm(input_ids=input_ids, attention_mask=attn).last_hidden_state

        h = self.proj(out)

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return h, pooled

    def forward_cls(self, input_ids, attention_mask=None):
        """Returns [CLS] embedding projected to d_model. For mode A/F."""
        attn = attention_mask.long() if attention_mask is not None else None
        if self.freeze:
            with torch.no_grad():
                out = self.esm(input_ids=input_ids, attention_mask=attn).last_hidden_state
            out = out.detach()
        else:
            out = self.esm(input_ids=input_ids, attention_mask=attn).last_hidden_state

        cls_emb = out[:, 0, :]  # [B, esm_hidden]
        return self.proj(cls_emb)  # [B, d_model]


# ============================================================
# 2. Cross-Attention Module
# ============================================================

class CrossAttention(nn.Module):
    """Bidirectional cross-attention, returns pooled reps for both directions."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.a2b = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.a2b_norm = nn.LayerNorm(d_model)
        self.b2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.b2a_norm = nn.LayerNorm(d_model)

    def forward(self, h_a, h_b, mask_a=None, mask_b=None):
        kpm_a = ~mask_a if mask_a is not None else None
        kpm_b = ~mask_b if mask_b is not None else None

        a2b_out, _ = self.a2b(h_a, h_b, h_b, key_padding_mask=kpm_b)
        a2b_out = self.a2b_norm(a2b_out + h_a)
        b2a_out, _ = self.b2a(h_b, h_a, h_a, key_padding_mask=kpm_a)
        b2a_out = self.b2a_norm(b2a_out + h_b)

        def _pool(x, mask):
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(1)

        return _pool(a2b_out, mask_a), _pool(b2a_out, mask_b)


# ============================================================
# 3. Rosetta Feature Encoder
# ============================================================

class RosettaEncoder(nn.Module):
    """3-layer MLP with LayerNorm + residual connections."""

    def __init__(self, n_features=N_ROSETTA_FEATURES, d_out=64, d_hidden=128, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.fc1 = nn.Linear(n_features, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.ln3 = nn.LayerNorm(d_out)
        self.res1 = nn.Linear(n_features, d_hidden)
        self.res2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.input_norm(x)
        h = self.act(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = 0.5 * (h + self.res1(x))
        r = h
        h = self.act(self.ln2(self.fc2(h)))
        h = self.drop(h)
        h = 0.5 * (h + r)
        r = self.res2(h)
        h = self.act(self.ln3(self.fc3(h)))
        h = self.drop(h)
        return 0.5 * (h + r)


# ============================================================
# 4. Gated Fusion (seq + rosetta)
# ============================================================

class GatedFusion(nn.Module):
    """Per-dimension sigmoid gate: fused = λ⊙s + (1-λ)⊙r."""

    def __init__(self, d_seq, d_rosetta, d_fused):
        super().__init__()
        self.proj_seq = nn.Linear(d_seq, d_fused)
        self.proj_rosetta = nn.Linear(d_rosetta, d_fused)
        self.gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.ReLU(),
            nn.Linear(d_fused, d_fused),
            nn.Sigmoid(),
        )

    def forward(self, f_seq, f_rosetta=None, rosetta_available=None):
        s = self.proj_seq(f_seq)
        if f_rosetta is None:
            return s
        r = self.proj_rosetta(f_rosetta)
        lam = self.gate(torch.cat([s, r], dim=-1))
        if rosetta_available is not None:
            no_r = (~rosetta_available).float().unsqueeze(-1)
            lam = lam * (1 - no_r) + no_r
        return lam * s + (1 - lam) * r


# ============================================================
# 5. Residual Classifier Head
# ============================================================

class ResidualClassifier(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.out = nn.Linear(d_hidden, 1)
        self.p = dropout

    def forward(self, x):
        r = self.proj(x)
        h = F.leaky_relu(self.ln1(self.fc1(x)))
        h = F.dropout(h, p=self.p, training=self.training)
        h = 0.5 * (r + h)
        r = h
        h = F.leaky_relu(self.ln2(self.fc2(h)))
        h = F.dropout(h, p=self.p, training=self.training)
        h = 0.5 * (r + h)
        return self.out(h)


# ============================================================
# 6. Unified Ablation Model
# ============================================================

class AblationModel(nn.Module):
    """
    Unified model supporting all ablation configurations A-H.

    The mode string determines:
      - How ESM processes input (1/2/4 chain passes, CLS vs mean pool)
      - Whether cross-attention layers are built/used
      - Whether Rosetta features are included
      - Dimension of the sequence representation

    Dimension summary per mode:
      A: d_model (CLS projected)                         → 128
      B: 4 * d_model (4 pools)                           → 512
      C: 4 * d_model + d_rosetta (gated)                 → 256 (fused)
      D: 10 * d_model (4 pools + 6 cross-attn)           → 1280
      E: 10 * d_model + d_rosetta (gated)                → 256 (fused)
      F: d_model + d_rosetta (gated)                     → 256 (fused)
      G: 2 * d_model (2 pools)                           → 256
      H: 2 * d_model + d_rosetta (gated)                 → 256 (fused)
    """

    def __init__(
        self,
        esm_model: nn.Module,
        esm_hidden: int = 480,
        mode: str = "E",
        freeze_esm: bool = True,
        n_tune_layers: int = 0,
        d_model: int = 128,
        n_cross_heads: int = 4,
        dropout: float = 0.2,
        n_rosetta_features: int = N_ROSETTA_FEATURES,
        d_rosetta: int = 64,
        d_fused: int = 256,
        clf_hidden: int = 256,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        assert mode in MODE_CONFIG, f"mode must be one of {list(MODE_CONFIG.keys())}"
        self.mode = mode
        self.cfg = MODE_CONFIG[mode]
        self.d_model = d_model
        self.register_buffer("pos_weight_buf", torch.tensor([min(pos_weight, 10.0)]))

        # ---- ESM encoder ----
        self.esm_encoder = ESMChainEncoder(
            esm_model, esm_hidden, d_model,
            freeze=freeze_esm, n_tune_layers=n_tune_layers,
        )

        # ---- Cross-attention (modes D, E only) ----
        if self.cfg["crossattn"]:
            self.cross_pep_mhc = CrossAttention(d_model, n_cross_heads, dropout)
            self.cross_tra_trb = CrossAttention(d_model, n_cross_heads, dropout)
            self.cross_tcr_pmhc = CrossAttention(d_model, n_cross_heads, dropout)

        # ---- Compute d_seq ----
        chain = self.cfg["chain"]
        if chain == "1chain":
            d_seq = d_model  # CLS
        elif chain == "2chain":
            d_seq = 2 * d_model  # 2 pools
        elif chain == "4chain" and self.cfg["crossattn"]:
            d_seq = 10 * d_model  # 4 pools + 6 cross
        elif chain == "4chain":
            d_seq = 4 * d_model  # 4 pools only
        self._d_seq = d_seq

        # ---- Rosetta encoder ----
        self._d_rosetta = d_rosetta
        if self.cfg["rosetta"]:
            self.rosetta_encoder = RosettaEncoder(
                n_rosetta_features, d_rosetta, d_hidden=128, dropout=dropout)
            self.fusion = GatedFusion(d_seq, d_rosetta, d_fused)
            clf_in = d_fused
        else:
            self.rosetta_encoder = None
            self.fusion = None
            # For non-rosetta modes, project to d_fused if needed
            if d_seq != d_fused:
                self.seq_proj = nn.Sequential(
                    nn.Linear(d_seq, d_fused), nn.LayerNorm(d_fused), nn.GELU())
                clf_in = d_fused
            else:
                self.seq_proj = None
                clf_in = d_seq

        self._clf_in = clf_in
        self.classifier = ResidualClassifier(clf_in, clf_hidden, dropout)

    def _describe(self):
        """Return human-readable description of this configuration."""
        cfg = self.cfg
        lines = [
            f"Mode {self.mode}: {ABLATION_MODES.get(self.mode, self.mode)}",
            f"  ESM passes: {cfg['esm_passes']}  |  Chain: {cfg['chain']}",
            f"  Cross-attention: {cfg['crossattn']}  |  Rosetta: {cfg['rosetta']}",
            f"  d_seq={self._d_seq}  d_clf_in={self._clf_in}",
        ]
        return "\n".join(lines)

    def forward(
        self,
        # Always provided by unified collate:
        concat_ids=None, concat_mask=None,       # 1-chain: "MHC.pep.TRA.TRB"
        pmhc_ids=None, pmhc_mask=None,           # 2-chain: "MHC.pep"
        tcr_ids=None, tcr_mask=None,             # 2-chain: "TRA.TRB"
        mhc_ids=None, mhc_mask=None,             # 4-chain
        pep_ids=None, pep_mask=None,
        tra_ids=None, tra_mask=None,
        trb_ids=None, trb_mask=None,
        rosetta_features=None,
        rosetta_available=None,
        labels=None,
        compute_loss=False,
    ):
        out = {}
        chain = self.cfg["chain"]

        # ==========================================
        # Sequence representation (mode-dependent)
        # ==========================================

        if chain == "1chain":
            # Mode A/F: single concatenated input → [CLS]
            f_seq = self.esm_encoder.forward_cls(concat_ids, concat_mask)  # [B, d_model]

        elif chain == "2chain":
            # Mode G/H: pMHC and TCR as 2 inputs → mean pool → concat
            _, pool_pmhc = self.esm_encoder(pmhc_ids, pmhc_mask)
            _, pool_tcr = self.esm_encoder(tcr_ids, tcr_mask)
            f_seq = torch.cat([pool_pmhc, pool_tcr], dim=-1)  # [B, 2*d_model]

        elif chain == "4chain":
            # Modes B/C/D/E: 4 independent chains
            h_mhc, pool_mhc = self.esm_encoder(mhc_ids, mhc_mask)
            h_pep, pool_pep = self.esm_encoder(pep_ids, pep_mask)
            h_tra, pool_tra = self.esm_encoder(tra_ids, tra_mask)
            h_trb, pool_trb = self.esm_encoder(trb_ids, trb_mask)

            if self.cfg["crossattn"]:
                # Mode D/E: 3 cross-attention layers
                c_pep_mhc, c_mhc_pep = self.cross_pep_mhc(h_pep, h_mhc, pep_mask, mhc_mask)
                c_tra_trb, c_trb_tra = self.cross_tra_trb(h_tra, h_trb, tra_mask, trb_mask)

                h_tcr = torch.cat([h_tra, h_trb], dim=1)
                h_pmhc = torch.cat([h_mhc, h_pep], dim=1)
                tcr_m = torch.cat([tra_mask, trb_mask], dim=1) if tra_mask is not None else None
                pmhc_m = torch.cat([mhc_mask, pep_mask], dim=1) if mhc_mask is not None else None
                c_tcr_pmhc, c_pmhc_tcr = self.cross_tcr_pmhc(h_tcr, h_pmhc, tcr_m, pmhc_m)

                f_seq = torch.cat([
                    pool_mhc, pool_pep, pool_tra, pool_trb,
                    c_pep_mhc, c_mhc_pep,
                    c_tra_trb, c_trb_tra,
                    c_tcr_pmhc, c_pmhc_tcr,
                ], dim=-1)  # [B, 10*d_model]
            else:
                # Mode B/C: just pool concat
                f_seq = torch.cat([pool_mhc, pool_pep, pool_tra, pool_trb], dim=-1)

        out["f_seq"] = f_seq

        # ==========================================
        # Rosetta branch (mode-dependent)
        # ==========================================
        f_rosetta = None
        if self.cfg["rosetta"] and self.rosetta_encoder is not None and rosetta_features is not None:
            f_rosetta = self.rosetta_encoder(torch.nan_to_num(rosetta_features, nan=0.0))
        out["f_rosetta"] = f_rosetta

        # ==========================================
        # Fusion → classifier
        # ==========================================
        if self.fusion is not None:
            fused = self.fusion(f_seq, f_rosetta, rosetta_available)
        elif self.seq_proj is not None:
            fused = self.seq_proj(f_seq)
        else:
            fused = f_seq

        out["fused"] = fused
        logit = self.classifier(fused)
        out["logit"] = logit
        out["prob"] = torch.sigmoid(logit)

        if not compute_loss:
            return out

        clamped = logit.view(-1).clamp(-10.0, 10.0)
        loss = F.binary_cross_entropy_with_logits(
            clamped, labels.float().view(-1),
            pos_weight=self.pos_weight_buf, reduction="mean",
        )
        out["loss"] = loss
        return out

    def set_esm_tuning(self, freeze=True, n_tune_layers=0):
        enc = self.esm_encoder
        for p in enc.esm.parameters():
            p.requires_grad = False
        enc.freeze = freeze
        if not freeze and hasattr(enc.esm, 'encoder') and hasattr(enc.esm.encoder, 'layer'):
            for layer in enc.esm.encoder.layer[-n_tune_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in self.parameters())
        print(f"[{self.mode}] freeze={freeze}, tune={n_tune_layers}, trainable={tp:,}/{tt:,}")


# ============================================================
# 7. Smoke Test
# ============================================================

if __name__ == "__main__":
    B, V, ESM_H, D = 3, 33, 64, 128

    class MockESM(nn.Module):
        def __init__(self, h, n=6):
            super().__init__()
            self.embeddings = nn.Module()
            self.embeddings.word_embeddings = nn.Embedding(V, h)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([
                nn.TransformerEncoderLayer(h, 4, h*4, batch_first=True) for _ in range(n)])
            self.hidden = h
        def forward(self, input_ids, attention_mask=None):
            x = self.embeddings.word_embeddings(input_ids)
            for l in self.encoder.layer: x = l(x)
            return type('o',(object,),{'last_hidden_state':x})()

    # Inputs
    concat_ids = torch.randint(0, V, (B, 200))
    concat_mask = torch.ones(B, 200, dtype=torch.bool)
    pmhc_ids = torch.randint(0, V, (B, 80))
    pmhc_mask = torch.ones(B, 80, dtype=torch.bool)
    tcr_ids = torch.randint(0, V, (B, 100))
    tcr_mask = torch.ones(B, 100, dtype=torch.bool)
    mhc_ids = torch.randint(0, V, (B, 60))
    mhc_mask = torch.ones(B, 60, dtype=torch.bool)
    pep_ids = torch.randint(0, V, (B, 12))
    pep_mask = torch.ones(B, 12, dtype=torch.bool)
    tra_ids = torch.randint(0, V, (B, 50))
    tra_mask = torch.ones(B, 50, dtype=torch.bool)
    trb_ids = torch.randint(0, V, (B, 50))
    trb_mask = torch.ones(B, 50, dtype=torch.bool)
    rosetta = torch.randn(B, N_ROSETTA_FEATURES)
    rosetta_avail = torch.ones(B, dtype=torch.bool)
    labels = torch.tensor([1, 0, 1], dtype=torch.float)

    print("=" * 70)
    print("  ABLATION MODEL SMOKE TESTS")
    print("=" * 70)

    for mode in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        esm = MockESM(ESM_H, n=4)
        model = AblationModel(
            esm_model=esm, esm_hidden=ESM_H, mode=mode,
            d_model=D, d_fused=256, clf_hidden=256,
            n_cross_heads=4, dropout=0.1, pos_weight=5.0,
        )

        out = model(
            concat_ids=concat_ids, concat_mask=concat_mask,
            pmhc_ids=pmhc_ids, pmhc_mask=pmhc_mask,
            tcr_ids=tcr_ids, tcr_mask=tcr_mask,
            mhc_ids=mhc_ids, mhc_mask=mhc_mask,
            pep_ids=pep_ids, pep_mask=pep_mask,
            tra_ids=tra_ids, tra_mask=tra_mask,
            trb_ids=trb_ids, trb_mask=trb_mask,
            rosetta_features=rosetta, rosetta_available=rosetta_avail,
            labels=labels, compute_loss=True,
        )

        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in model.parameters())
        print(f"\n{model._describe()}")
        print(f"  f_seq: {out['f_seq'].shape}  fused: {out['fused'].shape}")
        print(f"  loss: {out['loss']:.4f}  probs: {[f'{p:.3f}' for p in out['prob'].squeeze(-1).tolist()]}")
        print(f"  params: {tp:,} trainable / {tt:,} total")

    print(f"\n{'='*70}")
    print("  ALL SMOKE TESTS PASSED")
    print(f"{'='*70}")