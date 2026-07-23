"""
dataset_ablation.py — CSV-Based Dataset for Architecture Ablation
==================================================================

Reads all data directly from a single CSV file. No PDB files, no graph
construction, no graphein dependency. Designed for the ESM + Rosetta
ablation models that don't use a graph module.

Expected CSV columns:
  REQUIRED:
    tcr_id          : str, unique sample identifier
    tra_seq         : str, TCR alpha chain amino acid sequence
    trb_seq         : str, TCR beta chain amino acid sequence
    peptide         : str, peptide amino acid sequence (8-11mer)
    mhc_seq         : str, MHC class I heavy chain amino acid sequence
    binding_label   : int, 0 or 1

  ROSETTA FEATURES (optional, all float):
    feat_sc_value
    feat_hbonds_int
    feat_dG_separated_per_dSASA
    feat_per_residue_energy_int
    feat_dSASA_int
    feat_dSASA_hphobic
    feat_dSASA_polar
    feat_fa_atr
    feat_fa_sol
    feat_fa_elec
    feat_fa_rep
    feat_nres_int

  OPTIONAL:
    domain          : int, 0=in-silico, 1=crystal (for sample weighting)

Output per sample (tuple):
    (mhc_str, pep_str, tra_str, trb_str,
     label, rosetta_features, has_rosetta, sample_weight)

Collate function tokenizes all chain formats needed by the ablation models:
  - 1-chain: "MHC.pep.TRA.TRB" (for modes A, F)
  - 2-chain: "MHC.pep" + "TRA.TRB" (for modes G, H)
  - 4-chain: MHC, pep, TRA, TRB independently (for modes B, C, D, E)

Usage:
    from dataset_ablation import AblationDataset, make_ablation_collate_fn
    
    ds = AblationDataset("data.csv", crystal_weight=5.0)
    collate_fn = make_ablation_collate_fn(tokenizer)
    loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
ROSETTA_FEAT_COLS = [f"feat_{name}" for name in ROSETTA_FEATURE_NAMES]

# Required CSV columns
REQUIRED_COLS = ["tcr_id", "tra_seq", "trb_seq", "peptide", "mhc_seq", "binding_label"]

# Chain separator for concatenated inputs
SEP = "."


# ============================================================
# 1. Dataset
# ============================================================

class AblationDataset(Dataset):
    """
    Simple CSV-based dataset for TCR-pMHC binding prediction.
    No PDB files, no graph construction.

    Args:
        csv_path_or_df:  Path to CSV file, or a pandas DataFrame directly.
        crystal_weight:  Loss weight multiplier for crystal structures (domain=1).
        min_seq_len:     Drop samples where any chain is shorter than this.
        max_seq_len:     Truncate any chain longer than this (applied at getitem).
        verbose:         Print dataset summary.
    """

    def __init__(
        self,
        csv_path_or_df,
        crystal_weight: float = 5.0,
        min_seq_len: int = 1,
        max_seq_len: int = 512,
        verbose: bool = True,
    ):
        # ---- Load data ----
        if isinstance(csv_path_or_df, (str, os.PathLike)):
            df = pd.read_csv(csv_path_or_df)
        elif isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df.copy()
        else:
            raise TypeError(f"Expected str/Path/DataFrame, got {type(csv_path_or_df)}")

        # ---- Validate columns ----
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # ---- Clean data ----
        # Drop rows with missing required fields
        df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

        # Ensure sequences are strings
        for col in ["tra_seq", "trb_seq", "peptide", "mhc_seq"]:
            df[col] = df[col].astype(str).str.strip().str.upper()

        # Drop sequences shorter than minimum
        if min_seq_len > 0:
            len_mask = (
                (df["tra_seq"].str.len() >= min_seq_len) &
                (df["trb_seq"].str.len() >= min_seq_len) &
                (df["peptide"].str.len() >= min_seq_len) &
                (df["mhc_seq"].str.len() >= min_seq_len)
            )
            n_before = len(df)
            df = df[len_mask].reset_index(drop=True)
            if verbose and n_before != len(df):
                print(f"  Dropped {n_before - len(df)} samples with seq < {min_seq_len} aa")

        self.max_seq_len = max_seq_len

        # ---- Extract arrays for fast __getitem__ ----
        self._tcr_ids = df["tcr_id"].values
        self._tra_seqs = df["tra_seq"].values
        self._trb_seqs = df["trb_seq"].values
        self._pep_seqs = df["peptide"].values
        self._mhc_seqs = df["mhc_seq"].values
        self._labels = df["binding_label"].values.astype(np.int64)

        # ---- Rosetta features ----
        has_all = all(c in df.columns for c in ROSETTA_FEAT_COLS)
        if has_all:
            self._rosetta = df[ROSETTA_FEAT_COLS].values.astype(np.float32)
            self._has_rosetta = ~np.isnan(self._rosetta).any(axis=1)
            # Replace NaN with 0 for tensor compatibility
            self._rosetta = np.nan_to_num(self._rosetta, nan=0.0)
        else:
            self._rosetta = np.zeros((len(df), N_ROSETTA_FEATURES), dtype=np.float32)
            self._has_rosetta = np.zeros(len(df), dtype=bool)

        # ---- Domain / sample weights ----
        if "domain" in df.columns:
            self._domains = df["domain"].values.astype(np.int64)
        else:
            self._domains = np.zeros(len(df), dtype=np.int64)

        self._weights = np.ones(len(df), dtype=np.float32)
        self._weights[self._domains == 1] = crystal_weight

        # ---- Summary ----
        self._n = len(df)
        if verbose:
            n_bind = (self._labels == 1).sum()
            n_nonbind = (self._labels == 0).sum()
            n_crystal = (self._domains == 1).sum()
            n_rosetta = self._has_rosetta.sum()

            seq_lens = {
                "mhc": df["mhc_seq"].str.len(),
                "pep": df["peptide"].str.len(),
                "tra": df["tra_seq"].str.len(),
                "trb": df["trb_seq"].str.len(),
            }

            print(f"  AblationDataset: {self._n} samples")
            print(f"    Binders: {n_bind} ({100*n_bind/self._n:.1f}%) | "
                  f"Non-binders: {n_nonbind} ({100*n_nonbind/self._n:.1f}%)")
            print(f"    Crystal: {n_crystal} (weight={crystal_weight}x) | "
                  f"Rosetta features: {n_rosetta}")
            print(f"    Seq lengths (median): "
                  f"MHC={seq_lens['mhc'].median():.0f}, "
                  f"pep={seq_lens['pep'].median():.0f}, "
                  f"TRA={seq_lens['tra'].median():.0f}, "
                  f"TRB={seq_lens['trb'].median():.0f}")

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        # Truncate sequences if longer than max
        mhc = self._mhc_seqs[idx][:self.max_seq_len]
        pep = self._pep_seqs[idx][:self.max_seq_len]
        tra = self._tra_seqs[idx][:self.max_seq_len]
        trb = self._trb_seqs[idx][:self.max_seq_len]

        return (
            mhc,                                                          # str
            pep,                                                          # str
            tra,                                                          # str
            trb,                                                          # str
            torch.tensor(self._labels[idx], dtype=torch.long),            # [1]
            torch.tensor(self._rosetta[idx], dtype=torch.float32),        # [12]
            torch.tensor(self._has_rosetta[idx], dtype=torch.bool),       # [1]
            torch.tensor(self._weights[idx], dtype=torch.float32),        # [1]
        )

    # ---- Utility methods ----

    @property
    def pos_weight(self):
        """Compute BCE pos_weight for class imbalance. Capped at 10."""
        n_pos = (self._labels == 1).sum()
        n_neg = (self._labels == 0).sum()
        return min(n_neg / max(n_pos, 1), 10.0)

    @property
    def label_counts(self):
        return {"binder": int((self._labels == 1).sum()),
                "nonbinder": int((self._labels == 0).sum())}

    def split(self, test_size=0.15, seed=42, stratify=True):
        """
        Return two AblationDatasets: (train, val), stratified by binding_label.
        
        Args:
            test_size: fraction for validation
            seed: random seed
            stratify: stratify by binding_label (recommended)
            
        Returns:
            (train_dataset, val_dataset)
        """
        from sklearn.model_selection import train_test_split

        indices = np.arange(self._n)
        strat = self._labels if stratify else None
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=strat)

        train_ds = _SubsetDataset(self, train_idx)
        val_ds = _SubsetDataset(self, val_idx)
        return train_ds, val_ds


class _SubsetDataset(Dataset):
    """Lightweight index-based subset. Avoids copying data."""

    def __init__(self, parent: AblationDataset, indices: np.ndarray):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]

    @property
    def pos_weight(self):
        labels = self.parent._labels[self.indices]
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        return min(n_neg / max(n_pos, 1), 10.0)


# ============================================================
# 2. Collate Function — Tokenizes all chain formats
# ============================================================

def make_ablation_collate_fn(
    tokenizer,
    max_concat: int = 512,
    max_pmhc: int = 230,
    max_tcr: int = 400,
    max_mhc: int = 200,
    max_pep: int = 30,
    max_tra: int = 200,
    max_trb: int = 200,
):
    """
    Creates a collate function that tokenizes inputs for ALL ablation modes.
    Each model reads only the fields it needs based on its chain_mode.

    Tokenization strategy:
      - Space-separated amino acids: "M E T H I O N I N E"
      - ESM tokenizer handles BOS/EOS automatically
      - Padding to batch max length

    Returns dict with keys for all three chain formats:
      1-chain: concat_ids, concat_mask
      2-chain: pmhc_ids, pmhc_mask, tcr_ids, tcr_mask
      4-chain: mhc_ids, mhc_mask, pep_ids, pep_mask, tra_ids, tra_mask, trb_ids, trb_mask
      + labels, rosetta, has_rosetta, weights
    """

    def _tok(seqs: list, max_len: int):
        """Tokenize a list of amino acid strings."""
        # Space-separate for ESM tokenizer
        clean = [" ".join(list(s.strip())) if s.strip() else "<unk>" for s in seqs]
        t = tokenizer(
            clean, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        return t["input_ids"], t["attention_mask"].bool()

    def collate_fn(batch):
        mhc_s, pep_s, tra_s, trb_s, labels, rosetta, has_r, weights = zip(*batch)

        # ---- 1-chain: concatenated with separator ----
        concat_strs = [
            f"{m}{SEP}{p}{SEP}{a}{SEP}{b}"
            for m, p, a, b in zip(mhc_s, pep_s, tra_s, trb_s)
        ]
        concat_ids, concat_mask = _tok(concat_strs, max_concat)

        # ---- 2-chain: pMHC and TCR ----
        pmhc_strs = [f"{m}{SEP}{p}" for m, p in zip(mhc_s, pep_s)]
        tcr_strs = [f"{a}{SEP}{b}" for a, b in zip(tra_s, trb_s)]
        pmhc_ids, pmhc_mask = _tok(pmhc_strs, max_pmhc)
        tcr_ids, tcr_mask = _tok(tcr_strs, max_tcr)

        # ---- 4-chain: independent ----
        mhc_ids, mhc_mask = _tok(mhc_s, max_mhc)
        pep_ids, pep_mask = _tok(pep_s, max_pep)
        tra_ids, tra_mask = _tok(tra_s, max_tra)
        trb_ids, trb_mask = _tok(trb_s, max_trb)

        return {
            # 1-chain (modes A, F)
            "concat_ids": concat_ids,
            "concat_mask": concat_mask,
            # 2-chain (modes G, H)
            "pmhc_ids": pmhc_ids,
            "pmhc_mask": pmhc_mask,
            "tcr_ids": tcr_ids,
            "tcr_mask": tcr_mask,
            # 4-chain (modes B, C, D, E)
            "mhc_ids": mhc_ids,
            "mhc_mask": mhc_mask,
            "pep_ids": pep_ids,
            "pep_mask": pep_mask,
            "tra_ids": tra_ids,
            "tra_mask": tra_mask,
            "trb_ids": trb_ids,
            "trb_mask": trb_mask,
            # Targets and metadata
            "labels": torch.stack(labels),
            "rosetta": torch.stack(rosetta),
            "has_rosetta": torch.stack(has_r),
            "weights": torch.stack(weights),
        }

    return collate_fn


# ============================================================
# 3. Convenience: create train/val datasets from one CSV
# ============================================================

def load_datasets(
    data_csv: str,
    val_split: float = 0.15,
    crystal_weight: float = 5.0,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Load and split data into train/val AblationDatasets.

    Args:
        data_csv:        Path to training CSV. Crystal structures should be
                         included as rows with domain=1 if upweighting desired.
        val_split:       Fraction for validation.
        crystal_weight:  Loss weight for crystal samples (domain=1).
        seed:            Random seed for reproducible splits.

    Returns:
        (train_dataset, val_dataset, pos_weight)
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_csv)
    if "domain" not in df.columns:
        df["domain"] = 0

    # Stratified split
    df_train, df_val = train_test_split(
        df, test_size=val_split, random_state=seed,
        stratify=df["binding_label"],
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    if verbose:
        print(f"\n  Data split: train={len(df_train)}, val={len(df_val)}")

    train_ds = AblationDataset(df_train, crystal_weight=crystal_weight, verbose=verbose)
    val_ds = AblationDataset(df_val, crystal_weight=1.0, verbose=verbose)

    return train_ds, val_ds, train_ds.pos_weight


# ============================================================
# 4. Smoke Test
# ============================================================

if __name__ == "__main__":
    import tempfile

    # ---- Create a synthetic CSV ----
    np.random.seed(42)
    n = 100
    aa = "ACDEFGHIKLMNPQRSTVWY"

    def _rand_seq(length):
        return "".join(np.random.choice(list(aa), size=length))

    rows = []
    for i in range(n):
        label = 1 if i < 20 else 0
        row = {
            "tcr_id": f"sample_{i:04d}",
            "tra_seq": _rand_seq(np.random.randint(90, 130)),
            "trb_seq": _rand_seq(np.random.randint(100, 140)),
            "peptide": _rand_seq(np.random.randint(8, 12)),
            "mhc_seq": _rand_seq(np.random.randint(170, 185)),
            "binding_label": label,
            "domain": 1 if i < 5 else 0,
        }
        # Add Rosetta features for most samples
        if i < 80:
            for feat in ROSETTA_FEAT_COLS:
                row[feat] = np.random.randn()
        else:
            for feat in ROSETTA_FEAT_COLS:
                row[feat] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        csv_path = f.name
        df.to_csv(f, index=False)

    print("=" * 65)
    print("  DATASET SMOKE TEST")
    print("=" * 65)

    # ---- Test 1: Load dataset ----
    print("\n--- Test 1: Load from CSV ---")
    ds = AblationDataset(csv_path, crystal_weight=5.0)
    assert len(ds) == n
    print(f"  Length: {len(ds)}")
    print(f"  pos_weight: {ds.pos_weight:.2f}")

    # ---- Test 2: __getitem__ ----
    print("\n--- Test 2: __getitem__ ---")
    sample = ds[0]
    mhc, pep, tra, trb, label, rosetta, has_r, weight = sample
    print(f"  MHC: {mhc[:30]}... ({len(mhc)} aa)")
    print(f"  Peptide: {pep} ({len(pep)} aa)")
    print(f"  TRA: {tra[:30]}... ({len(tra)} aa)")
    print(f"  TRB: {trb[:30]}... ({len(trb)} aa)")
    print(f"  Label: {label.item()}")
    print(f"  Rosetta: {rosetta.shape}, has_rosetta: {has_r.item()}")
    print(f"  Weight: {weight.item()}")

    # ---- Test 3: Sample without Rosetta ----
    print("\n--- Test 3: Missing Rosetta ---")
    sample_no_r = ds[90]
    assert not sample_no_r[6].item()  # has_rosetta = False
    assert sample_no_r[5].sum().item() == 0.0  # features are 0
    print("  Correctly handles missing Rosetta features")

    # ---- Test 4: Collate with mock tokenizer ----
    print("\n--- Test 4: Collate function ---")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    except Exception:
        # Fallback: create minimal tokenizer-like object for testing
        print("  (ESM tokenizer not available, skipping collate test)")
        tokenizer = None

    if tokenizer is not None:
        collate_fn = make_ablation_collate_fn(tokenizer)
        batch_data = [ds[i] for i in range(4)]
        batch = collate_fn(batch_data)

        print(f"  Batch keys: {sorted(batch.keys())}")
        print(f"  concat_ids: {batch['concat_ids'].shape}")
        print(f"  pmhc_ids:   {batch['pmhc_ids'].shape}")
        print(f"  tcr_ids:    {batch['tcr_ids'].shape}")
        print(f"  mhc_ids:    {batch['mhc_ids'].shape}")
        print(f"  pep_ids:    {batch['pep_ids'].shape}")
        print(f"  tra_ids:    {batch['tra_ids'].shape}")
        print(f"  trb_ids:    {batch['trb_ids'].shape}")
        print(f"  labels:     {batch['labels'].shape} = {batch['labels'].tolist()}")
        print(f"  rosetta:    {batch['rosetta'].shape}")
        print(f"  has_rosetta:{batch['has_rosetta'].shape} = {batch['has_rosetta'].tolist()}")
        print(f"  weights:    {batch['weights'].tolist()}")

        # Verify shapes
        B = 4
        assert batch["labels"].shape == (B,)
        assert batch["rosetta"].shape == (B, N_ROSETTA_FEATURES)
        assert batch["concat_ids"].shape[0] == B
        assert batch["pmhc_ids"].shape[0] == B
        assert batch["mhc_ids"].shape[0] == B
        print("  All shape assertions passed")

    # ---- Test 5: DataLoader integration ----
    print("\n--- Test 5: DataLoader ---")
    if tokenizer is not None:
        loader = DataLoader(ds, batch_size=8, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
        batch = next(iter(loader))
        print(f"  DataLoader batch: {batch['labels'].shape[0]} samples")
        print(f"  concat_ids: {batch['concat_ids'].shape}")
        print("  DataLoader works correctly")

    # ---- Test 6: split() method ----
    print("\n--- Test 6: Train/val split ---")
    train_ds, val_ds = ds.split(test_size=0.2, seed=42)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    assert len(train_ds) + len(val_ds) == len(ds)
    sample_t = train_ds[0]
    assert len(sample_t) == 8
    print("  Split works correctly")

    # ---- Test 7: load_datasets convenience ----
    print("\n--- Test 7: load_datasets ---")
    train_ds2, val_ds2, pw = load_datasets(csv_path, val_split=0.2, seed=42)
    print(f"  Train: {len(train_ds2)}, Val: {len(val_ds2)}, pos_weight: {pw:.2f}")

    # ---- Test 8: Load from DataFrame ----
    print("\n--- Test 8: Load from DataFrame ---")
    ds_from_df = AblationDataset(df, crystal_weight=3.0)
    assert len(ds_from_df) == n
    print(f"  Created from DataFrame: {len(ds_from_df)} samples")

    # Cleanup
    os.unlink(csv_path)

    print(f"\n{'='*65}")
    print("  ALL SMOKE TESTS PASSED")
    print(f"{'='*65}")

# ============================================================
# 4. Leak-free split loader (train / val / held-out TEST)
# ============================================================

def load_datasets_from_split(data_csv, split_json, fold=0, crystal_weight=5.0, verbose=True):
    """Build train/val/TEST AblationDatasets from a leak-free split JSON.

    `split_json` is a JSON list of folds; each fold is a dict with
    'fold', and 'train'/'val'/'test' lists of ROW INDICES into `data_csv`
    (peptide/TCR-aware, leak-free). Returns (train_ds, val_ds, test_ds, pos_weight).
    Metrics must be reported on the held-out `test` split — never on val.
    """
    import json
    df = pd.read_csv(data_csv)
    if "domain" not in df.columns:
        df["domain"] = 0
    folds = json.load(open(split_json))
    rec = next((f for f in folds if int(f.get("fold", -1)) == int(fold)), None)
    if rec is None:
        rec = folds[int(fold)]

    def _sub(idx):
        return df.iloc[list(idx)].reset_index(drop=True)

    train_ds = AblationDataset(_sub(rec["train"]), crystal_weight=crystal_weight, verbose=verbose)
    val_ds = AblationDataset(_sub(rec["val"]), crystal_weight=1.0, verbose=False)
    test_ds = AblationDataset(_sub(rec["test"]), crystal_weight=1.0, verbose=False)
    if verbose:
        print(f"\n  Fold {fold}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  (held-out test)")
    return train_ds, val_ds, test_ds, train_ds.pos_weight
