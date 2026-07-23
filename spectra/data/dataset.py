"""
dataset_v2.py — Redesigned ProteinGraphDataset for STAG-LLM v2
================================================================

Key changes from dataset.py / dataset2.py:

1. **4 independent chain sequences** instead of 2 concatenated strings.
   Returns (mhc_str, pep_str, tra_str, trb_str) — each chain tokenized
   separately by ESM so there's no cross-chain positional contamination.

2. **4-class chain_id**: A=0(MHC), C=1(pep), D=2(TRA), E=3(TRB).
   Previous dataset2 collapsed TRA and TRB into one group (2), losing
   the ability to distinguish α from β chain in the graph.

3. **chain_pos stored in ndata**: 0-indexed residue position within each
   chain's sorted node list. Required for encoder-mode ESM→graph mapping.

4. **ESM collate tokenizes 4 chains independently**: No stripping of | or J.
   Each chain gets its own ESM input [BOS, aa1, aa2, ..., EOS].
   Clean 1:1 alignment: ESM token position k+1 = chain_pos k (offset by BOS).

5. **Backward compatible**: Still returns graph + sequence data in a tuple.
   The collate function returns a dict-style batch for clarity.
"""

import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.geometry import add_sidechain_vector
import dgl
import networkx as nx
from graphein.protein.subgraphs import extract_subgraph_from_node_list
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import time

pdbparser = PDBParser(QUIET=True)


# ============================================================
# Amino acid encoding (unchanged — for backward compat / Model A)
# ============================================================

enc_dict = {
    'ALA': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'CYS': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'ASP': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'GLU': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'PHE': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'GLY': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'HIS': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'ILE': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'LYS': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'LEU': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'MET': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'ASN': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'PRO': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'GLN': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'ARG': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'SER': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'THR': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'VAL': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'TRP': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'TYR': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'MASK': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
}

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY' + 'J' + '|'
char_to_int = {c: i for i, c in enumerate(AA_VOCAB)}
PAD_IDX = char_to_int['J']
SEP_IDX = char_to_int['|']


def _safe_resname(n):
    parts = str(n).split(":")
    return parts[1] if len(parts) > 1 else "MASK"


# ============================================================
# Edge feature encoding (unchanged)
# ============================================================

BOND_TYPE_TO_IDX = {
    "peptide_bond": 0,
    "hbond": 1,
    "hydrophobic": 2,
    "ionic": 3,
    "aromatic": 4,
    "cation_pi": 5,
    "aromatic_sulphur": 6,
}
NUM_BOND_TYPES = len(BOND_TYPE_TO_IDX)


def _encode_edge_features(sg_nx, src_nodes, dst_nodes):
    """Build multi-hot edge feature tensor from Graphein edge 'kind' attributes."""
    pair_to_types = {}
    if isinstance(sg_nx, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, key, data in sg_nx.edges(data=True, keys=True):
            kind = data.get("kind", set())
            if isinstance(kind, str):
                kind = {kind}
            pair_to_types.setdefault((u, v), set()).update(kind)
            pair_to_types.setdefault((v, u), set()).update(kind)
    else:
        for u, v, data in sg_nx.edges(data=True):
            kind = data.get("kind", set())
            if isinstance(kind, str):
                kind = {kind}
            pair_to_types.setdefault((u, v), set()).update(kind)
            pair_to_types.setdefault((v, u), set()).update(kind)

    E = len(src_nodes)
    edge_feat = torch.zeros(E, NUM_BOND_TYPES, dtype=torch.float32)
    for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
        kinds = pair_to_types.get((s, d), set())
        for k in kinds:
            idx = BOND_TYPE_TO_IDX.get(k)
            if idx is not None:
                edge_feat[i, idx] = 1.0
    return edge_feat


# ============================================================
# Graphein config (unchanged)
# ============================================================

def get_graphein_config():
    from graphein.protein.config import ProteinGraphConfig
    from graphein.protein.edges.distance import (
        add_hydrogen_bond_interactions, add_peptide_bonds,
        add_hydrophobic_interactions, add_ionic_interactions,
        add_aromatic_interactions, add_cation_pi_interactions,
        add_aromatic_sulphur_interactions,
    )
    from graphein.protein.features.nodes.amino_acid import (
        amino_acid_one_hot, hydrogen_bond_acceptor,
        hydrogen_bond_donor, expasy_protein_scale,
    )

    return ProteinGraphConfig(
        granularity='CA',
        keep_hets=[],
        alt_locs='max_occupancy',
        verbose=False,
        exclude_waters=True,
        deprotonate=False,
        edge_construction_functions=[
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_hydrophobic_interactions,
            add_ionic_interactions,
            add_aromatic_interactions,
            add_cation_pi_interactions,
            add_aromatic_sulphur_interactions,
        ],
        node_metadata_functions=[
            hydrogen_bond_acceptor,
            hydrogen_bond_donor,
            expasy_protein_scale,
        ],
    )


def parse_node(n: str):
    parts = str(n).split(":")
    return parts[0], parts[1], int(parts[2])


def pad_or_truncate(sequence: str, max_length: int, pad_char: str = 'J') -> str:
    return sequence[:max_length].ljust(max_length, pad_char)


# ============================================================
# Chain ID and edge type constants
# ============================================================

# 4-class chain_id (distinct for each chain)
CHAIN_ID_MAP = {"A": 0, "C": 1, "D": 2, "E": 3}  # MHC, pep, TRA, TRB
NUM_CHAIN_IDS = 4

# Chain groups for edge type classification (TRA/TRB grouped as "tcr")
CHAIN_GROUP = {"A": "mhc", "C": "pep", "D": "tcr", "E": "tcr"}

# 5 edge types
ETYPE_MAP = {
    ("tcr", "tcr"): 0,  # tcr2tcr (includes TRA↔TRB)
    ("pep", "pep"): 1,  # pep2pep
    ("mhc", "mhc"): 2,  # mhc2mhc
    ("pep", "mhc"): 3, ("mhc", "pep"): 3,  # pep_mhc
    ("tcr", "pep"): 4, ("pep", "tcr"): 4,  # tcr_pmhc (interface)
    ("tcr", "mhc"): 4, ("mhc", "tcr"): 4,
}

# Max residues per chain
MAX_RESIDUES = {"A": 180, "C": 14, "D": 180, "E": 180}


# ============================================================
# Dataset
# ============================================================

class ProteinGraphDataset(Dataset):
    """
    Builds protein graphs from PDB files with 4 independent chain sequences.

    Returns per sample:
        tcr_id       : str
        label        : tensor(long)
        graph        : DGL graph with ndata/edata
        struct_feat  : tensor(float) — extra CSV features
        mhc_str      : str — raw MHC AA sequence (no padding)
        pep_str      : str — raw peptide AA sequence (no padding)
        tra_str      : str — raw TCRα AA sequence (no padding)
        trb_str      : str — raw TCRβ AA sequence (no padding)
        chain_lengths: dict — {chain_id: n_residues} for alignment

    ndata stored:
        "x"                : [N, 20]   AA one-hot
        "coords"           : [N, 3]    Cα coordinates
        "hbond_acceptors"  : [N, 1]    H-bond acceptor flag
        "hbond_donors"     : [N, 1]    H-bond donor flag
        "sidechain_vector" : [N, 3]    Sidechain direction vector
        "chain_id"         : [N]       4-class: 0=MHC, 1=pep, 2=TRA, 3=TRB
        "chain_pos"        : [N]       0-indexed position within chain

    edata stored:
        "feat"  : [E, 7]  bond type multi-hot
        "etype" : [E]     5-class edge type
    """

    CACHE_VERSION = 3  # bump to invalidate old caches

    def __init__(self, annotation_file=None, pdb_dfs=None,
                 pdb_dir=os.environ.get('SPECTRA_PDB_DIR', './data/pdbs'),
                 graphein_config=None,
                 cache_dir=os.environ.get('SPECTRA_CACHE_DIR', './.cache'),
                 use_cache=True):
        if graphein_config is None:
            graphein_config = get_graphein_config()

        if annotation_file is not None:
            self.pdb_dfs = pd.read_csv(annotation_file)
        else:
            assert pdb_dfs is not None
            self.pdb_dfs = pdb_dfs.copy()

        self.pdb_dfs = self.pdb_dfs.dropna(axis=0, how="any").reset_index(drop=True)

        avail_pdbs = {Path(i).stem for i in glob(os.path.join(pdb_dir, '*.pdb'))}
        self.pdb_dfs = self.pdb_dfs[
            self.pdb_dfs['tcr_id'].isin(avail_pdbs)
        ].reset_index(drop=True)

        drop_cols = {
            'tcr_id', 'CDR3a', 'CDR3b', 'MHC_sequence', 'peptide',
            'TCR_A_sequence', 'TCR_B_sequence', 'label',
        }
        self.feature_cols = [c for c in self.pdb_dfs.columns if c not in drop_cols]

        self.pdb_dir = pdb_dir
        self.graphein_config = graphein_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        self._mask = enc_dict["MASK"]
        self._enc_dict = enc_dict
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_paths(self, tcr_id: str):
        sub = tcr_id[:2]
        d = self.cache_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        return (d / f"{tcr_id}.graph.bin", d / f"{tcr_id}.pt")

    def __len__(self):
        return len(self.pdb_dfs)

    def __getitem__(self, idx):
        row = self.pdb_dfs.iloc[idx]
        tcr_id = row['tcr_id']
        label = int(row['label'])

        struct_features = torch.as_tensor(
            row[self.feature_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        graph_path, aux_path = self._cache_paths(tcr_id)

        if self.use_cache and graph_path.exists() and aux_path.exists():
            gs, _ = dgl.load_graphs(str(graph_path))
            g_cached = gs[0]
            # Check cache version — invalidate if outdated
            aux = torch.load(str(aux_path), map_location="cpu", weights_only=False)
            if aux.get("cache_version", 0) >= self.CACHE_VERSION:
                self._cache_hits += 1
                return (
                    tcr_id,
                    torch.tensor(label, dtype=torch.long),
                    g_cached,
                    aux["struct_features"],
                    aux["mhc_str"],
                    aux["pep_str"],
                    aux["tra_str"],
                    aux["trb_str"],
                    aux["chain_lengths"],
                )
            else:
                self._cache_misses += 1  # cache outdated, rebuild
        else:
            self._cache_misses += 1

        # ---- Build sample ----
        result = self._build_sample(tcr_id)
        g, mhc_str, pep_str, tra_str, trb_str, chain_lengths = result

        aux = {
            "struct_features": struct_features.cpu(),
            "mhc_str": mhc_str,
            "pep_str": pep_str,
            "tra_str": tra_str,
            "trb_str": trb_str,
            "chain_lengths": chain_lengths,
            "cache_version": self.CACHE_VERSION,
        }

        if self.use_cache:
            dgl.save_graphs(str(graph_path), [g.cpu()])
            torch.save(aux, str(aux_path))

        return (
            tcr_id,
            torch.tensor(label, dtype=torch.long),
            g,
            struct_features,
            mhc_str,
            pep_str,
            tra_str,
            trb_str,
            chain_lengths,
        )

    def _build_sample(self, tcr_id: str):
        pdb_path = os.path.join(self.pdb_dir, f"{tcr_id}.pdb")
        g_nx = construct_graph(config=self.graphein_config, path=pdb_path)

        # Select nodes within residue limits per chain
        node_list = [
            n for n, d in g_nx.nodes(data=True)
            if d.get('chain_id') in MAX_RESIDUES
            and 1 <= d.get("residue_number", 0) <= MAX_RESIDUES[d.get('chain_id')]
        ]
        sg_nx = extract_subgraph_from_node_list(g_nx, node_list)
        add_sidechain_vector(sg_nx)

        # Ensure attributes exist
        for n, d in sg_nx.nodes(data=True):
            res = _safe_resname(n)
            d["aa20"] = self._enc_dict.get(res, self._mask)
            d["coords"] = d.get("coords", (0.0, 0.0, 0.0))
            d["hbond_acceptors"] = d.get("hbond_acceptors", 0.0)
            d["hbond_donors"] = d.get("hbond_donors", 0.0)
            d["sidechain_vector"] = d.get("sidechain_vector", (0.0, 0.0, 0.0))

        # ---- Ordered node list + per-node metadata ----
        nodes = list(sg_nx.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Parse chain info for each node
        node_chain_letter = {}  # node_name → chain letter (A/C/D/E)
        node_chain_id = {}      # node_name → chain_id (0/1/2/3)
        node_chain_group = {}   # node_name → chain group (mhc/pep/tcr)
        for n in nodes:
            ch, _, _ = parse_node(n)
            node_chain_letter[n] = ch
            node_chain_id[n] = CHAIN_ID_MAP.get(ch, 0)
            node_chain_group[n] = CHAIN_GROUP.get(ch, "tcr")

        # ---- chain_pos: sorted position within each chain ----
        from collections import defaultdict
        chain_sorted = defaultdict(list)  # chain_letter → [(residue_pos, node_idx)]
        for i, n in enumerate(nodes):
            _, _, pos = parse_node(n)
            chain_sorted[node_chain_letter[n]].append((pos, i))
        for ch in chain_sorted:
            chain_sorted[ch].sort()

        chain_pos_arr = [0] * len(nodes)
        for ch, items in chain_sorted.items():
            for rank, (_, orig_idx) in enumerate(items):
                chain_pos_arr[orig_idx] = rank

        # ---- Build edges ----
        src_names, dst_names = [], []
        if isinstance(sg_nx, (nx.MultiGraph, nx.MultiDiGraph)):
            seen = set()
            for u, v, _ in sg_nx.edges(keys=True):
                if (u, v) not in seen:
                    src_names.append(u)
                    dst_names.append(v)
                    seen.add((u, v))
        else:
            for u, v in sg_nx.edges():
                src_names.append(u)
                dst_names.append(v)

        src_idx = [node_to_idx[n] for n in src_names]
        dst_idx = [node_to_idx[n] for n in dst_names]

        g = dgl.graph((src_idx, dst_idx), num_nodes=len(nodes))

        # Edge features (multi-hot bond types)
        edge_feat = _encode_edge_features(sg_nx, src_names, dst_names)
        g.edata["feat"] = edge_feat

        # Edge types (5 classes)
        etypes = []
        for s, d in zip(src_names, dst_names):
            sg = node_chain_group[s]
            dg = node_chain_group[d]
            etypes.append(ETYPE_MAP.get((sg, dg), 0))
        g.edata["etype"] = torch.tensor(etypes, dtype=torch.long)

        # Add reverse edges (symmetric)
        g = dgl.add_reverse_edges(g, copy_edata=True)

        # ---- Node features ----
        x = torch.as_tensor(np.array(
            [self._enc_dict.get(_safe_resname(n), self._mask) for n in nodes],
            dtype=np.float32
        ))
        coords = torch.as_tensor(np.asarray(
            [sg_nx.nodes[n]["coords"] for n in nodes], dtype=np.float32
        ))
        hacc = torch.as_tensor(np.asarray(
            [sg_nx.nodes[n]["hbond_acceptors"] for n in nodes], dtype=np.float32
        )).unsqueeze(1)
        hdon = torch.as_tensor(np.asarray(
            [sg_nx.nodes[n]["hbond_donors"] for n in nodes], dtype=np.float32
        )).unsqueeze(1)
        scv = torch.as_tensor(np.asarray(
            [sg_nx.nodes[n]["sidechain_vector"] for n in nodes], dtype=np.float32
        ))

        g.ndata["x"] = x                     # [N, 20]
        g.ndata["coords"] = coords           # [N, 3]
        g.ndata["hbond_acceptors"] = hacc     # [N, 1]
        g.ndata["hbond_donors"] = hdon        # [N, 1]
        g.ndata["sidechain_vector"] = scv     # [N, 3]

        # 4-class chain_id + chain_pos
        g.ndata["chain_id"] = torch.tensor(
            [node_chain_id[n] for n in nodes], dtype=torch.long
        )
        g.ndata["chain_pos"] = torch.tensor(chain_pos_arr, dtype=torch.long)

        # ---- Extract per-chain sequences (raw, no padding) ----
        def _chain_to_seq(chain_letter):
            items = chain_sorted.get(chain_letter, [])
            seq = []
            for _, node_idx in items:
                n = nodes[node_idx]
                res3 = _safe_resname(n)
                try:
                    seq.append(seq1(res3))
                except Exception:
                    seq.append("X")
            return "".join(seq)

        mhc_str = _chain_to_seq("A")
        pep_str = _chain_to_seq("C")
        tra_str = _chain_to_seq("D")
        trb_str = _chain_to_seq("E")

        chain_lengths = {
            "mhc": len(mhc_str),
            "pep": len(pep_str),
            "tra": len(tra_str),
            "trb": len(trb_str),
        }

        return g, mhc_str, pep_str, tra_str, trb_str, chain_lengths


# ============================================================
# Collate functions
# ============================================================

def make_esm_collate_fn(esm_tokenizer):
    """
    Collate for Model B (ESM-based) with 4 independent chain inputs.

    Tokenizes each chain separately:
      - MHC → [BOS, mhc_aa1, ..., mhc_aaN, EOS]
      - Peptide → [BOS, pep_aa1, ..., pep_aaM, EOS]
      - TCRα → [BOS, tra_aa1, ..., tra_aaK, EOS]
      - TCRβ → [BOS, trb_aa1, ..., trb_aaL, EOS]

    This gives clean 1:1 alignment between ESM token positions and
    graph node chain_pos values (with BOS offset of 1).

    Returns dict with keys:
        tcr_ids      : List[str]
        labels       : [B] long
        mhc_ids      : [B, L_mhc] ESM token IDs
        mhc_mask     : [B, L_mhc] bool
        pep_ids      : [B, L_pep] ESM token IDs
        pep_mask     : [B, L_pep] bool
        tra_ids      : [B, L_tra] ESM token IDs
        tra_mask     : [B, L_tra] bool
        trb_ids      : [B, L_trb] ESM token IDs
        trb_mask     : [B, L_trb] bool
        graph        : DGL batched graph
        chain_lengths: List[dict]
    """

    def _tokenize(sequences, max_length=256):
        """Tokenize raw AA strings (no stripping needed — already clean)."""
        clean = [" ".join(list(s)) if s else "<unk>" for s in sequences]
        out = esm_tokenizer(
            clean, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return out["input_ids"], out["attention_mask"].bool()

    def collate_fn(batch):
        (tcr_ids, labels, graphs, struct_feats,
         mhc_strs, pep_strs, tra_strs, trb_strs,
         chain_lengths_list) = zip(*batch)

        # Tokenize each chain independently
        mhc_ids, mhc_mask = _tokenize(mhc_strs, max_length=200)
        pep_ids, pep_mask = _tokenize(pep_strs, max_length=30)
        tra_ids, tra_mask = _tokenize(tra_strs, max_length=200)
        trb_ids, trb_mask = _tokenize(trb_strs, max_length=200)

        return {
            "tcr_ids": list(tcr_ids),
            "labels": torch.stack(labels),
            "mhc_ids": mhc_ids,
            "mhc_mask": mhc_mask,
            "pep_ids": pep_ids,
            "pep_mask": pep_mask,
            "tra_ids": tra_ids,
            "tra_mask": tra_mask,
            "trb_ids": trb_ids,
            "trb_mask": trb_mask,
            "graph": dgl.batch(graphs),
            "chain_lengths": list(chain_lengths_list),
        }

    return collate_fn


def custom_collate_fn(batch):
    """Collate for Model A (backward compat). Concatenates chains with separators."""
    (tcr_ids, labels, graphs, struct_feats,
     mhc_strs, pep_strs, tra_strs, trb_strs,
     chain_lengths_list) = zip(*batch)

    tcr_ids = list(tcr_ids)
    labels = torch.stack(labels)

    # Reconstruct concatenated sequences for backward compatibility
    full_seqs = []
    for mhc, pep, tra, trb in zip(mhc_strs, pep_strs, tra_strs, trb_strs):
        s = pad_or_truncate(mhc, 180) + "|" + pad_or_truncate(pep, 14) + "|" \
            + pad_or_truncate(tra, 180) + "|" + pad_or_truncate(trb, 180)
        full_seqs.append(torch.tensor(
            [char_to_int.get(c, PAD_IDX) for c in s], dtype=torch.long))

    from torch.nn.utils.rnn import pad_sequence
    full_seq = pad_sequence(full_seqs, batch_first=True, padding_value=PAD_IDX)

    return (tcr_ids, labels, full_seq, dgl.batch(graphs), None)


# ============================================================
# Smoke test
# ============================================================

if __name__ == '__main__':
    start_time = time.perf_counter()

    # Use the ImmunoStruct training CSV
    csv_path = os.environ.get('SPECTRA_DATA_CSV', './data/training.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.getenv('SCRATCH', '/tmp'),
                                "data/STAG-LLM/features/structure_feat.csv")

    df = pd.read_csv(csv_path)
    ds = ProteinGraphDataset(
        pdb_dfs=df,
        cache_dir=os.environ.get('SPECTRA_CACHE_DIR', './.cache'),
        use_cache=True,
    )

    from dgl.dataloading import GraphDataLoader
    dataloader = GraphDataLoader(
        ds, batch_size=2, collate_fn=custom_collate_fn, shuffle=True)
    batch = next(iter(dataloader))
    tcr_id, label, full_seq, bg, _ = batch

    print(f"Batch size:        {len(tcr_id)}")
    print(f"full_seq shape:    {full_seq.shape}")
    print(f"graph nodes:       {bg.num_nodes()}")
    print(f"graph edges:       {bg.num_edges()}")
    print(f"node feat shape:   {bg.ndata['x'].shape}")
    print(f"coord shape:       {bg.ndata['coords'].shape}")
    print(f"edge feat shape:   {bg.edata['feat'].shape}")
    print(f"chain_id unique:   {bg.ndata['chain_id'].unique().tolist()}")
    print(f"chain_pos range:   {bg.ndata['chain_pos'].min().item()}-{bg.ndata['chain_pos'].max().item()}")
    print(f"unique etypes:     {bg.edata['etype'].unique().tolist()}")

    # Test the ESM collate
    print("\n--- ESM collate test ---")
    sample = ds[0]
    tcr_id_s, label_s, g_s, sf_s, mhc_s, pep_s, tra_s, trb_s, cl_s = sample
    print(f"tcr_id:          {tcr_id_s}")
    print(f"MHC seq ({cl_s['mhc']}aa): {mhc_s[:30]}...")
    print(f"Peptide ({cl_s['pep']}aa):  {pep_s}")
    print(f"TRA seq ({cl_s['tra']}aa):  {tra_s[:30]}...")
    print(f"TRB seq ({cl_s['trb']}aa):  {trb_s[:30]}...")
    print(f"chain_id values: {g_s.ndata['chain_id'].unique().tolist()}")
    print(f"Total nodes: {g_s.num_nodes()}")

    # Verify chain_pos alignment
    for cid, cname, cseq in [(0, "MHC", mhc_s), (1, "pep", pep_s),
                               (2, "TRA", tra_s), (3, "TRB", trb_s)]:
        mask = (g_s.ndata["chain_id"] == cid)
        n_nodes = mask.sum().item()
        assert n_nodes == len(cseq), \
            f"Mismatch: chain {cname} has {n_nodes} nodes but seq len {len(cseq)}"
        positions = g_s.ndata["chain_pos"][mask]
        assert positions.max().item() == n_nodes - 1, \
            f"chain_pos max should be {n_nodes-1}, got {positions.max().item()}"
    print("Chain-pos alignment: VERIFIED")

    end_time = time.perf_counter()
    print(f"\nExecution time: {end_time - start_time:.4f} seconds")