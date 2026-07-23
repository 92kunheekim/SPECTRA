"""
ProteinGraphDataset — corrected version
=========================================
Fixes applied:
  1. Cache return signature matches non-cache path (7 values)
  2. Duplicate one_hot_encode_sequence removed
  3. Multi-hot edge features (bond type encoding) instead of constant 1s
  4. Optional: concatenate hbond_acceptors/donors/sidechain_vector into ndata["x"]
  5. Undirected→directed conversion via dgl.add_reverse_edges for symmetric message passing
  6. ndata["x"] stays [N,20] separate from coords [N,3] for EGNN compatibility
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
from torch.nn.utils.rnn import pad_sequence
import time

pdbparser = PDBParser(QUIET=True)

# ============================================================
# Amino acid encoding
# ============================================================

enc_dict = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'MASK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY' + 'J' + '|'  # J=PAD, |=SEP
char_to_int = {c: i for i, c in enumerate(AA_VOCAB)}
PAD_IDX = char_to_int['J']
SEP_IDX = char_to_int['|']

# ============================================================
# FIX #2: Single definition of one_hot_encode_sequence
# ============================================================

def one_hot_encode_sequence(sequence: str):
    """Encode a string of 1-letter AA codes into integer indices."""
    one_hot_encoded = np.zeros((len(sequence), len(char_to_int)), dtype=np.float32)
    for i, char in enumerate(sequence):
        one_hot_encoded[i, char_to_int.get(char, PAD_IDX)] = 1.0
    return torch.from_numpy(one_hot_encoded)


def _safe_resname(n):
    parts = str(n).split(":")
    return parts[1] if len(parts) > 1 else "MASK"


# ============================================================
# FIX #3: Multi-hot edge feature encoding
# ============================================================

# Graphein stores edge "kind" attribute. Map to multi-hot indices.
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
    """
    Build multi-hot edge feature tensor from Graphein edge 'kind' attributes.

    For a MultiGraph, there can be multiple edges between the same pair of nodes
    (e.g., two residues connected by both a peptide bond and a hydrogen bond).
    We collapse these into a single multi-hot vector per (src, dst) pair.

    Args:
        sg_nx:     NetworkX graph (MultiGraph or Graph)
        src_nodes: list of source node names (aligned to DGL edge order)
        dst_nodes: list of destination node names

    Returns:
        edge_feat: [E, NUM_BOND_TYPES] float tensor
    """
    # Pre-build a lookup: (u, v) -> set of bond types
    pair_to_types = {}
    if isinstance(sg_nx, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, key, data in sg_nx.edges(data=True, keys=True):
            kind = data.get("kind", set())
            if isinstance(kind, str):
                kind = {kind}
            pair_to_types.setdefault((u, v), set()).update(kind)
            # Also store reverse for undirected
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
# Graphein config
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


def get_sequence(structure, chainid):
    seq = []
    for chain in structure.get_chains():
        if chain.id == chainid:
            for residue in chain:
                hetflag, _, _ = residue.id
                if hetflag != " ":
                    continue
                seq.append(seq1(residue.get_resname()))
    return "".join(seq)


def parse_node(n: str):
    parts = str(n).split(":")
    return parts[0], parts[1], int(parts[2])


def chain_sequence_from_graphein_nodes(g_nx, chain_id: str):
    items = []
    for n in g_nx.nodes():
        ch, res3, pos = parse_node(n)
        if ch == chain_id:
            items.append((pos, res3))
    items.sort(key=lambda x: x[0])
    seq = []
    for _, res3 in items:
        try:
            seq.append(seq1(res3))
        except Exception:
            seq.append("X")
    return "".join(seq)


def pad_peptide_sequence(sequence, max_length=11, padding_char='J'):
    return sequence.ljust(max_length, padding_char)


def pad_or_truncate(sequence: str, max_length: int, pad_char: str = 'J') -> str:
    """Truncate if longer than max_length, pad with pad_char if shorter."""
    return sequence[:max_length].ljust(max_length, pad_char)


# ============================================================
# Dataset
# ============================================================

class ProteinGraphDataset(Dataset):
    def __init__(self, annotation_file=None, pdb_dfs=None,
                 pdb_dir='${SPECTRA_ROOT}/data/STAG-LLM/data/top_structures/',
                 graphein_config=None,
                 cache_dir='${SPECTRA_ROOT}/data/STAG-LLM/immunostruct_cache',
                 use_cache=True):
        if graphein_config is None:
            graphein_config = get_graphein_config()

        if annotation_file is not None:
            self.pdb_dfs = pd.read_csv(annotation_file)
        else:
            assert pdb_dfs is not None
            self.pdb_dfs = pdb_dfs.copy()

        self.pdb_dfs = self.pdb_dfs.dropna(axis=0, how="any").reset_index(drop=True)
        # df_lab = pd.read_csv("${SPECTRA_ROOT}/data/STAG-LLM/data/final_dataset_modeled.csv")
        # df_lab['tcr_id'] = df_lab['peptide'] + "_" + df_lab['CDR3a'] + "_" + df_lab['CDR3b']
        # self.pdb_dfs = pd.merge(self.pdb_dfs, df_lab, on='tcr_id', how='inner')

        avail_pdbs = {Path(i).stem for i in glob(os.path.join(pdb_dir, '*.pdb'))}
        self.pdb_dfs = self.pdb_dfs[self.pdb_dfs['tcr_id'].isin(avail_pdbs)].reset_index(drop=True)

        drop_cols = {'tcr_id', 'CDR3a','CDR3b','MHC_sequence','peptide','TCR_A_sequence','TCR_B_sequence','label'}
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

        # ---- FIX #5: Cache path returns same 7-value signature as non-cache ----
        if self.use_cache and graph_path.exists() and aux_path.exists():
            self._cache_hits += 1
            gs, _ = dgl.load_graphs(str(graph_path))
            aux = torch.load(str(aux_path), map_location="cpu", weights_only=False)
            return (
                tcr_id,
                torch.tensor(label, dtype=torch.long),
                aux["pmhc_seq"],           # was missing before
                aux["tcr_seq"],            # was missing before
                aux["full_seq"],
                gs[0],
                aux["struct_features"],
            )
        else:
            self._cache_misses += 1

        # ---- Build sample ----
        g, full_seq, pmhc_seq, tcr_seq = self._build_sample(tcr_id)

        aux = {
            "pmhc_seq": pmhc_seq.to(torch.long).cpu(),
            "tcr_seq": tcr_seq.to(torch.long).cpu(),
            "full_seq": full_seq.to(torch.long).cpu(),
            "struct_features": struct_features.cpu(),
        }

        if self.use_cache:
            dgl.save_graphs(str(graph_path), [g.cpu()])
            torch.save(aux, str(aux_path))

        return (
            tcr_id,
            torch.tensor(label, dtype=torch.long),
            aux["pmhc_seq"],
            aux["tcr_seq"],
            aux["full_seq"],
            g,
            struct_features,
        )

    def _build_sample(self, tcr_id: str):
        pdb_path = os.path.join(self.pdb_dir, f"{tcr_id}.pdb")

        # --- Graphein graph ---
        g_nx = construct_graph(config=self.graphein_config, path=pdb_path)

        # Select nodes with residue number limits matching sequence truncation:
        #   A (MHC):     residues 1-180
        #   C (peptide): residues 1-14
        #   D (TCR α):   residues 1-180
        #   E (TCR β):   residues 1-180
        MAX_RESIDUES = {"A": 180, "C": 14, "D": 180, "E": 180}
        node_list = [
            n for n, d in g_nx.nodes(data=True)
            if d.get('chain_id') in MAX_RESIDUES
            and 1 <= d.get("residue_number", 0) <= MAX_RESIDUES[d.get('chain_id')]
        ]
        sg_nx = extract_subgraph_from_node_list(g_nx, node_list)
        add_sidechain_vector(sg_nx)

        # Ensure attrs exist on all nodes
        for n, d in sg_nx.nodes(data=True):
            res = _safe_resname(n)
            d["aa20"] = self._enc_dict.get(res, self._mask)
            d["coords"] = d.get("coords", (0.0, 0.0, 0.0))
            d["hbond_acceptors"] = d.get("hbond_acceptors", 0.0)
            d["hbond_donors"] = d.get("hbond_donors", 0.0)
            d["sidechain_vector"] = d.get("sidechain_vector", (0.0, 0.0, 0.0))

        # ---- Build DGL graph ----
        # Convert to DGL. from_networkx on an undirected graph creates one
        # directed edge per undirected edge. We then add reverse edges to
        # ensure symmetric message passing.
        nodes = list(sg_nx.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Collect edges from NetworkX
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

        # ---- FIX #3: Multi-hot edge features instead of constant 1s ----
        edge_feat = _encode_edge_features(sg_nx, src_names, dst_names)
        g.edata["feat"] = edge_feat  # [E, 7]

        # ---- FIX #7: Add reverse edges for symmetric message passing ----
        # dgl.add_reverse_edges copies edge features to the new reverse edges
        g = dgl.add_reverse_edges(g, copy_edata=True)

        # ---- Node features ----
        # ndata["x"]: one-hot AA encoding [N, 20] — kept separate from coords for EGNN
        x = torch.as_tensor(
            np.array([self._enc_dict.get(_safe_resname(n), self._mask)
                       for n in nodes], dtype=np.float32)
        )  # [N, 20]

        coords = torch.as_tensor(
            np.asarray([sg_nx.nodes[n]["coords"] for n in nodes], dtype=np.float32)
        )  # [N, 3]

        g.ndata["x"] = x
        g.ndata["coords"] = coords

        # Optional extra node features (stored but not concatenated into "x"
        # to maintain EGNN compatibility; concatenate into "x" if using a
        # non-equivariant GNN instead)
        hacc = torch.as_tensor(
            np.asarray([sg_nx.nodes[n]["hbond_acceptors"] for n in nodes], dtype=np.float32)
        ).unsqueeze(1)
        hdon = torch.as_tensor(
            np.asarray([sg_nx.nodes[n]["hbond_donors"] for n in nodes], dtype=np.float32)
        ).unsqueeze(1)
        scv = torch.as_tensor(
            np.asarray([sg_nx.nodes[n]["sidechain_vector"] for n in nodes], dtype=np.float32)
        )
        g.ndata["hbond_acceptors"] = hacc   # [N, 1]
        g.ndata["hbond_donors"] = hdon      # [N, 1]
        g.ndata["sidechain_vector"] = scv   # [N, 3]

        # --- Sequences (truncate + pad to fixed lengths) ---
        mhc_seq = pad_or_truncate(chain_sequence_from_graphein_nodes(sg_nx, "A"), 180)
        pep_seq = pad_or_truncate(chain_sequence_from_graphein_nodes(sg_nx, "C"), 14)
        tra_seq = pad_or_truncate(chain_sequence_from_graphein_nodes(sg_nx, "D"), 180)
        trb_seq = pad_or_truncate(chain_sequence_from_graphein_nodes(sg_nx, "E"), 180)

        pmhc_seq = mhc_seq + "|" + pep_seq
        tcr_seq = tra_seq + "|" + trb_seq
        full_seq = mhc_seq + "|" + pep_seq + "|" + tra_seq + "|" + trb_seq

        pmhc_seq = torch.tensor([char_to_int.get(c, PAD_IDX) for c in pmhc_seq], dtype=torch.long)
        tcr_seq = torch.tensor([char_to_int.get(c, PAD_IDX) for c in tcr_seq], dtype=torch.long)
        full_seq = torch.tensor([char_to_int.get(c, PAD_IDX) for c in full_seq], dtype=torch.long)

        return g, full_seq, pmhc_seq, tcr_seq


# ============================================================
# Collate
# ============================================================

def custom_collate_fn(batch):
    tcr_ids, labels, pmhc_seq, tcr_seq, full_seq, graphs, struct_features = zip(*batch)
    # for id, i in zip(tcr_ids,tcr_seq):
    #     if i.shape[0] != 321:
    #         print(f"TCR_id {id} has length {i.shape[0]}. {i}")
    tcr_ids = list(tcr_ids)
    labels = torch.stack(labels)
    pmhc_seq = torch.stack(pmhc_seq)
    tcr_seq = torch.stack(tcr_seq)
    full_seq = torch.stack(full_seq)
    batched_graph = dgl.batch(graphs)
    # struct_features = torch.stack(struct_features)
    struct_features=None
    return tcr_ids, labels, pmhc_seq, tcr_seq, full_seq, batched_graph, struct_features


# ============================================================
# Smoke test
# ============================================================

if __name__ == '__main__':
    start_time = time.perf_counter()
    df = pd.read_csv(os.path.join(os.getenv('SCRATCH'), "data/STAG-LLM/features/structure_feat.csv"))
    ds = ProteinGraphDataset(pdb_dfs=df)

    from dgl.dataloading import GraphDataLoader
    dataloader = GraphDataLoader(ds, batch_size=2, collate_fn=custom_collate_fn, shuffle=True)
    batch = next(iter(dataloader))
    tcr_id, label, pmhc_seq, tcr_seq, full_seq, bg, struct_features = batch

    print(f"full_seq shape:    {full_seq.shape}")
    print(f"pmhc_seq shape:    {pmhc_seq.shape}")
    print(f"tcr_seq shape:     {tcr_seq.shape}")
    print(f"graph nodes:       {bg.num_nodes()}")
    print(f"graph edges:       {bg.num_edges()}")
    print(f"node feat shape:   {bg.ndata['x'].shape}")       # [N_total, 20]
    print(f"coord shape:       {bg.ndata['coords'].shape}")   # [N_total, 3]
    print(f"edge feat shape:   {bg.edata['feat'].shape}")     # [E_total, 7]

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")