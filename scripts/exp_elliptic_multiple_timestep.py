
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Optional

from src.graph.components import largest_component
from src.feeds.trivial import S0_oracle_high_illicit, S1_random, S2_largest_degree_difference, S3_mixed_collect_distribute
from src.feeds.motif import S4_motif_based_coherent
from src.spnsa import spnsa


# =========================
# CONFIG
# =========================
DATASET_NAME = "Elliptic"

# SPNSA parameters
(k,r) = (200, 1)

# S4 parameters (match paper defaults)
S4_PARAMS = dict(
    C=300_000,
    centers=5,
    d_max=2,
    motif_params=dict(
        alpha=0.2, beta=0.3, gamma=0.8,
        s_in=0.5, s_out=0.4, s_leaf=0.6,
        star_ratio=15.0, leaf_deg_max=2
    ),
    max_in=500,
    max_out=500,
)

S1_TRIALS = 100   # paper table: S1 averaged over 100 trials
SEED = 0




def load_time_step_map(features_csv: str) -> dict[str, int]:
    """
    Load txId -> time_step from elliptic_txs_features.csv.

    Works for both:
      - headered files with columns ['txId', 'time_step', ...]
      - headerless original Elliptic feature file where:
            col 0 = txId, col 1 = time_step
    """
    try:
        # Case 1: file has headers
        df = pd.read_csv(
            features_csv,
            usecols=["txId", "time_step"],
            dtype={"txId": str, "time_step": int},
        )
    except ValueError:
        # Case 2: original file without headers
        df = pd.read_csv(
            features_csv,
            header=None,
            usecols=[0, 1],
            names=["txId", "time_step"],
            dtype={0: str, 1: int},
        )

    df["txId"] = df["txId"].str.strip()
    return dict(zip(df["txId"], df["time_step"]))


def build_graph_from_edgelist(
    edgelist_csv: str,
    directed: bool = True,
    node_subset: Optional[set[str]] = None,
) -> nx.Graph:
    """
    Build a NetworkX graph from the Elliptic edgelist CSV.

    Expected columns: `txId1`, `txId2`.

    If node_subset is given, build the induced graph on that node set:
      - keep only edges with both endpoints in node_subset
      - add all nodes in node_subset explicitly so isolated nodes are preserved
    """
    df = pd.read_csv(edgelist_csv, usecols=["txId1", "txId2"], dtype=str)
    df["txId1"] = df["txId1"].str.strip()
    df["txId2"] = df["txId2"].str.strip()

    if node_subset is not None:
        mask = df["txId1"].isin(node_subset) & df["txId2"].isin(node_subset)
        df = df.loc[mask]

    G = nx.DiGraph() if directed else nx.Graph()

    if node_subset is not None:
        G.add_nodes_from(node_subset)

    G.add_edges_from(zip(df["txId1"], df["txId2"]))
    return G


def load_label_sets(classes_csv: str) -> tuple[set[str], set[str]]:
    """
    Load illicit and licit node sets from elliptic_txs_classes.csv.
    Assumes labels:
      '1' -> illicit
      '2' -> licit
    Unknowns are ignored.
    """
    df = pd.read_csv(classes_csv, usecols=["txId", "class"], dtype=str)
    df["txId"] = df["txId"].str.strip()
    df["class"] = df["class"].str.strip()

    illicit = set(df.loc[df["class"] == "1", "txId"])
    licit = set(df.loc[df["class"] == "2", "txId"])
    return illicit, licit


def load_graph_and_illicit(split: str = "tune", t: int = 34):
    """
    Load either:
      - split='tune' : first t time steps (1..t)
      - split='test' : remaining time steps (t+1..end)

    Returns
    -------
    G : nx.DiGraph
        Graph induced by the nodes in the requested time range.
    illicit : set[str]
        Illicit nodes in that split.
    licit : set[str]
        Licit nodes in that split.
    """
    classes_csv = Path("data/elliptic/elliptic_txs_classes.csv")
    edgelist_csv = Path("data/elliptic/elliptic_txs_edgelist.csv")
    features_csv = Path("data/elliptic/elliptic_txs_features.csv")

    for p in [classes_csv, edgelist_csv, features_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p} (place it under data/elliptic/)")

    tx_to_time = load_time_step_map(str(features_csv))

    if split == "tune":
        split_nodes = {tx for tx, ts in tx_to_time.items() if ts <= t}
    elif split == "test":
        split_nodes = {tx for tx, ts in tx_to_time.items() if ts > t}
    else:
        raise ValueError("split must be either 'tune' or 'test'")

    G = build_graph_from_edgelist(
        str(edgelist_csv),
        directed=True,
        node_subset=split_nodes,
    )

    illicit_all, licit_all = load_label_sets(str(classes_csv))

    illicit = {v for v in illicit_all if v in split_nodes}
    licit = {v for v in licit_all if v in split_nodes}

    return G, illicit, licit


def load_tune_and_test_graphs(t: int = 34):
    """
    Convenience wrapper returning both splits.
    """
    G_tune, illicit_tune, licit_tune = load_graph_and_illicit(split="tune", t=t)
    G_test, illicit_test, licit_test = load_graph_and_illicit(split="test", t=t)

    return {
        "tune": (G_tune, illicit_tune, licit_tune),
        "test": (G_test, illicit_test, licit_test),
    }




def metrics(H: nx.Graph, illicit: set, licit: set):
    nodes = set(H.nodes())
    V = int(H.number_of_nodes())
    E = int(H.number_of_edges())
    I = float(len(nodes & illicit))
    L = float(len(nodes & licit))
    IR = float(I / V) if V > 0 else 0.0
    LIR = float(I / (L + I)) if (L+I) > 0 else 0.0
    return V, E, I, IR, L, LIR


def spnsa_metrics(G: nx.DiGraph, feed, r: int, illicit: set, licit: set):
    H, _ = spnsa(G, feed, radius=r)
    return metrics(H, illicit, licit)


def main():
    #G, illicit, licit = load_graph_and_illicit()
    # First 34 time steps
    G, illicit, licit = load_graph_and_illicit(split="test", t=32)

    nodes = list(G.nodes())

    rows = []

    # S0
    f0 = S0_oracle_high_illicit(G, illicit, k=k, seed=SEED)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f0, r=r, illicit=illicit, licit=licit)
    rows.append(("S0", k, r, V, E, I, IR, L, LIR))

    # S1 (avg over trials)
    Vs, Es, Is, IRs, Ls, LIRs = [], [], [], [], [], []
    for t in tqdm(range(S1_TRIALS), desc="S1 trials"):
        f1 = S1_random(nodes, k=k, seed=SEED + t)
        H, _ = spnsa(G, f1, radius=r, verbose=False)
        v, e, i, ir, l, lir = metrics(H, illicit, licit)
        Vs.append(v); Es.append(e); Is.append(i); IRs.append(ir); Ls.append(l); LIRs.append(lir)
    rows.append((
        "S1", k, r,
        int(round(sum(Vs)/len(Vs))),
        int(round(sum(Es)/len(Es))),
        float(sum(Is)/len(Is)),
        float(sum(IRs)/len(IRs)),
        float(sum(Ls)/len(Ls)),
        float(sum(LIRs)/len(LIRs))
    ))

    # S2
    f2 = S2_largest_degree_difference(G, k=k)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f2, r=r, illicit=illicit, licit=licit)
    rows.append(("S2", k, r, V, E, I, IR, L, LIR))

    # S3
    f3 = S3_mixed_collect_distribute(G, k=k)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f3, r=r, illicit=illicit, licit=licit)
    rows.append(("S3", k, r, V, E, I, IR, L, LIR))

    # S4
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f4, r=r, illicit=illicit, licit=licit)
    rows.append(("S4", k, r, V, E, I, IR, L, LIR))

    
    # S4 induced subgraph
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    H = G.subgraph(f4).copy()
    V, E, I, IR, L, LIR = metrics(H, illicit, licit)
    rows.append(("S4 induced", k, r, V, E, I, IR, L, LIR))

    df = pd.DataFrame(rows, columns=["strategy", "k", "r", "|V|", "|E|", "|I|", "IR", "L", "LIR"])
    print("\n=== Feed strategies (S0–S4):", DATASET_NAME, "===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
