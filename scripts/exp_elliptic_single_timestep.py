
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm

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


from pathlib import Path
from typing import Optional
import pandas as pd
import networkx as nx


def _load_tx_time_map(features_csv: str) -> dict[str, int]:
    """
    Load txId -> timestep from elliptic_txs_features.csv.

    Supports both:
      - headered files with columns ['txId', 'time_step', ...]
      - original headerless Elliptic file where:
            col 0 = txId
            col 1 = timestep
    """
    try:
        # Headered version
        df = pd.read_csv(
            features_csv,
            usecols=["txId", "time_step"],
            dtype={"txId": str, "time_step": int},
        )
    except ValueError:
        # Original headerless version
        df = pd.read_csv(
            features_csv,
            header=None,
            usecols=[0, 1],
            names=["txId", "time_step"],
            dtype={0: str, 1: int},
        )

    df["txId"] = df["txId"].str.strip()
    return dict(zip(df["txId"], df["time_step"]))


def _load_label_sets(classes_csv: str) -> tuple[set[str], set[str], set[str]]:
    """
    Load illicit / licit / unknown node sets from elliptic_txs_classes.csv.
    Labels:
      '1'       -> illicit
      '2'       -> licit
      'unknown' -> unknown
    """
    df = pd.read_csv(classes_csv, usecols=["txId", "class"], dtype=str)
    df["txId"] = df["txId"].str.strip()
    df["class"] = df["class"].str.strip()

    illicit = set(df.loc[df["class"] == "1", "txId"])
    licit = set(df.loc[df["class"] == "2", "txId"])
    unknown = set(df.loc[df["class"] == "unknown", "txId"])
    return illicit, licit, unknown


def build_graph_from_edgelist(
    edgelist_csv: str,
    directed: bool = True,
    node_subset: Optional[set[str]] = None,
) -> nx.Graph:
    """
    Build a NetworkX graph from elliptic_txs_edgelist.csv.

    Expected columns: txId1, txId2

    If node_subset is given, returns the induced graph on node_subset.
    Isolated nodes in node_subset are preserved.
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


def load_elliptic_time_step_graph(
    t: int,
    data_dir: str = "data/elliptic",
    directed: bool = True,
):
    """
    Load a single Elliptic time step t as a separate graph.

    Parameters
    ----------
    t : int
        Time step in {1, ..., 49}.
    data_dir : str
        Directory containing:
          - elliptic_txs_features.csv
          - elliptic_txs_edgelist.csv
          - elliptic_txs_classes.csv
    directed : bool
        Whether to build a DiGraph (default) or Graph.

    Returns
    -------
    G_t : nx.Graph
        Graph for time step t.
    illicit_t : set[str]
        Illicit nodes in time step t.
    licit_t : set[str]
        Licit nodes in time step t.
    unknown_t : set[str]
        Unknown-labeled nodes in time step t.
    """
    if not (1 <= t <= 49):
        raise ValueError(f"t must be between 1 and 49, got {t}")

    data_dir = Path(data_dir)
    features_csv = data_dir / "elliptic_txs_features.csv"
    edgelist_csv = data_dir / "elliptic_txs_edgelist.csv"
    classes_csv = data_dir / "elliptic_txs_classes.csv"

    for p in [features_csv, edgelist_csv, classes_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    tx_to_time = _load_tx_time_map(str(features_csv))
    timestep_nodes = {tx for tx, ts in tx_to_time.items() if ts == t}

    G_t = build_graph_from_edgelist(
        str(edgelist_csv),
        directed=directed,
        node_subset=timestep_nodes,
    )

    illicit_all, licit_all, unknown_all = _load_label_sets(str(classes_csv))

    illicit_t = illicit_all & timestep_nodes
    licit_t = licit_all & timestep_nodes
    unknown_t = unknown_all & timestep_nodes

    return G_t, illicit_t, licit_t, unknown_t



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
    G, illicit, licit, unknown = load_elliptic_time_step_graph(t=1)
    nodes = list(G.nodes())
    print(G)
    print(f"Illicit nodes: {len(illicit)}")
    print(f"Licit nodes: {len(licit)}")
    print(f"Unknown nodes: {len(unknown)}")

    rows = []

    # S0
    f0 = S0_oracle_high_illicit(G, illicit, k=k, seed=SEED)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f0, r=r, illicit=illicit, licit=licit)
    rows.append(("S0", k, r, V, E, I, IR, L, LIR))
    '''
    # S0 induced subgraph
    H = G.subgraph(f0).copy()
    V, E, I, IR, L, LIR = metrics(H, illicit, licit)
    rows.append(("S0 induced", k, r, V, E, I, IR, L, LIR))

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
    '''
    # S4
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    V, E, I, IR, L, LIR = spnsa_metrics(G, f4, r=r, illicit=illicit, licit=licit)
    rows.append(("S4", k, r, V, E, I, IR, L, LIR))

    # S4 induced subgraph
    H = G.subgraph(f4).copy()
    V, E, I, IR, L, LIR = metrics(H, illicit, licit)
    rows.append(("S4 induced", k, r, V, E, I, IR, L, LIR))

    df = pd.DataFrame(rows, columns=["strategy", "k", "r", "|V|", "|E|", "|I|", "IR", "L", "LIR"])
    print("\n=== Feed strategies (S0–S4):", DATASET_NAME, "===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
