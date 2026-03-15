
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.elliptic import build_criminal_graph

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
    centers=7,
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


def load_graph_and_illicit():
    """Load graph and illicit node set for the dataset."""
    classes_csv = Path("data/elliptic/elliptic_txs_classes.csv")
    edgelist_csv = Path("data/elliptic/elliptic_txs_edgelist.csv")
    if not classes_csv.exists():
        raise FileNotFoundError(f"Missing file: {classes_csv} (place it under data/elliptic/)")
    if not edgelist_csv.exists():
        raise FileNotFoundError(f"Missing file: {edgelist_csv} (place it under data/elliptic/)")
    from src.data.elliptic import build_graph_from_edgelist, illicit_nodes_from_classes
    G = build_graph_from_edgelist(str(edgelist_csv), directed=True)
    G_criminal = build_criminal_graph(str(classes_csv), str(edgelist_csv), illicit_label="1", directed=True)
    print(G_criminal)
    #illicit = illicit_nodes_from_classes(str(classes_csv), illicit_label="1")
    #illicit = set(v for v in illicit if v in G)
    illicit = set(G_criminal.nodes())
    illicit_edges = set(G_criminal.edges())
    return G, illicit, illicit_edges


def metrics(H: nx.Graph, illicit: set, illicit_edges: set):
    nodes = set(H.nodes())
    edges = set(H.edges())
    V = int(H.number_of_nodes())
    E = int(H.number_of_edges())
    I = float(len(nodes & illicit))
    IR = float(I / V) if V > 0 else 0.0
    I_E = float(len(edges & illicit_edges))
    IR_E = float(I_E / E) if E > 0 else 0.0
    COMPONENTS = int(nx.number_weakly_connected_components(H))
    return V, E, I, IR, I_E, IR_E, COMPONENTS


def spnsa_metrics(G: nx.DiGraph, feed, r: int, illicit: set, illicit_edges: set):
    H, _ = spnsa(G, feed, radius=r)
    return metrics(H, illicit, illicit_edges)

import networkx as nx

def knn(G, F, r):
    """
    Return the union of the individual r-hop neighborhood graphs
    of nodes in F.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph.
    F : iterable
        Nodes whose r-hop neighborhoods will be taken.
    r : int
        Hop radius.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Graph obtained by union of the separate r-hop neighborhood graphs.
    """
    if r < 0:
        raise ValueError("r must be nonnegative")

    F = set(F)

    missing = F - set(G.nodes)
    if missing:
        raise ValueError(f"These nodes are not in G: {missing}")

    neighborhood_graphs = [
        nx.ego_graph(G, n=node, radius=r)
        for node in F
    ]

    return nx.compose_all(neighborhood_graphs)

def main():
    G, illicit, illicit_edges = load_graph_and_illicit()
    nodes = list(G.nodes())

    rows = []

    # knn
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    H = knn(G, f4, r=r)
    V, E, I, IR, I_E, IR_E, COMPONENTS = metrics(H, illicit, illicit_edges)
    rows.append(("S4 knn subgraph", k, r, V, E, I, IR, I_E, IR_E, COMPONENTS))

    # S4 feed illicit ratio
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    H = G.subgraph(f4).copy()
    V, E, I, IR, I_E, IR_E, COMPONENTS = metrics(H, illicit, illicit_edges)
    rows.append(("S4 induced subgraph", k, r, V, E, I, IR, I_E, IR_E, COMPONENTS))

    # S4
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    V, E, I, IR, I_E, IR_E, COMPONENTS = spnsa_metrics(G, f4, r=r, illicit=illicit, illicit_edges=illicit_edges)
    rows.append(("S4 SPNSA subgraph", k, r, V, E, I, IR, I_E, IR_E, COMPONENTS))

    df = pd.DataFrame(rows, columns=["strategy", "k", "r", "|V|", "|E|", "|I|", "IR", "|I_E|", "IR_E", "COMPONENTS"])
    print("\n=== Feed strategies (S0–S4):", DATASET_NAME, "===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
