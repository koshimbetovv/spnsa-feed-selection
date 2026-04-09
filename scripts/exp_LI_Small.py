
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm

from src.feeds.trivial import S0_oracle_high_illicit, S1_random, S2_largest_degree_difference, S3_mixed_collect_distribute
from src.feeds.motif import S4_motif_based_coherent
from src.spnsa import spnsa


# =========================
# CONFIG
# =========================
DATASET_NAME = "IBM-AML (LI-Small)"

# SPNSA parameters
(k,r) = (200, 1)

# S4 parameters (match your notebooks/paper defaults)
S4_PARAMS = dict(
    C=300_000,
    centers=5,
    d_max=2,
    motif_params=dict(
        alpha=0.2, beta=0.3, gamma=0.8,
        s_in=0.2, s_out=0.4, s_leaf=0.6,
        star_ratio=15.0, leaf_deg_max=2
    ),
    max_in=500,
    max_out=500,
)

S1_TRIALS = 100   # paper table: S1 averaged over 100 trials
SEED = 0


def load_graph_and_illicit():
    """Load graph and illicit node set for the dataset."""
    csv_path = Path("data/ibm-aml/LI-Small_Trans.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path} (place it under data/ibm-aml/)")
    from src.data.ibm_aml import build_graph_from_transactions, illicit_nodes_from_transactions
    G = build_graph_from_transactions(str(csv_path))
    illicit = illicit_nodes_from_transactions(str(csv_path))
    return G, illicit


def metrics(H: nx.Graph, illicit: set):
    nodes = set(H.nodes())
    V = int(H.number_of_nodes())
    E = int(H.number_of_edges())
    I = float(len(nodes & illicit))
    IR = float(I / V) if V > 0 else 0.0
    return V, E, I, IR


def spnsa_metrics(G: nx.DiGraph, feed, r: int, illicit: set):
    H, _ = spnsa(G, feed, radius=r)
    return metrics(H, illicit)


def main():
    G, illicit = load_graph_and_illicit()
    nodes = list(G.nodes())

    rows = []
    
    # S0
    f0 = S0_oracle_high_illicit(G, illicit, k=k, seed=SEED)
    V, E, I, IR = spnsa_metrics(G, f0, r=r, illicit=illicit)
    rows.append(("S0", k, r, V, E, I, IR))

    # S1 (avg over trials)
    Vs, Es, Is, IRs = [], [], [], []
    for t in tqdm(range(S1_TRIALS), desc="S1 trials"):
        f1 = S1_random(nodes, k=k, seed=SEED + t)
        H, _ = spnsa(G, f1, radius=r, verbose=False)
        v, e, i, ir = metrics(H, illicit)
        Vs.append(v); Es.append(e); Is.append(i); IRs.append(ir)
    rows.append((
        "S1", k, r,
        int(round(sum(Vs)/len(Vs))),
        int(round(sum(Es)/len(Es))),
        float(sum(Is)/len(Is)),
        float(sum(IRs)/len(IRs)),
    ))

    # S2
    f2 = S2_largest_degree_difference(G, k=k)
    V, E, I, IR = spnsa_metrics(G, f2, r=r, illicit=illicit)
    rows.append(("S2", k, r, V, E, I, IR))

    # S3
    f3 = S3_mixed_collect_distribute(G, k=k)
    V, E, I, IR = spnsa_metrics(G, f3, r=r, illicit=illicit)
    rows.append(("S3", k, r, V, E, I, IR))
    
    # S4
    f4 = S4_motif_based_coherent(G, k=k, **S4_PARAMS)
    V, E, I, IR = spnsa_metrics(G, f4, r=r, illicit=illicit)
    rows.append(("S4", k, r, V, E, I, IR))
    
    df = pd.DataFrame(rows, columns=["strategy", "k", "r", "|V|", "|E|", "|I|", "IR"])
    print("\n=== Feed strategies (S0–S4):", DATASET_NAME, "===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
