#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from spnsa_feed.data.ibm_aml import build_graph_from_transactions, suspicious_nodes_from_transactions
from spnsa_feed.graph.components import largest_component
from spnsa_feed.experiments.motif_experiment import run_repeated_experiment


def main():
    ap = argparse.ArgumentParser(description="Run motif-based S4 experiment on an IBM-AML CSV.")
    ap.add_argument("--csv", required=True, help="Path to *_Trans.csv (IBM-AML).")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--C", type=int, default=300_000)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--centers", type=int, default=5)
    ap.add_argument("--d-max", type=int, default=2)
    args = ap.parse_args()

    G = build_graph_from_transactions(args.csv)
    G = largest_component(G, component_type="weak")
    criminal_nodes = suspicious_nodes_from_transactions(args.csv)

    df, summary = run_repeated_experiment(
        G=G,
        criminal_nodes=set(criminal_nodes),
        R=args.R,
        C=args.C,
        k=args.k,
        centers=args.centers,
        d_max=args.d_max,
        radius=args.radius,
    )
    print(df.sort_values("illicit_ratio", ascending=False).head(5))


if __name__ == "__main__":
    main()
