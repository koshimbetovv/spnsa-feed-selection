#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from spnsa_feed.spnsa import spnsa
from spnsa_feed.feeds.trivial import (
    random_feed,
    top_degree_imbalance_feed,
    top_in_out_union_feed,
)
from spnsa_feed.graph.components import largest_component
from spnsa_feed.data.ibm_aml import build_graph_from_transactions, suspicious_nodes_from_transactions
from spnsa_feed.data.elliptic import build_graph_from_edgelist, illicit_nodes_from_classes


@dataclass(frozen=True)
class Metrics:
    V: int
    E: int
    I: float
    IR: float


def _metrics(H: nx.Graph, illicit_nodes: set) -> Metrics:
    nodes = set(H.nodes())
    I = float(len(nodes & illicit_nodes))
    V = int(H.number_of_nodes())
    E = int(H.number_of_edges())
    IR = float(I / V) if V > 0 else 0.0
    return Metrics(V=V, E=E, I=I, IR=IR)


def _mean_metrics(ms: Sequence[Metrics]) -> Metrics:
    if not ms:
        return Metrics(V=0, E=0, I=0.0, IR=0.0)
    return Metrics(
        V=int(round(float(np.mean([m.V for m in ms])))),
        E=int(round(float(np.mean([m.E for m in ms])))),
        I=float(np.mean([m.I for m in ms])),
        IR=float(np.mean([m.IR for m in ms])),
    )


def _oracle_illicit_feed(available_nodes: Sequence[Hashable], illicit_nodes: set, k: int, seed: int) -> List[Hashable]:
    """S0: sample feed nodes from illicit set (oracle feed for evaluation)."""
    pool = [v for v in available_nodes if v in illicit_nodes]
    if len(pool) < k:
        raise ValueError(f"Not enough illicit nodes in the evaluated graph: need k={k}, have {len(pool)}.")
    rng = random.Random(seed)
    return rng.sample(pool, k)


def evaluate_once(G: nx.DiGraph, illicit_nodes: set, feed: List[Hashable], r: int) -> Metrics:
    H, _paths = spnsa(G, feed, radius=r)
    return _metrics(H, illicit_nodes)


def run_trivial_suite(
    G: nx.DiGraph,
    illicit_nodes: set,
    k: int,
    r: int,
    trials: int,
    seed: int,
) -> pd.DataFrame:
    """Compute S0–S3 results for a single (k,r) operating point.

    Definitions (consistent with your paper-style strategy naming):
    - S0: Oracle illicit feed (sample k nodes from illicit nodes)
    - S1: Random feed (uniform k-sample), averaged over `trials`
    - S2: Top-|in-out| feed (degree-imbalance)
    - S3: Union(top-in, top-out) feed

    Returns a dataframe with columns:
    strategy, k, r, |V|, |E|, |I|, IR
    """
    nodes = list(G.nodes())

    # S0 (oracle)
    s0_feed = _oracle_illicit_feed(nodes, illicit_nodes, k=k, seed=seed)
    s0 = evaluate_once(G, illicit_nodes, s0_feed, r=r)

    # S1 (random, averaged)
    s1_runs = []
    for t in range(trials):
        f = random_feed(nodes, k=k, seed=seed + t)
        s1_runs.append(evaluate_once(G, illicit_nodes, f, r=r))
    s1 = _mean_metrics(s1_runs)

    # S2 (imbalance)
    s2_feed = top_degree_imbalance_feed(G, k=k)
    s2 = evaluate_once(G, illicit_nodes, s2_feed, r=r)

    # S3 (top in ∪ top out)
    s3_feed = top_in_out_union_feed(G, k=k)
    s3 = evaluate_once(G, illicit_nodes, s3_feed, r=r)

    rows = [
        ("S0", k, r, s0.V, s0.E, s0.I, s0.IR),
        ("S1", k, r, s1.V, s1.E, s1.I, s1.IR),
        ("S2", k, r, s2.V, s2.E, s2.I, s2.IR),
        ("S3", k, r, s3.V, s3.E, s3.I, s3.IR),
    ]
    return pd.DataFrame(rows, columns=["strategy", "k", "r", "|V|", "|E|", "|I|", "IR"])


def _load_ibm(csv_path: str, component: str) -> Tuple[nx.DiGraph, set]:
    G = build_graph_from_transactions(csv_path)
    G = largest_component(G, component_type=component)
    illicit = suspicious_nodes_from_transactions(csv_path)
    # Keep only illicit nodes that exist in the evaluated component
    illicit = set(v for v in illicit if v in G)
    return G, illicit


def _load_elliptic(classes_csv: str, edgelist_csv: str, component: str) -> Tuple[nx.DiGraph, set]:
    G = build_graph_from_edgelist(edgelist_csv, directed=True)
    G = largest_component(G, component_type=component)
    illicit = illicit_nodes_from_classes(classes_csv, illicit_label="1")
    illicit = set(v for v in illicit if v in G)
    return G, illicit


def main():
    ap = argparse.ArgumentParser(
        description="Run trivial feed strategies (S0–S3) for SPNSA at a given (k,r). "
                    "S1 is averaged over multiple random trials."
    )
    ap.add_argument("--dataset", choices=["ibm", "elliptic"], required=True)

    # IBM-AML
    ap.add_argument("--csv", help="IBM-AML *_Trans.csv file path (required if --dataset ibm).")

    # Elliptic
    ap.add_argument("--classes", help="Elliptic classes CSV (required if --dataset elliptic).")
    ap.add_argument("--edgelist", help="Elliptic edgelist CSV (required if --dataset elliptic).")

    # Experiment params
    ap.add_argument("--k", type=int, default=200, help="Feed size k.")
    ap.add_argument("--r", type=int, default=1, help="SPNSA ego radius r.")
    ap.add_argument("--trials", type=int, default=100, help="Number of random trials for S1.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (used for S0 sampling and S1 trials).")
    ap.add_argument(
        "--component",
        choices=["weak", "strong"],
        default="weak",
        help="Which component type to take as the 'largest component' for directed graphs.",
    )
    ap.add_argument("--out", default=None, help="Optional: write CSV results to this path.")

    args = ap.parse_args()

    if args.dataset == "ibm":
        if not args.csv:
            raise SystemExit("--csv is required for --dataset ibm")
        G, illicit = _load_ibm(args.csv, component=args.component)
    else:
        if not args.classes or not args.edgelist:
            raise SystemExit("--classes and --edgelist are required for --dataset elliptic")
        G, illicit = _load_elliptic(args.classes, args.edgelist, component=args.component)

    df = run_trivial_suite(G, illicit_nodes=illicit, k=args.k, r=args.r, trials=args.trials, seed=args.seed)

    # Print in a paper-friendly format
    print(df.to_string(index=False))

    if args.out:
        out_path = args.out
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved CSV to: {out_path}")


if __name__ == "__main__":
    main()
