from __future__ import annotations

import random
import time
from typing import Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from spnsa_feed.spnsa import spnsa
from spnsa_feed.feeds.motif import motif_scores, coherent_topk_feed


def shuffle_edge_insertion_order(G: nx.DiGraph, seed: int) -> nx.DiGraph:
    """Return a copy of `G` with edges inserted in a deterministic shuffled order.

    This matches the reproducibility approach used in your notebooks.
    """
    rng = random.Random(seed)
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(list(G.nodes()))
    edges = list(G.edges(data=True))
    rng.shuffle(edges)
    H.add_edges_from(edges)
    return H


def run_once(
    G_base: nx.DiGraph,
    criminal_nodes: set,
    seed: int,
    C: int = 100_000,
    k: int = 200,
    centers: int = 5,
    d_max: int = 2,
    radius: int = 1,
    motif_params: Optional[dict] = None,
    max_in: int = 500,
    max_out: int = 500,
) -> Dict[str, float]:
    """Run the motif-based S4 pipeline once and return one-row metrics dict."""
    motif_params = motif_params or dict(alpha=0.2, star_ratio=15.0, s_in=0.3, s_out=0.4)

    random.seed(seed)
    np.random.seed(seed)

    t1 = time.perf_counter()

    # perturb adjacency order deterministically
    G = shuffle_edge_insertion_order(G_base, seed=seed)

    # ---- stage 1: proxy + candidates ----
    t2 = time.perf_counter()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    proxy = {v: in_deg[v] * out_deg[v] for v in G.nodes()}
    candidates = sorted(G.nodes(), key=lambda v: (-proxy[v], str(v)))[:C]
    t3 = time.perf_counter()

    # ---- stage 2: motif scoring ----
    scores, _details = motif_scores(G, candidates, max_in=max_in, max_out=max_out, **motif_params)
    ranked = sorted(scores.keys(), key=lambda v: (-scores[v], str(v)))
    t4 = time.perf_counter()

    # ---- stage 3: feed selection ----
    feed = coherent_topk_feed(G, ranked, k=k, centers=centers, d_max=d_max)
    t5 = time.perf_counter()

    # ---- stage 4: SPNSA ----
    SPN, _paths = spnsa(G, feed, radius=radius)
    t6 = time.perf_counter()

    n_components = nx.number_connected_components(SPN.to_undirected())
    U = SPN.to_undirected()
    cc_sizes = [len(c) for c in nx.connected_components(U)]
    lcc_frac = (max(cc_sizes) / SPN.number_of_nodes()) if cc_sizes else 0.0

    spn_nodes = set(SPN.nodes())
    illicit = len(spn_nodes & criminal_nodes)
    ratio = illicit / max(1, SPN.number_of_nodes())

    return {
        "seed": seed,
        "cand_size": len(candidates),
        "feed_size": len(feed),
        "spn_nodes": SPN.number_of_nodes(),
        "spn_edges": SPN.number_of_edges(),
        "spn_components": n_components,
        "spn_lcc_frac": lcc_frac,
        "illicit_nodes": illicit,
        "illicit_ratio": ratio,
        "t_total": t6 - t2,
        "t_candidates": t3 - t2,
        "t_motif": t4 - t3,
        "t_feed": t5 - t4,
        "t_spnsa": t6 - t5,
    }


def summarize_series(x: np.ndarray) -> Dict[str, float]:
    """Return mean/std/median/IQR/min/max summary for a numeric vector."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def run_repeated_experiment(
    G: nx.DiGraph,
    criminal_nodes: set,
    R: int = 20,
    seeds: Optional[Sequence[int]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Run `run_once` for multiple seeds and print summary stats."""
    if seeds is None:
        seeds = list(range(R))

    rows = [run_once(G, criminal_nodes, seed=s, **kwargs) for s in seeds]
    df = pd.DataFrame(rows)

    ratio_sum = summarize_series(df["illicit_ratio"].to_numpy())
    cc_sum = summarize_series(df["spn_components"].to_numpy())
    sn_sum = summarize_series(df["spn_nodes"].to_numpy())
    se_sum = summarize_series(df["spn_edges"].to_numpy())
    lcc_sum = summarize_series(df["spn_lcc_frac"].to_numpy())
    illicit_sum = summarize_series(df["illicit_nodes"].to_numpy())

    # Print concise summary (same information as notebook)
    print("Illicit ratio over runs:")
    print(f"  mean ± std : {ratio_sum['mean']:.4f} ± {ratio_sum['std']:.4f}")
    print(f"  median [IQR]: {ratio_sum['median']:.4f} [{ratio_sum['q25']:.4f}, {ratio_sum['q75']:.4f}]")
    print(f"  min..max   : {ratio_sum['min']:.4f} .. {ratio_sum['max']:.4f}")

    print("\nConnected components in SPN (undirected) over runs:")
    print(f"  mean ± std : {cc_sum['mean']:.2f} ± {cc_sum['std']:.2f}")

    print("\nNumber of nodes in SPN over runs:")
    print(f"  mean ± std : {sn_sum['mean']:.2f} ± {sn_sum['std']:.2f}")

    print("\nNumber of edges in SPN over runs:")
    print(f"  mean ± std : {se_sum['mean']:.2f} ± {se_sum['std']:.2f}")

    print("\nFraction of lcc in SPN over runs:")
    print(f"  mean ± std : {lcc_sum['mean']:.2f} ± {lcc_sum['std']:.2f}")

    print("\nNumber of illicit nodes in SPN over runs:")
    print(f"  mean ± std : {illicit_sum['mean']:.2f} ± {illicit_sum['std']:.2f}")

    # Timing summaries
    print("\nRuntime (seconds) over runs:")
    for col, name in [
        ("t_total", "Total"),
        ("t_candidates", "Candidates"),
        ("t_motif", "Motif scoring"),
        ("t_feed", "Feed selection"),
        ("t_spnsa", "SPNSA"),
    ]:
        s = summarize_series(df[col].to_numpy())
        print(f"  {name:14s}: mean±std {s['mean']:.3f}±{s['std']:.3f} | "
              f"median[IQR] {s['median']:.3f}[{s['q25']:.3f},{s['q75']:.3f}] | "
              f"min..max {s['min']:.3f}..{s['max']:.3f}")

    return df, {
        "illicit_ratio": ratio_sum,
        "spn_components": cc_sum,
        "spn_nodes": sn_sum,
        "spn_edges": se_sum,
        "spn_lcc_frac": lcc_sum,
        "illicit_nodes": illicit_sum,
        "t_total": summarize_series(df["t_total"].to_numpy()),
    }
