from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.spnsa import spnsa
from src.feeds.motif import motif_scores, coherent_topk_feed


def run_once(
    G: nx.DiGraph,
    illicit_nodes: set,
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

    # ---- stage 1: proxy + candidates ----
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    proxy = {v: in_deg[v] * out_deg[v] for v in G.nodes()}
    candidates = sorted(G.nodes(), key=lambda v: (-proxy[v], str(v)))[:C]

    # ---- stage 2: motif scoring ----
    scores, _details = motif_scores(G, candidates, max_in=max_in, max_out=max_out, **motif_params)
    ranked = sorted(scores.keys(), key=lambda v: (-scores[v], str(v)))

    # ---- stage 3: feed selection ----
    feed = coherent_topk_feed(G, ranked, k=k, centers=centers, d_max=d_max)

    # ---- stage 4: SPNSA ----
    SPN, _paths = spnsa(G, feed, radius=radius)

    n_components = nx.number_connected_components(SPN.to_undirected())
    U = SPN.to_undirected()
    cc_sizes = [len(c) for c in nx.connected_components(U)]
    lcc_frac = (max(cc_sizes) / SPN.number_of_nodes()) if cc_sizes else 0.0

    spn_nodes = set(SPN.nodes())
    illicit = len(spn_nodes & illicit_nodes)
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
        "illicit_ratio": ratio
    }



