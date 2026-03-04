from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple
import random
import networkx as nx

def S0_oracle_high_illicit(G: nx.DiGraph, illicit: set, k: int, seed: int) -> List[Hashable]:
    '''Rank nodes by their number of connections to known illicit nodes and take top-k (oracle).'''
    G_criminal = G.subgraph(illicit)
    illicit_degree = dict(G_criminal.degree())
    pool = sorted(illicit_degree.items(), key=lambda x: x[1], reverse=True)
    feed = [node_name for (node_name, degree) in pool[:k]]
                
    return feed

def S1_random(nodes: Sequence[Hashable], k: int, seed: int | None = None) -> List[Hashable]:
    """Uniformly sample `k` distinct nodes from a node list."""
    rng = random.Random(seed)
    if k <= 0:
        return []
    if k > len(nodes):
        raise ValueError("k larger than the number of available nodes.")
    return rng.sample(list(nodes), k)


def S2_largest_degree_difference(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Top-k nodes by |in_degree - out_degree| (descending)."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    degree_diff = {v: abs(in_deg.get(v, 0) - out_deg.get(v, 0)) for v in G.nodes()}
    ranked = sorted(degree_diff.items(), key=lambda x: (-x[1], str(x[0])))
    return [v for v, _ in ranked[:k]]

def S3_mixed_collect_distribute(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Union of top-(k/2) (in-degree>0, out-degree=0) and top-(k/2) (in-degree=0, out-degree>0) nodes."""
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    nonzero_in_degree = {
    node: in_degree.get(node, 0)
    for node in G.nodes() if out_degree.get(node, 0) == 0
    }

    nonzero_out_degree = {
    node: out_degree.get(node, 0)
    for node in G.nodes() if in_degree.get(node, 0) == 0
    }

    nonzero_in_degree_largest = dict(sorted(nonzero_in_degree.items(), key=lambda x: x[1], reverse=True)[:k//2])
    nonzero_out_degree_largest = dict(sorted(nonzero_out_degree.items(), key=lambda x: x[1], reverse=True)[:k//2])
    feed = nonzero_in_degree_largest | nonzero_out_degree_largest
    
    return feed
