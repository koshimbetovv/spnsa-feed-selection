from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple
import random
import networkx as nx


def random_feed(nodes: Sequence[Hashable], k: int, seed: int | None = None) -> List[Hashable]:
    """Uniformly sample `k` distinct nodes from a node list."""
    rng = random.Random(seed)
    if k <= 0:
        return []
    if k > len(nodes):
        raise ValueError("k larger than the number of available nodes.")
    return rng.sample(list(nodes), k)


def top_degree_imbalance_feed(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Top-k nodes by |in_degree - out_degree| (descending)."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    degree_diff = {v: abs(in_deg.get(v, 0) - out_deg.get(v, 0)) for v in G.nodes()}
    ranked = sorted(degree_diff.items(), key=lambda x: (-x[1], str(x[0])))
    return [v for v, _ in ranked[:k]]


def top_in_degree_feed(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Top-k nodes by in-degree (descending)."""
    ranked = sorted(G.in_degree(), key=lambda x: (-x[1], str(x[0])))
    return [v for v, _ in ranked[:k]]


def top_out_degree_feed(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Top-k nodes by out-degree (descending)."""
    ranked = sorted(G.out_degree(), key=lambda x: (-x[1], str(x[0])))
    return [v for v, _ in ranked[:k]]


def top_in_out_union_feed(G: nx.DiGraph, k: int) -> List[Hashable]:
    """Union of top-(k/2) in-degree and top-(k/2) out-degree nodes."""
    half = k // 2
    top_in = [v for v, _ in sorted(G.in_degree(), key=lambda x: (-x[1], str(x[0])))[:half]]
    top_out = [v for v, _ in sorted(G.out_degree(), key=lambda x: (-x[1], str(x[0])))[:half]]
    # preserve order: in first, then out
    seen = set()
    feed: List[Hashable] = []
    for v in top_in + top_out:
        if v not in seen:
            seen.add(v)
            feed.append(v)
    # if odd k, fill with remaining in-degree nodes
    if len(feed) < k:
        for v, _ in sorted(G.in_degree(), key=lambda x: (-x[1], str(x[0]))):
            if v not in seen:
                seen.add(v)
                feed.append(v)
                if len(feed) >= k:
                    break
    return feed[:k]
