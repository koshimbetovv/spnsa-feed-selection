from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple, Optional

import networkx as nx


def motif_scores(
    G: nx.DiGraph,
    candidates: Iterable[Hashable],
    max_in: int | None = None,
    max_out: int | None = None,
    # weights for motifs
    alpha: float = 1.0,   # reciprocity
    beta: float = 0.3,    # relay proxy
    gamma: float = 0.8,   # hub penalty
    # star-like motifs
    s_in: float = 0.4,    # fan-in star strength
    s_out: float = 0.4,   # fan-out star strength
    s_leaf: float = 0.6,  # star where neighbors are low-degree leaves
    star_ratio: float = 3.0,      # require in/out dominance for "pure" star signal
    leaf_deg_max: int = 2,        # neighbor considered "leaf-like" if total degree <= this
):
    """Compute motif-based anomaly scores for candidate nodes in a directed graph.

    The scoring is **label-free** and aggregates local structural signals:

    - Reciprocity: ``|In(v) ∩ Out(v)|`` (2-cycles)
    - Directed 3-cycles through v: count of ``u→v→w`` with ``w→u``
    - Relay proxy: ``|In(v)| * |Out(v)|``
    - Star motifs:
        * fan-in dominance (many in, few out)
        * fan-out dominance (many out, few in)
        * leaf-star: number of low-degree neighbors
    - Hub penalty: penalize very large degree nodes

    Score:
    ``score(v) = log(1+cycle3) + alpha*log(1+recip) + beta*log(1+relay)
                + s_in*log(1+fanin_star) + s_out*log(1+fanout_star) + s_leaf*log(1+leaf_star)
                - gamma*log(1+deg)``

    Returns
    -------
    scores:
        Mapping ``node -> score``.
    details:
        Raw motif counts for debugging and ablation.
    """
    candidates = list(candidates)

    # Precompute degrees for leaf detection (use total degree in directed sense)
    total_deg = dict(G.degree())  # in+out for DiGraph

    # Precompute neighborhood sets for candidates
    succ = {v: set(G.successors(v)) for v in candidates}
    pred = {v: set(G.predecessors(v)) for v in candidates}

    scores: Dict[Hashable, float] = {}
    details: Dict[Hashable, Dict[str, int]] = {}

    for v in candidates:
        inN = pred[v]
        outN = succ[v]

        # Optional neighborhood capping for scalability (deterministic: keeps highest-degree neighbors)
        if max_in is not None and len(inN) > max_in:
            in_list = sorted(inN, key=lambda u: (-total_deg[u], str(u)))[:max_in]
            inN = set(in_list)

        if max_out is not None and len(outN) > max_out:
            out_list = sorted(outN, key=lambda u: (-total_deg[u], str(u)))[:max_out]
            outN = set(out_list)

        indeg = len(inN)
        outdeg = len(outN)
        deg = indeg + outdeg

        recip = len(inN & outN)
        relay = indeg * outdeg

        cycle3 = 0
        for w in outN:
            cycle3 += len(set(G.successors(w)) & inN)

        fanin_star = indeg if indeg >= star_ratio * (outdeg + 1) else 0
        fanout_star = outdeg if outdeg >= star_ratio * (indeg + 1) else 0

        leaf_in = sum(1 for u in inN if total_deg.get(u, 0) <= leaf_deg_max)
        leaf_out = sum(1 for u in outN if total_deg.get(u, 0) <= leaf_deg_max)
        leaf_star = leaf_in + leaf_out

        score = (
            0.2 + math.log1p(cycle3)
            + alpha * math.log1p(recip)
            + beta * math.log1p(relay)
            + s_in * math.log1p(fanin_star)
            + s_out * math.log1p(fanout_star)
            + s_leaf * math.log1p(leaf_star)
            - gamma * math.log1p(deg)
        )

        scores[v] = score
        details[v] = {
            "cycle3": int(cycle3),
            "recip": int(recip),
            "relay": int(relay),
            "deg": int(deg),
            "fanin_star": int(fanin_star),
            "fanout_star": int(fanout_star),
            "leaf_star": int(leaf_star),
            "leaf_in": int(leaf_in),
            "leaf_out": int(leaf_out),
        }

    return scores, details


def coherent_topk_feed(
    G_lcc: nx.DiGraph,
    ranked_nodes: Sequence[Hashable],
    k: int = 200,
    centers: int = 2,
    d_max: int = 3,
) -> List[Hashable]:
    """Select a locality-constrained feed from a ranked node list.

    Choose a few top-ranked **centers**, collect nodes
    within undirected distance `d_max` around each center, then fill the feed with the highest
    ranked nodes that lie inside these balls.

    This aims to make feeds more "coherent" for SPNSA (overlapping ego-nets).
    """
    U = G_lcc.to_undirected()

    feed: List[Hashable] = []
    used = set()

    anchor_nodes = list(ranked_nodes[:centers])

    for c in anchor_nodes:
        if c in used:
            continue

        dist = nx.single_source_shortest_path_length(U, c, cutoff=d_max)
        ball = set(dist.keys())

        for v in ranked_nodes:
            if v in used:
                continue
            if v in ball:
                feed.append(v)
                used.add(v)
                if len(feed) >= k:
                    return feed

    for v in ranked_nodes:
        if v not in used:
            feed.append(v)
            used.add(v)
            if len(feed) >= k:
                break

    return feed


def S4_motif_based_coherent(
    G: nx.DiGraph,
    k: int = 200,
    C: int = 100_000,
    centers: int = 5,
    d_max: int = 2,
    motif_params: Optional[dict] = None,
    max_in: int = 500,
    max_out: int = 500,
) -> List[Hashable]:
    '''Motif-based coherent feed selection (S4).'''
    
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

    return feed