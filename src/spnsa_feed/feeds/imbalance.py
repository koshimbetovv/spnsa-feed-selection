from __future__ import annotations

from typing import Dict, Hashable, Iterable, Literal, List, Sequence, Tuple

import networkx as nx


def top_nodes_by_degree_imbalance(
    G: nx.DiGraph,
    top_k: int = 5,
    num_components: int = 20,
    component_type: Literal["weak", "strong"] = "weak",
    require_full: bool = True,
) -> List[Tuple[Hashable, int]]:
    """Select nodes with the largest in/out degree imbalance per component.

    This is a utility used in your exploratory notebooks for picking potential
    suspicious anchors in *multiple* connected components.

    The function:
    1) Takes the `num_components` largest (weak/strong) components of a directed graph.
    2) Within each component, ranks nodes by ``abs(out_degree - in_degree)`` (descending).
    3) Returns the top `top_k` nodes per component, **including the imbalance score**.

    Parameters
    ----------
    G:
        Directed graph (DiGraph / MultiDiGraph).
    top_k:
        How many nodes to return per component.
    num_components:
        How many largest components to consider.
    component_type:
        `"weak"` uses weakly connected components, `"strong"` uses strongly connected components.
    require_full:
        If True, raises if there are fewer than `num_components` components or a component is smaller
        than `top_k`.

    Returns
    -------
    selected:
        A flat list of ``(node, imbalance_score)`` tuples of length ``top_k * num_components`` (unless
        `require_full=False` and the graph/components are smaller).
    """
    if top_k <= 0 or num_components <= 0:
        raise ValueError("top_k and num_components must be positive integers.")
    if not G.is_directed():
        raise TypeError("G must be a directed NetworkX graph (e.g., DiGraph/MultiDiGraph).")

    comp_fn = nx.weakly_connected_components if component_type == "weak" else nx.strongly_connected_components

    comps = list(comp_fn(G))
    comps.sort(key=len, reverse=True)
    comps = comps[:num_components]

    if require_full and len(comps) < num_components:
        raise ValueError(
            f"Graph has only {len(comps)} {component_type} component(s), "
            f"but num_components={num_components} was requested."
        )

    chosen: List[Tuple[Hashable, int]] = []
    for idx, comp_nodes in enumerate(comps):
        H = G.subgraph(comp_nodes)
        nodes = list(H.nodes())

        if require_full and len(nodes) < top_k:
            raise ValueError(f"Component #{idx} has only {len(nodes)} node(s), but top_k={top_k}.")

        out_deg = dict(H.out_degree())
        in_deg = dict(H.in_degree())
        scores = {n: abs(out_deg[n] - in_deg[n]) for n in nodes}

        # Stable ranking: sort by (-score, original_position)
        order = sorted(range(len(nodes)), key=lambda i: (-scores[nodes[i]], i))
        chosen.extend((nodes[i], scores[nodes[i]]) for i in order[:top_k])

    return chosen


def criminals_per_largest_components(
    G: nx.DiGraph,
    criminal_nodes: Iterable[Hashable],
    num_components: int = 20,
    component_type: Literal["weak", "strong"] = "weak",
    require_full: bool = False,
) -> Dict[str, int]:
    """Count how many `criminal_nodes` appear in each of the largest components.

    Parameters
    ----------
    G:
        Directed graph.
    criminal_nodes:
        Nodes flagged as illicit/suspicious.
    num_components:
        Number of largest components to evaluate.
    component_type:
        `"weak"` or `"strong"` connected components.
    require_full:
        If True, raises when the graph has fewer than `num_components` components.

    Returns
    -------
    counts:
        Mapping like ``{"1st largest comp": 12, "2nd largest comp": 5, ...}``.
    """
    if num_components <= 0:
        raise ValueError("num_components must be a positive integer.")
    if not G.is_directed():
        raise TypeError("G must be a directed NetworkX graph (e.g., DiGraph/MultiDiGraph).")

    criminals = set(criminal_nodes)

    comp_fn = nx.weakly_connected_components if component_type == "weak" else nx.strongly_connected_components
    comps = list(comp_fn(G))
    comps.sort(key=len, reverse=True)

    if require_full and len(comps) < num_components:
        raise ValueError(
            f"Graph has only {len(comps)} {component_type} component(s), "
            f"but num_components={num_components} was requested."
        )

    comps = comps[:num_components]

    def ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 13:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    out: Dict[str, int] = {}
    for idx, comp_nodes in enumerate(comps, start=1):
        out[f"{ordinal(idx)} largest comp"] = sum(1 for v in comp_nodes if v in criminals)
    return out
