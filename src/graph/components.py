from __future__ import annotations

from typing import List
import networkx as nx


def largest_component(G: nx.Graph, component_type: str = "weak") -> nx.Graph:
    """Return the largest component as an induced subgraph (copied).

    - Undirected graphs: uses `nx.connected_components`.
    - Directed graphs: uses weakly connected components by default, or strongly connected components
      if `component_type="strong"`.

    """
    if G.number_of_nodes() == 0:
        return G.copy()

    if G.is_directed():
        if component_type not in {"weak", "strong"}:
            raise ValueError("component_type must be 'weak' or 'strong' for directed graphs.")
        comps = nx.weakly_connected_components(G) if component_type == "weak" else nx.strongly_connected_components(G)
    else:
        comps = nx.connected_components(G)

    largest_cc = max(comps, key=len)
    return G.subgraph(largest_cc).copy()


