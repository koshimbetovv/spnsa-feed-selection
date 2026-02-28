from __future__ import annotations

from typing import List
import networkx as nx


def largest_component(G: nx.Graph, component_type: str = "weak") -> nx.Graph:
    """Return the largest component as an induced subgraph (copied).

    - Undirected graphs: uses `nx.connected_components`.
    - Directed graphs: uses weakly connected components by default, or strongly connected components
      if `component_type="strong"`.

    This is extracted from the notebooks without changing behavior.
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


def connected_components_as_graphs(G: nx.Graph) -> List[nx.Graph]:
    """Return a list of connected components as separate graphs (undirected connectivity)."""
    if nx.number_connected_components(G) == 1:
        return [G]
    comps = []
    for component in nx.connected_components(G):
        comps.append(G.subgraph(component).copy())
    return comps


def eccentricity_distributions(components: List[nx.Graph]):
    """Compute eccentricity dict for each component (for exploratory analysis)."""
    return [nx.eccentricity(comp) for comp in components]
