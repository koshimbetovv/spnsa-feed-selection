from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def _safe_undirected_layout(G: nx.Graph, layout: str = "spring", seed: int = 42, **kwargs):
    """Compute a 2D layout on the undirected projection for stability."""
    U = G.to_undirected() if G.is_directed() else G

    if layout == "spring":
        return nx.spring_layout(U, seed=seed, **kwargs)
    if layout == "kamada":
        return nx.kamada_kawai_layout(U, **kwargs)
    if layout == "shell":
        return nx.shell_layout(U)
    if layout == "circular":
        return nx.circular_layout(U)
    return nx.spring_layout(U, seed=seed, **kwargs)


def save_spn_components(
    SPN: nx.Graph,
    illicit_nodes: Optional[Iterable] = None,
    illicit_edges: Optional[Iterable[Tuple]] = None,
    save_dir: str = "outputs/spn_components",
    prefix: str = "spn_comp",
    max_components: int = 50,
    min_size: int = 5,
    layout: str = "spring",
    seed: int = 42,
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 35,
    alpha_edges: float = 0.35,
    arrows: bool = False,
    dpi: int = 200,
    save_pdf: bool = False,
) -> List[str]:
    """Save each connected component of SPN as a separate image file.

    This is extracted from your notebook utility.

    Returns
    -------
    saved_paths:
        List of written file paths (png, and pdf if enabled).
    """
    os.makedirs(save_dir, exist_ok=True)

    illicit_nodes = set() if illicit_nodes is None else set(illicit_nodes)
    illicit_edges = set() if illicit_edges is None else set(illicit_edges)

    if SPN.is_directed():
        comps = list(nx.weakly_connected_components(SPN))
    else:
        comps = list(nx.connected_components(SPN))
    comps = sorted(comps, key=len, reverse=True)

    saved_paths: List[str] = []
    saved = 0

    for comp_nodes in comps:
        if len(comp_nodes) < min_size:
            continue
        if saved >= max_components:
            break

        H = SPN.subgraph(comp_nodes).copy()

        U = H.to_undirected() if H.is_directed() else H
        if layout == "spring":
            pos = nx.spring_layout(U, seed=seed, iterations=150)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(U)
        elif layout == "circular":
            pos = nx.circular_layout(U)
        else:
            pos = nx.spring_layout(U, seed=seed, iterations=150)

        nodes = list(H.nodes())
        node_colors = ["tab:red" if n in illicit_nodes else "lightgray" for n in nodes]

        edges = list(H.edges())
        edge_colors, edge_widths = [], []
        for (u, v) in edges:
            illicit = (u, v) in illicit_edges or (v, u) in illicit_edges
            edge_colors.append("tab:red" if illicit else "gray")
            edge_widths.append(1.2 if illicit else 0.6)

        comp_illicit = sum(1 for n in nodes if n in illicit_nodes)
        comp_ratio = comp_illicit / len(nodes)

        plt.figure(figsize=figsize)
        plt.axis("off")
        plt.title(f"Component {saved+1} (|V|={H.number_of_nodes()}, |E|={H.number_of_edges()})")

        nx.draw_networkx_edges(
            H, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=alpha_edges,
            arrows=arrows,
            arrowstyle="-|>" if arrows else None,
            arrowsize=10 if arrows else None,
            connectionstyle="arc3,rad=0.05" if arrows else "arc3,rad=0.0",
        )
        nx.draw_networkx_nodes(
            H, pos,
            node_color=node_colors,
            node_size=node_size,
            linewidths=0.0,
        )

        plt.text(
            0.01, 0.01,
            f"illicit nodes: {comp_illicit} | illicit ratio: {comp_ratio:.3f}",
            transform=plt.gca().transAxes,
        )

        fname = f"{prefix}_{saved+1:03d}_nodes{H.number_of_nodes()}_edges{H.number_of_edges()}"
        png_path = os.path.join(save_dir, fname + ".png")
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(png_path)

        if save_pdf:
            pdf_path = os.path.join(save_dir, fname + ".pdf")
            plt.savefig(pdf_path, bbox_inches="tight")
            saved_paths.append(pdf_path)

        plt.close()
        saved += 1

    print(f"Saved {saved} component figure(s) to: {os.path.abspath(save_dir)}")
    return saved_paths
