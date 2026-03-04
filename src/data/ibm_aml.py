from __future__ import annotations

from typing import Iterable, Set, Tuple
import networkx as nx


def build_graph_from_transactions(file_path: str) -> nx.DiGraph:
    """Build a directed graph from an IBM-AML *_Trans.csv file.

    parse the CSV as text and use columns:
    - From Bank: p[4]
    - To Bank:   p[2]
    """
    G = nx.DiGraph()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            p = [x.strip() for x in line.split(",")]
            v = p[2]
            w = p[4]
            G.add_edge(v, w)
    return G


def build_criminal_graph_from_transactions(file_path: str) -> nx.DiGraph:
    """Build the subgraph consisting only of laundering edges (label==1).

    laundering label is column p[10] == '1'.
    """
    G = nx.DiGraph()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            p = [x.strip() for x in line.split(",")]
            if p[10] == "1":
                v = p[2]
                w = p[4]
                G.add_edge(v, w)
    return G


def illicit_nodes_from_transactions(file_path: str) -> Set[str]:
    """Return the set of nodes that participate in any laundering edge (label==1)."""
    illicit_nodes: Set[str] = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            p = [x.strip() for x in line.split(",")]
            if p[10] == "1":
                illicit_nodes.add(p[2])
                illicit_nodes.add(p[4])
    return illicit_nodes


def illicit_edges_from_transactions(file_path: str) -> Set[Tuple[str, str]]:
    """Return the set of (src,dst) edges whose label is laundering (label==1)."""
    illicit_edges: Set[Tuple[str, str]] = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            p = [x.strip() for x in line.split(",")]
            if p[10] == "1":
                illicit_edges.add((p[2], p[4]))
    return illicit_edges
