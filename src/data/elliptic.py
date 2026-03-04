from __future__ import annotations

from typing import Set
import networkx as nx
import pandas as pd


def build_graph_from_edgelist(edgelist_csv: str, directed: bool = True) -> nx.Graph:
    """Build a NetworkX graph from the Elliptic edgelist CSV using original tx ids.

    Expected columns: `txId1`, `txId2`.
    """
    df = pd.read_csv(edgelist_csv, usecols=["txId1", "txId2"], dtype=str)
    df["txId1"] = df["txId1"].str.strip()
    df["txId2"] = df["txId2"].str.strip()

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(zip(df["txId1"].tolist(), df["txId2"].tolist()))
    return G


def illicit_nodes_from_classes(classes_csv: str, illicit_label: str = "1") -> Set[str]:
    """Return illicit txIds (original string ids). Expected columns: `txId`, `class`."""
    df = pd.read_csv(classes_csv, usecols=["txId", "class"], dtype=str)
    df["txId"] = df["txId"].str.strip()
    df["class"] = df["class"].str.strip()
    return set(df.loc[df["class"] == str(illicit_label), "txId"].tolist())


def build_criminal_graph(classes_csv: str, edgelist_csv: str, illicit_label: str = "1", directed: bool = True) -> nx.Graph:
    """Induced subgraph on illicit nodes: keep only illicit nodes and edges between them."""
    illicit_ids = illicit_nodes_from_classes(classes_csv, illicit_label=illicit_label)

    e = pd.read_csv(edgelist_csv, usecols=["txId1", "txId2"], dtype=str)
    e["txId1"] = e["txId1"].str.strip()
    e["txId2"] = e["txId2"].str.strip()

    e_f = e[e["txId1"].isin(illicit_ids) & e["txId2"].isin(illicit_ids)]

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(illicit_ids)
    G.add_edges_from(zip(e_f["txId1"].tolist(), e_f["txId2"].tolist()))
    nx.set_node_attributes(G, {tid: 1 for tid in illicit_ids}, "y")
    return G
