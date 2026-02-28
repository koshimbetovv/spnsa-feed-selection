import networkx as nx
from spnsa_feed.spnsa import spnsa


def test_spnsa_smoke():
    G = nx.DiGraph()
    G.add_edges_from([
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("a", "d"),
        ("d", "a"),
    ])
    feed = ["a", "c"]
    H, paths = spnsa(G, feed, radius=1)
    assert isinstance(H, nx.DiGraph)
    assert "a" in H
    assert "c" in H
    assert isinstance(paths, dict)
