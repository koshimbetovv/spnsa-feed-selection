from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

import networkx as nx
from tqdm import tqdm


def spnsa(G, AC, radius=1, weight=None):
    """
    Shortest Paths Network Search Algorithm (SPNSA).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph (can be directed or undirected).
    AC : iterable
        List of feed nodes (nodes of interest). Nodes must exist in G.
    radius : int, optional
        Ego network radius in hops. Default 1 (one-hop ego network).
    weight : str or None, optional
        Edge attribute for weighted shortest path. If None, unweighted shortest paths used.
    
    Returns
    -------
    subgraph : networkx.Graph or networkx.DiGraph
        A new graph composed of nodes/edges from all collected shortest paths. The returned graph
        type matches G (directedness).
    paths_by_ego : dict
        Nested dictionary with recorded paths for every ego, useful for debugging/visualization.

    Notes
    -----
    - If eigenvector centrality fails (e.g., power iteration issues), code falls back to degree centrality.
    - Nodes or paths that do not exist or where no path exists are skipped (no exception).
    """

    if not isinstance(AC, (list, set, tuple)):
        AC = list(AC)

    # Ensure AC nodes exist in G
    AC = [n for n in AC if n in G]
    if len(AC) == 0:
        raise ValueError("No AC nodes found in G.")

    # Prepare results
    paths_by_ego = {}
    # We'll build the resulting graph by adding edges from paths
    if G.is_directed():
        R = nx.DiGraph()
    else:
        R = nx.Graph()

    # Helper to safely compute eigenvector centrality with fallback
    def compute_eigenvector_centrality(subG):
        #try:
        # for directed graphs, optionally use undirected version for eigenvector
        #if subG.is_directed():
        #    cg = subG.to_undirected()
        #else:
        #    cg = subG
        subG
        if not subG.is_directed():
            # if undirected use faster version of calculation
            if subG.number_of_nodes() < 10:
                ev = nx.eigenvector_centrality(subG, max_iter=2000)
                return ev
            else:
                # Use numpy-based eigenvector for reliability
                ev = nx.eigenvector_centrality_numpy(subG)
                return ev
        else:
            # numpy version doesnt work on directed graph
            ev = nx.eigenvector_centrality(subG, max_iter=3000)
            return ev   
        '''
        if subG.number_of_nodes() < 10:
            ev = nx.eigenvector_centrality(subG, max_iter=2000)
            return ev
        else:
            # Use numpy-based eigenvector for reliability
            ev = nx.eigenvector_centrality_numpy(cg)
            return ev
        '''
        #except Exception:
        #    # fallback: degree centrality
        #    print(ego_net)
        #    print('fallback in eigenvector_centrality: moved to degree centrality')
        #    return nx.degree_centrality(subG)


    for ego in tqdm(AC):
        # extract ego network
        try:
            ego_net = nx.ego_graph(G, ego, radius=radius, center=True)

        except Exception as e:
            # if something goes wrong, skip this ego
            paths_by_ego[ego] = {'error': str(e)}
            continue

        if ego_net.number_of_nodes() == 0:
            paths_by_ego[ego] = {'error': 'empty ego network'}
            continue

        # Compute betweenness centrality and eigenvector centrality inside the ego network
        # For betweenness, use unnormalized betweenness restricted to the ego network
        try:
            bet = nx.betweenness_centrality(ego_net, normalized=True)  # dict
            bet.pop(ego)
        except Exception:
            print('occured error in betweenness_centrality')
            bet = {n: 0.0 for n in ego_net.nodes()}

        ev = compute_eigenvector_centrality(ego_net)
        ev.pop(ego)
        
        # choose MM (highest betweenness), MI (highest eigenvector)
        MM = max(bet.items(), key=lambda kv: (kv[1], kv[0]))[0] if len(bet) else None
        MI = max(ev.items(), key=lambda kv: (kv[1], kv[0]))[0] if len(ev) else None
        
        
        
        # OC = other feed nodes present in this ego network (but excluding ego itself)
        OC = [n for n in AC if (n != ego and n in ego_net)]

        # store paths found for this ego
        ego_paths = {
            'MM': MM,
            'MI': MI,
            'ego_to_centers': {},
            'ego_to_OC': {},
            'OC_to_centers': defaultdict(dict)  # OC -> {'to_MM': path, 'to_MI': path}
        }

        # helper to get shortest path and add to result graph R
        def safe_add_path(u, v, dest_key=None, storage_dict=None):
            try:
                #p = nx.shortest_path(G, source=u, target=v, weight=weight)
                p = nx.shortest_path(ego_net, source=u, target=v, weight=weight)
                # add path edges to R
                if len(p) >= 2:
                    for a, b in zip(p[:-1], p[1:]):
                        #if G.has_edge(a, b):
                        if ego_net.has_edge(a, b):
                            #R.add_edge(a, b, **(G.get_edge_data(a, b) or {}))
                            R.add_edge(a, b, **(ego_net.get_edge_data(a, b) or {}))
                        #else:
                        #    R.add_edge(a, b)
                else:
                    R.add_node(u)

                # store path if dictionary provided
                if storage_dict is not None and dest_key is not None:
                    storage_dict[dest_key] = p
                return p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                if storage_dict is not None and dest_key is not None:
                    storage_dict[dest_key] = None
                return None


        # 1) ego -> MM and ego -> MI
        if MM is not None:
            pmm = safe_add_path(ego, MM, dest_key='MM', storage_dict=ego_paths['ego_to_centers'])
        else:
            ego_paths['ego_to_centers']['MM'] = None

        if MI is not None:
            pmi = safe_add_path(ego, MI, dest_key='MI', storage_dict=ego_paths['ego_to_centers'])
        else:
            ego_paths['ego_to_centers']['MI'] = None

        # 2) ego -> each OC
        for oc in OC:
            poc = safe_add_path(ego, oc, dest_key=oc, storage_dict=ego_paths['ego_to_OC'])

        # 3) each OC -> centers (MM, MI)
        for oc in OC:
            if MM is not None:
                ego_paths['OC_to_centers'][oc]['to_MM'] = safe_add_path(oc, MM, storage_dict=None)
            if MI is not None:
                ego_paths['OC_to_centers'][oc]['to_MI'] = safe_add_path(oc, MI, storage_dict=None)


        paths_by_ego[ego] = ego_paths

    # Also include any isolated nodes that were in AC but never added via paths
    for n in AC:
        if n in G and n not in R:
            R.add_node(n)

    # Return the subgraph (graph built by edges) and paths_by_ego
    return R, paths_by_ego


