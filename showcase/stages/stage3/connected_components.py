"""Connected Components clustering — O(n+m), deterministic."""
import networkx as nx


def cluster_cc(df, match_pairs):
    """
    Groups matched pairs into entity clusters via connected components.
    Returns list[frozenset] — every record in exactly one cluster.
    """
    G = nx.Graph()
    G.add_nodes_from(df["id"])
    G.add_edges_from(match_pairs)
    return [frozenset(c) for c in nx.connected_components(G)]
