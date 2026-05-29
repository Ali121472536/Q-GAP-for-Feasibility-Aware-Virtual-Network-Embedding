import torch
from torch_geometric.data import Data


def nx_to_pyg(graph):

    node_features = []

    node_mapping = {}

    # ======================================
    # Node Features
    # ======================================
    for idx, node in enumerate(graph.nodes):

        node_mapping[node] = idx

        cpu = graph.nodes[node].get('cpu', 0)

        degree = graph.degree[node]

        feature_vector = [
            cpu,
            degree
        ]

        node_features.append(feature_vector)

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # ======================================
    # Edge Index
    # ======================================
    edge_index = []

    for u, v in graph.edges:

        edge_index.append([
            node_mapping[u],
            node_mapping[v]
        ])

        edge_index.append([
            node_mapping[v],
            node_mapping[u]
        ])

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index
    )