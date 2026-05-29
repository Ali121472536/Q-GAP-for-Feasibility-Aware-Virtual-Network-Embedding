import networkx as nx
import random


class VNRGenerator:
    def __init__(self, min_nodes=3, max_nodes=8):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def generate_vnr(self):
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        graph = nx.connected_watts_strogatz_graph(
            n=num_nodes,
            k=min(4, num_nodes - 1),
            p=0.3
        )

        for node in graph.nodes:
            graph.nodes[node]['cpu'] = random.randint(10, 30)

        for u, v in graph.edges:
            graph[u][v]['bw'] = random.randint(20, 80)

        return graph