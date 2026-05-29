import networkx as nx
import random


class SubstrateNetwork:
    def __init__(self, num_nodes, link_prob):
        self.graph = nx.erdos_renyi_graph(num_nodes, link_prob)
        self.initialize_resources()

    def initialize_resources(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]['cpu'] = random.randint(50, 100)

        for u, v in self.graph.edges:
            self.graph[u][v]['bw'] = random.randint(100, 300)
            self.graph[u][v]['latency'] = random.uniform(1, 10)

    def get_graph(self):
        return self.graph