import networkx as nx


class Routing:

    @staticmethod
    def shortest_path(substrate_graph, source, target):
        try:
            path = nx.shortest_path(
                substrate_graph,
                source=source,
                target=target,
                weight='latency'
            )
            return path

        except nx.NetworkXNoPath:
            return None