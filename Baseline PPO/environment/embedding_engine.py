import random
import networkx as nx


class EmbeddingEngine:

    # ==================================================
    # Initialize Embedding Engine
    # ==================================================
    def __init__(self, substrate_graph):

        self.substrate_graph = substrate_graph

        # Active VNR embeddings
        self.active_embeddings = []

    # ==================================================
    # Generate Candidate Node Mappings
    # ==================================================
    def generate_candidate_mappings(
            self,
            vnr_graph,
            ranked_nodes,
            k=10
    ):

        candidates = []

        ranked_nodes = ranked_nodes.tolist()

        for _ in range(k):

            mapping = {}

            used_nodes = set()

            ranked_copy = ranked_nodes.copy()

            random.shuffle(ranked_copy)

            for vnode in vnr_graph.nodes:

                assigned = False

                for snode in ranked_copy:

                    if snode not in used_nodes:

                        mapping[vnode] = snode

                        used_nodes.add(snode)

                        assigned = True
                        break

                if not assigned:
                    break

            if len(mapping) == len(vnr_graph.nodes):
                candidates.append(mapping)

        return candidates

    # ==================================================
    # Feasibility Check
    # ==================================================
    def feasibility_check(self, vnr_graph, mapping):

        # --------------------------------------
        # Node Resource Check
        # --------------------------------------
        for vnode, snode in mapping.items():

            required_cpu = (
                vnr_graph.nodes[vnode]['cpu']
            )

            available_cpu = (
                self.substrate_graph.nodes[snode]['cpu']
            )

            if required_cpu > available_cpu:
                return False

        # --------------------------------------
        # Link Resource Check
        # --------------------------------------
        for u, v in vnr_graph.edges:

            su = mapping[u]
            sv = mapping[v]

            try:

                path = nx.shortest_path(
                    self.substrate_graph,
                    source=su,
                    target=sv,
                    weight='latency'
                )

            except nx.NetworkXNoPath:
                return False

            required_bw = vnr_graph[u][v]['bw']

            for i in range(len(path) - 1):

                n1 = path[i]
                n2 = path[i + 1]

                available_bw = (
                    self.substrate_graph[n1][n2]['bw']
                )

                if required_bw > available_bw:
                    return False

        return True

    # ==================================================
    # Evaluate Mapping Quality
    # ==================================================
    def evaluate_mapping(self, vnr_graph, mapping):

        total_cpu_slack = 0
        total_bw_slack = 0

        total_hops = 0
        total_links = 0

        # --------------------------------------
        # CPU Slack
        # --------------------------------------
        for vnode, snode in mapping.items():

            available_cpu = (
                self.substrate_graph.nodes[snode]['cpu']
            )

            required_cpu = (
                vnr_graph.nodes[vnode]['cpu']
            )

            slack = (
                available_cpu - required_cpu
            ) / max(available_cpu, 1)

            total_cpu_slack += slack

        # --------------------------------------
        # Bandwidth + Hop Count
        # --------------------------------------
        for u, v in vnr_graph.edges:

            su = mapping[u]
            sv = mapping[v]

            try:

                path = nx.shortest_path(
                    self.substrate_graph,
                    source=su,
                    target=sv,
                    weight='latency'
                )

                total_hops += len(path) - 1
                total_links += 1

                required_bw = vnr_graph[u][v]['bw']

                for i in range(len(path) - 1):

                    n1 = path[i]
                    n2 = path[i + 1]

                    available_bw = (
                        self.substrate_graph[n1][n2]['bw']
                    )

                    slack = (
                        available_bw - required_bw
                    ) / max(available_bw, 1)

                    total_bw_slack += slack

            except nx.NetworkXNoPath:

                return 0, 0, 100

        # --------------------------------------
        # Normalize
        # --------------------------------------
        avg_cpu_slack = (
            total_cpu_slack / max(len(mapping), 1)
        )

        avg_bw_slack = (
            total_bw_slack / max(total_links, 1)
        )

        avg_hops = (
            total_hops / max(total_links, 1)
        )

        return (
            avg_cpu_slack,
            avg_bw_slack,
            avg_hops
        )

    # ==================================================
    # Commit Embedding
    # ==================================================
    def commit_embedding(
            self,
            vnr_graph,
            mapping,
            lifetime=20
    ):

        # --------------------------------------
        # Reserve Node Resources
        # --------------------------------------
        for vnode, snode in mapping.items():

            cpu_demand = (
                vnr_graph.nodes[vnode]['cpu']
            )

            self.substrate_graph.nodes[snode]['cpu'] -= (
                cpu_demand
            )

        # --------------------------------------
        # Reserve Link Resources
        # --------------------------------------
        for u, v in vnr_graph.edges:

            su = mapping[u]
            sv = mapping[v]

            path = nx.shortest_path(
                self.substrate_graph,
                source=su,
                target=sv,
                weight='latency'
            )

            bw_demand = (
                vnr_graph[u][v]['bw']
            )

            for i in range(len(path) - 1):

                n1 = path[i]
                n2 = path[i + 1]

                self.substrate_graph[n1][n2]['bw'] -= (
                    bw_demand
                )

        # --------------------------------------
        # Store Active Embedding
        # --------------------------------------
        self.active_embeddings.append({

            'mapping': mapping,

            'vnr_graph': vnr_graph,

            'release_episode': lifetime

        })

    # ==================================================
    # Release Expired Resources
    # ==================================================
    def release_resources(self):

        remaining_embeddings = []

        for embedding in self.active_embeddings:

            embedding['release_episode'] -= 1

            # --------------------------------------
            # Release Resources
            # --------------------------------------
            if embedding['release_episode'] <= 0:

                mapping = embedding['mapping']

                vnr_graph = embedding['vnr_graph']

                # ----------------------------------
                # Restore CPU
                # ----------------------------------
                for vnode, snode in mapping.items():

                    cpu_demand = (
                        vnr_graph.nodes[vnode]['cpu']
                    )

                    self.substrate_graph.nodes[snode]['cpu'] += (
                        cpu_demand
                    )

                # ----------------------------------
                # Restore Bandwidth
                # ----------------------------------
                for u, v in vnr_graph.edges:

                    su = mapping[u]
                    sv = mapping[v]

                    path = nx.shortest_path(
                        self.substrate_graph,
                        source=su,
                        target=sv,
                        weight='latency'
                    )

                    bw_demand = (
                        vnr_graph[u][v]['bw']
                    )

                    for i in range(len(path) - 1):

                        n1 = path[i]
                        n2 = path[i + 1]

                        self.substrate_graph[n1][n2]['bw'] += (
                            bw_demand
                        )

            else:

                remaining_embeddings.append(
                    embedding
                )

        self.active_embeddings = (
            remaining_embeddings
        )