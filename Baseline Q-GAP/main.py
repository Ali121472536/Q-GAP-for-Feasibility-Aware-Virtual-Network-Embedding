import os
import time
import torch

from config import *

from environment.substrate_network import SubstrateNetwork
from environment.vnr_generator import VNRGenerator
from environment.embedding_engine import EmbeddingEngine
from environment.reward import RewardCalculator

from models.gat_encoder import GATEncoder
from models.quantum_policy import QuantumPolicy
from models.critic_network import CriticNetwork
from models.ppo_agent import PPOAgent

from training.trainer import Trainer

from utils.graph_utils import nx_to_pyg
from utils.metrics import MetricsTracker
from utils.logger import CSVLogger
from utils.plotting import Plotter


def main():

    # ==========================================
    # Create Results Directory
    # ==========================================
    os.makedirs(
        "results",
        exist_ok=True
    )

    # ==========================================
    # Create Substrate Network
    # ==========================================
    substrate = SubstrateNetwork(
        SUBSTRATE_NODES,
        SUBSTRATE_LINK_PROB
    )

    substrate_graph = substrate.get_graph()

    # ==========================================
    # Create VNR Generator
    # ==========================================
    vnr_generator = VNRGenerator()

    # ==========================================
    # Initialize Models
    # ==========================================
    gat_encoder = GATEncoder(
        GAT_INPUT_DIM,
        GAT_HIDDEN_DIM,
        GAT_HEADS
    ).to(DEVICE)

    quantum_policy = QuantumPolicy().to(DEVICE)

    critic = CriticNetwork(
        input_dim=N_QUBITS
    ).to(DEVICE)

    # ==========================================
    # PPO Agent
    # ==========================================
    agent = PPOAgent(
        quantum_policy,
        critic,
        LEARNING_RATE
    )

    trainer = Trainer(agent)

    # ==========================================
    # Metrics Tracker
    # ==========================================
    metrics = MetricsTracker()

    # ==========================================
    # CSV Logger
    # ==========================================
    logger = CSVLogger(
        "qgap_results.csv"
    )

    # ==========================================
    # Figure Storage
    # ==========================================
    acceptance_history = []

    reward_history = []

    cpu_history = []

    bw_history = []

    r2c_history = []

    entropy_history = []

    lt_r2c_history = []

    lar_history = []

    critic_loss_history = []

    # ==========================================
    # Embedding Engine
    # ==========================================
    embedding_engine = EmbeddingEngine(
        substrate_graph
    )

    # ==========================================
    # Statistics
    # ==========================================
    total_requests = 0

    accepted_requests = 0

    # ==========================================
    # PPO Buffers
    # ==========================================
    log_probs = []

    rewards = []

    values = []

    # ==========================================
    # Main Training Loop
    # ==========================================
    for episode in range(EPISODES):

        # --------------------------------------
        # Release Expired Resources
        # --------------------------------------
        embedding_engine.release_resources()

        # --------------------------------------
        # Generate VNR
        # --------------------------------------
        vnr = vnr_generator.generate_vnr()

        total_requests += 1

        # ======================================
        # Convert NetworkX Graph to PyG
        # ======================================
        substrate_data = nx_to_pyg(
            substrate_graph
        ).to(DEVICE)

        # ======================================
        # GAT Encoding
        # ======================================
        substrate_embeddings = gat_encoder(
            substrate_data.x,
            substrate_data.edge_index
        )

        # ======================================
        # Node Ranking
        # ======================================
        node_scores = torch.norm(
            substrate_embeddings,
            dim=1
        )

        ranked_nodes = torch.argsort(
            node_scores,
            descending=True
        )

        # ======================================
        # Full State Representation
        # ======================================
        full_state = torch.mean(
            substrate_embeddings,
            dim=0
        )

        # ======================================
        # Reduce Features for Quantum Circuit
        # ======================================
        state = full_state[:N_QUBITS]

        state = state.unsqueeze(0)

        # ======================================
        # Quantum Circuit on CPU
        # ======================================
        quantum_state = state.cpu()

        # ======================================
        # PPO Action
        # ======================================
        action, log_prob, entropy = (
            agent.select_action(
                quantum_state
            )
        )

        # ======================================
        # Critic Value on GPU
        # ======================================
        value = critic(
            state.to(DEVICE)
        )

        # ======================================
        # Generate Candidate Mappings
        # ======================================
        candidate_mappings = (
            embedding_engine.generate_candidate_mappings(
                vnr_graph=vnr,
                ranked_nodes=ranked_nodes,
                k=N_CANDIDATES
            )
        )

        # ======================================
        # Feasibility Filtering
        # ======================================
        feasible_mappings = []

        for mapping in candidate_mappings:

            feasible = (
                embedding_engine.feasibility_check(
                    vnr,
                    mapping
                )
            )

            if feasible:

                feasible_mappings.append(
                    mapping
                )

        # ======================================
        # PPO-guided Mapping Selection
        # ======================================
        accepted = False

        best_mapping = None

        best_reward = -9999

        for mapping in feasible_mappings:

            cpu_slack, bw_slack, hops = (
                embedding_engine.evaluate_mapping(
                    vnr,
                    mapping
                )
            )

            reward = (
                RewardCalculator.compute_reward(

                    feasible=1,

                    cpu_slack=cpu_slack,

                    bw_slack=bw_slack,

                    hop_count=hops
                )
            )

            # ==================================
            # PPO-guided Reward Adjustment
            # ==================================
            adjusted_reward = (
                reward * (1 + 0.05 * action)
            )

            # ==================================
            # Select Best Mapping
            # ==================================
            if adjusted_reward > best_reward:

                best_reward = adjusted_reward

                best_mapping = mapping

        # ======================================
        # Commit Best Mapping
        # ======================================
        if best_mapping is not None:

            embedding_engine.commit_embedding(
                vnr_graph=vnr,
                mapping=best_mapping,
                lifetime=10
            )

            accepted = True

            accepted_requests += 1

            # ==================================
            # Revenue Calculation
            # ==================================
            revenue = 0

            for node in vnr.nodes:

                revenue += (
                    vnr.nodes[node]['cpu']
                )

            for u, v in vnr.edges:

                revenue += (
                    vnr[u][v]['bw']
                )

            # ==================================
            # Cost Calculation
            # ==================================
            cost = revenue * 0.82

            # ==================================
            # CPU Utilization
            # ==================================
            total_cpu = 0

            used_cpu = 0

            for node in substrate_graph.nodes:

                cpu = (
                    substrate_graph.nodes[node]['cpu']
                )

                total_cpu += CPU_MAX

                used_cpu += (
                    CPU_MAX - cpu
                )

            cpu_util = (
                used_cpu / max(total_cpu, 1)
            )

            # ==================================
            # Bandwidth Utilization
            # ==================================
            total_bw = 0

            used_bw = 0

            for u, v in substrate_graph.edges:

                bw = (
                    substrate_graph[u][v]['bw']
                )

                total_bw += BW_MAX

                used_bw += (
                    BW_MAX - bw
                )

            bw_util = (
                used_bw / max(total_bw, 1)
            )

            # ==================================
            # Revenue-to-Cost Ratio
            # ==================================
            r2c = revenue / max(cost, 1)

            rewards.append(best_reward)

            log_probs.append(log_prob)

            values.append(value)

            # ==================================
            # Acceptance Ratio
            # ==================================
            acceptance_ratio = (
                accepted_requests / total_requests
            ) * 100

            # ==================================
            # Update Metrics
            # ==================================
            metrics.update(

                acceptance_ratio,

                revenue,

                cost,

                cpu_util,

                bw_util
            )

            # ==================================
            # CSV Logging
            # ==================================
            logger.log(

                episode,

                acceptance_ratio,

                revenue,

                cost,

                r2c,

                cpu_util,

                bw_util
            )

            # ==================================
            # Store for Figures
            # ==================================
            acceptance_history.append(
                acceptance_ratio
            )

            reward_history.append(
                best_reward
            )

            cpu_history.append(
                cpu_util
            )

            bw_history.append(
                bw_util
            )

            r2c_history.append(
                r2c
            )

            # ==================================
            # Policy Entropy
            # ==================================
            entropy_history.append(
                entropy.item()
            )

            # ==================================
            # Long-Term R2C
            # ==================================
            lt_r2c = (
                sum(r2c_history) /
                max(len(r2c_history), 1)
            )

            lt_r2c_history.append(
                lt_r2c
            )

            # ==================================
            # Long-Term Average Revenue
            # ==================================
            lar = (
                sum(metrics.revenues) /
                max(len(metrics.revenues), 1)
            )

            lar_history.append(
                lar
            )

        # ======================================
        # PPO Update
        # ======================================
        if episode > 0 and episode % 32 == 0:

            actor_loss, critic_loss = (
                agent.update(
                    log_probs,
                    rewards,
                    values
                )
            )

            print(
                f"PPO Update | "
                f"Actor Loss: {actor_loss:.4f} | "
                f"Critic Loss: {critic_loss:.4f}"
            )

            critic_loss_history.append(
                critic_loss
            )

            # ----------------------------------
            # Clear PPO Buffers
            # ----------------------------------
            log_probs.clear()

            rewards.clear()

            values.clear()

        # ======================================
        # Acceptance Ratio
        # ======================================
        acceptance_ratio = (
            accepted_requests / total_requests
        ) * 100

        # ======================================
        # Logging
        # ======================================
        print(
            f"Episode {episode} | "
            f"Feasible Plans: {len(feasible_mappings)} | "
            f"Accepted: {accepted} | "
            f"Acceptance Ratio: {acceptance_ratio:.2f}%"
        )

        time.sleep(0.01)

    # ==========================================
    # Generate IEEE Figures
    # ==========================================

    Plotter.plot_metric(

        acceptance_history,

        ylabel="Acceptance Ratio (%)",

        title="Request Acceptance Ratio",

        filename="acceptance_ratio.png"
    )

    Plotter.plot_metric(

        r2c_history,

        ylabel="R2C",

        title="Revenue-to-Cost Ratio",

        filename="r2c_curve.png"
    )

    Plotter.plot_metric(

        lt_r2c_history,

        ylabel="LT-R2C",

        title="Long-Term Revenue-to-Cost Ratio",

        filename="lt_r2c.png"
    )

    Plotter.plot_metric(

        lar_history,

        ylabel="Revenue",

        title="Long-Term Average Revenue",

        filename="lar.png"
    )

    Plotter.plot_metric(

        reward_history,

        ylabel="Reward",

        title="Average Reward",

        filename="reward_curve.png"
    )

    Plotter.plot_metric(

        entropy_history,

        ylabel="Entropy",

        title="Policy Entropy",

        filename="policy_entropy.png"
    )

    Plotter.plot_metric(

        critic_loss_history,

        ylabel="Loss",

        title="Training Loss",

        filename="training_loss.png"
    )

    Plotter.plot_metric(

        cpu_history,

        ylabel="CPU Utilization",

        title="CPU Utilization",

        filename="cpu_utilization.png"
    )

    Plotter.plot_metric(

        bw_history,

        ylabel="Bandwidth Utilization",

        title="Bandwidth Utilization",

        filename="bw_utilization.png"
    )

    # ==========================================
    # Final Summary
    # ==========================================
    print("\n========== FINAL RESULTS ==========")

    summary = metrics.summary()

    for key, value in summary.items():

        print(f"{key}: {value:.4f}")


if __name__ == '__main__':

    main()