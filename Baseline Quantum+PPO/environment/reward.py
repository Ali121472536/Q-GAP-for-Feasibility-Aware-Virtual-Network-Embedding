from config import *


class RewardCalculator:

    @staticmethod
    def compute_reward(

            feasible,
            cpu_slack,
            bw_slack,
            hop_count

    ):

        # ======================================
        # Normalized Metrics
        # ======================================
        slack_score = (
            cpu_slack + bw_slack
        ) / 2.0

        fragmentation = 1.0 - slack_score

        hop_penalty = hop_count / 10.0

        # ======================================
        # Reward Equation
        # ======================================
        reward = (

            ALPHA * feasible +

            BETA * slack_score -

            GAMMA_REWARD * fragmentation -

            DELTA * hop_penalty

        )

        return reward