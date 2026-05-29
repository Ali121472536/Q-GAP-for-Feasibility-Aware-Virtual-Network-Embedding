import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPOAgent:

    def __init__(

            self,
            actor,
            critic,
            learning_rate=0.001,
            gamma=0.99

    ):

        self.actor = actor
        self.critic = critic

        self.gamma = gamma

        self.optimizer_actor = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate
        )

        self.optimizer_critic = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate
        )

    # ==================================================
    # Select Action
    # ==================================================
    def select_action(self, state):

        probs = self.actor(state)

        distribution = torch.distributions.Categorical(
            probs
        )
        entropy = distribution.entropy()

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        return (
            action.item(),
            log_prob,
            entropy
        )

    # ==================================================
    # PPO Update
    # ==================================================
    def update(

            self,
            log_probs,
            rewards,
            values

    ):

        returns = []

        discounted_reward = 0

        # --------------------------------------
        # Compute Returns
        # --------------------------------------
        for reward in reversed(rewards):

            discounted_reward = (
                    reward +
                    self.gamma * discounted_reward
            )

            returns.insert(
                0,
                discounted_reward
            )

        # --------------------------------------
        # Stack Critic Values
        # --------------------------------------
        values = torch.stack(
            values
        ).squeeze()

        # --------------------------------------
        # Move Returns to Same Device
        # --------------------------------------
        returns = torch.tensor(
            returns,
            dtype=torch.float32
        ).to(values.device)

        # --------------------------------------
        # Stack Log Probabilities
        # --------------------------------------
        log_probs = torch.stack(
            log_probs
        ).to(values.device)

        # --------------------------------------
        # Advantage
        # --------------------------------------
        advantages = (
            returns - values.detach()
        )

        # --------------------------------------
        # Actor Loss
        # --------------------------------------
        actor_loss = -(
            log_probs * advantages
        ).mean()

        # --------------------------------------
        # Critic Loss
        # --------------------------------------
        critic_loss = F.mse_loss(
            values,
            returns
        )

        # --------------------------------------
        # Update Actor
        # --------------------------------------
        self.optimizer_actor.zero_grad()

        actor_loss.backward(retain_graph=True)

        self.optimizer_actor.step()

        # --------------------------------------
        # Update Critic
        # --------------------------------------
        self.optimizer_critic.zero_grad()

        critic_loss.backward()

        self.optimizer_critic.step()

        # --------------------------------------
        # Return Losses
        # --------------------------------------
        return (
            actor_loss.item(),
            critic_loss.item()
        )