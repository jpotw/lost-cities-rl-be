"""PPO (Proximal Policy Optimization) agent implementation."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor
from torch import device as Device

from models.model import LostCitiesNet

# from models.model2 import LostCitiesNet


class PPOAgent:
    """PPO agent for training and playing Lost Cities.

    This agent uses the Proximal Policy Optimization algorithm to learn optimal play.
    It maintains a policy network that outputs both action probabilities and value
    estimates.

    Attributes:
        device: Device to run computations on (CPU/GPU).
        model: Neural network model.
        optimizer: Adam optimizer for model training.
        gamma: Discount factor for future rewards.
        clip_ratio: PPO clipping parameter.
        entropy_coef: Coefficient for entropy bonus.
        value_loss_coef: Coefficient for value loss.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[Device] = None,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            state_size: Size of the input state vector.
            action_size: Size of the action space.
            hidden_size: Size of hidden layers in the network.
            lr: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            clip_ratio: PPO clipping parameter.
            entropy_coef: Coefficient for entropy bonus.
            value_loss_coef: Coefficient for value loss.
            max_grad_norm: Maximum gradient norm for clipping.
            device: Device to run computations on (CPU/GPU).
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LostCitiesNet(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def select_action(
        self,
        state: Union[Tensor, NDArray[np.float32]],
        valid_actions: List[Tuple[int, int, int]],
    ) -> Tuple[Tuple[int, int, int], int, Tensor]:
        """Choose an action based on the current policy.

        Args:
            state: Current game state.
            valid_actions: List of valid actions in the current state.

        Returns:
            Tuple containing:
                - Decoded action (card_index, play_or_discard, draw_source)
                - Action index in the action space
                - Log probability of the selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(self.device)

        with torch.no_grad():  # No gradients needed for action selection
            policy_logits, _ = self.model(state.unsqueeze(0))

        # Create a mask for valid actions
        mask = (
            torch.ones_like(policy_logits) * -1e9
        )  # Initialize with large negative values
        for action_index in range(policy_logits.size(1)):
            if any(
                list(action) == list(self.decode_action(action_index))
                for action in valid_actions
            ):
                mask[0, action_index] = 0
        # Apply the mask
        masked_logits = policy_logits + mask

        # Softmax to get probabilities
        probs = F.softmax(masked_logits, dim=-1)

        # Sample an action
        m = torch.distributions.Categorical(probs)
        action_index = m.sample()
        return (
            self.decode_action(int(action_index.item())),  # Ensure int type
            int(action_index.item()),  # Ensure int type
            m.log_prob(action_index),
        )

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decode action index to game action components.

        Args:
            action_index: Index in the flattened action space.

        Returns:
            Tuple containing:
                - Card index in hand
                - Play (0) or discard (1)
                - Draw source (0 for deck, 1-6 for discard piles)
        """
        n_action_per_card = 7 * 2
        card_index = action_index // n_action_per_card
        remainder = action_index % n_action_per_card
        play_or_discard = remainder // 7
        draw_source = remainder % 7
        return (card_index, play_or_discard, draw_source)

    def encode_action(self, action: Tuple[int, int, int]) -> int:
        """Encode game action components to action index.

        Args:
            action: Tuple of (card_index, play_or_discard, draw_source).

        Returns:
            int: Index in the flattened action space.
        """
        card_index, play_or_discard, draw_source = action
        n_action_per_card = 7 * 2
        return card_index * n_action_per_card + play_or_discard * 7 + draw_source

    def compute_returns(self, rewards: Sequence[float]) -> Tensor:
        """Compute discounted returns from rewards.

        Args:
            rewards: Sequence of rewards in chronological order.

        Returns:
            Tensor: Discounted returns for each timestep.
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(
        self,
        states: Union[Sequence[Union[NDArray[np.float32], Tensor]], Tensor],
        actions: Union[Sequence[int], Tensor],
        old_log_probs: Union[Sequence[float], Tensor],
        returns: Tensor,
        advantages: Tensor,
    ) -> float:
        """Update the agent's policy using PPO.

        Args:
            states: States visited during rollout.
            actions: Actions taken during rollout.
            old_log_probs: Log probabilities of actions when they were taken.
            returns: Discounted returns for each timestep.
            advantages: Advantage estimates for each timestep.

        Returns:
            float: Total loss value from the update.
        """
        # Convert the data to tensors
        if not isinstance(states, Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        if not isinstance(actions, Tensor):
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        if not isinstance(old_log_probs, Tensor):
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(
                self.device
            )
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Forward pass
        policy_logits, values = self.model(states)
        values = values.squeeze()

        # Get the policy and value from the model
        probs = F.softmax(policy_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(actions)  # actions is now guaranteed to be a Tensor

        # PPO loss calculation
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (for exploration)
        entropy_loss = -m.entropy().mean()

        # Total loss
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()
