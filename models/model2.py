"""Alternative neural network model for Lost Cities game with residual connections."""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class LostCitiesNet(nn.Module):
    """Neural network model for Lost Cities game with residual connections.

    This model takes a game state as input and outputs both a policy (action probabilities)
    and a value estimate. The architecture consists of three fully connected layers with
    residual connections, batch normalization, and dropout, followed by separate policy
    and value heads.

    Attributes:
        dropout_prob: Dropout probability.
        fc1: First fully connected layer.
        bn1: First batch normalization layer.
        fc2: Second fully connected layer.
        bn2: Second batch normalization layer.
        fc3: Third fully connected layer.
        bn3: Third batch normalization layer.
        policy_head: Output layer for policy (action probabilities).
        value_head: Output layer for value estimate.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        dropout_prob: float = 0.5,
    ) -> None:
        """Initialize the network.

        Args:
            state_size: Size of the input state vector.
            action_size: Size of the action space.
            hidden_size: Size of hidden layers (default: 256).
            dropout_prob: Dropout probability (default: 0.5).
        """
        super(LostCitiesNet, self).__init__()

        self.dropout_prob = dropout_prob

        # First layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Second layer (residual)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # Third layer (residual)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # Heads
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize policy head with orthogonal initialization
        nn.init.orthogonal_(self.policy_head.weight, gain=1)

    def forward(
        self, x: Union[torch.Tensor, NDArray[np.float32]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input state tensor or numpy array.

        Returns:
            Tuple containing:
                - Policy logits (unnormalized action probabilities)
                - Value estimate
        """
        # Convert numpy array to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(next(self.parameters()).device)  # Move input to same device as model

        # First layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Second layer (residual)
        h = self.bn2(self.fc2(x))
        h = h + x  # residual connection
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_prob, training=self.training)
        x = h

        # Third layer (residual)
        h = self.bn3(self.fc3(x))
        h = h + x  # residual connection
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_prob, training=self.training)
        x = h

        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
