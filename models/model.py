"""Neural network model for Lost Cities game.

This model takes a game state as input and outputs both a policy (action probabilities)
and a value estimate. The architecture consists of three fully connected layers
followed by separate policy and value heads.

Attributes:
    fc1: First fully connected layer.
    fc2: Second fully connected layer.
    fc3: Third fully connected layer.
    policy_head: Output layer for policy (action probabilities).
    value_head: Output layer for value estimate.
"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class LostCitiesNet(nn.Module):
    """Neural network model for Lost Cities game.

    This model takes a game state as input and outputs both a policy (action probabilities)
    and a value estimate. The architecture consists of three fully connected layers followed
    by separate policy and value heads.

    Attributes:
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
        fc3: Third fully connected layer.
        policy_head: Output layer for policy (action probabilities).
        value_head: Output layer for value estimate.
    """

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 256
    ) -> None:
        """Initialize the network.

        Args:
            state_size: Size of the input state vector.
            action_size: Size of the action space.
            hidden_size: Size of hidden layers (default: 256).
        """
        super(LostCitiesNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3, self.policy_head, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

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
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(next(self.parameters()).device)  # Move input to same device as model

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
