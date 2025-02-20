import torch
import torch.nn as nn
import torch.nn.functional as F

class LostCitiesNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, dropout_prob=0.5):
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

    def forward(self, x):
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