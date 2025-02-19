import torch
import torch.nn as nn
import torch.nn.functional as F

class LostCitiesNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(LostCitiesNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # Use third layer
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value