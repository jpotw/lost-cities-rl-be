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

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3, self.policy_head, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.to(next(self.parameters()).device)  # Move input to same device as model
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value