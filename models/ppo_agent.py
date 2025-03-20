import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.model import LostCitiesNet
# from models.model2 import LostCitiesNet

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=256, lr=1e-4, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LostCitiesNet(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, state, valid_actions):
        """
        choose action based on the policy calculated by the model
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(self.device)
        
        with torch.no_grad():  # No gradients needed for action selection
            policy_logits, _ = self.model(state.unsqueeze(0))

        # Create a mask for valid actions
        mask = torch.ones_like(policy_logits) * -1e9  # Initialize with large negative values
        for action_index in range(policy_logits.size(1)):
            if any(list(action) == list(self.decode_action(action_index)) for action in valid_actions):
                mask[0, action_index] = 0
        # Apply the mask
        masked_logits = policy_logits + mask

        # Softmax to get probabilities
        probs = F.softmax(masked_logits, dim=-1)

        # Sample an action
        m = torch.distributions.Categorical(probs)
        action_index = m.sample()
        return self.decode_action(action_index.item()), action_index.item(), m.log_prob(action_index)

    def decode_action(self, action_index):
        """
        decode the action index to card index, play or discard, draw source
        """
        #num of possible action for one card is 14 (play (yes/no) * draw_source (0~6))
        #num of card in hand at max is 8
        n_action_per_card = 7 * 2
        card_index = action_index // n_action_per_card
        remainder = action_index % n_action_per_card
        play_or_discard = remainder // 7
        draw_source = remainder % 7
        return (card_index, play_or_discard, draw_source)
    
    def encode_action(self, action):
        """
        encode the action to action index
        """
        card_index, play_or_discard, draw_source = action
        n_action_per_card = 7*2
        return card_index * n_action_per_card + play_or_discard * 7 + draw_source

    def compute_returns(self, rewards):
        """
        compute the returns from the rewards
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        core learning method where the agent's neural network gets updated based on the collected data
        """
        # convert the data to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device) # convert list to numpy array
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # forward pass
        policy_logits, values = self.model(states)
        values = values.squeeze()

        # get the policy and value from the model
        probs = F.softmax(policy_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(actions)

        # PPO loss calculation
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (for exploration)
        entropy_loss = -m.entropy().mean()

        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Apply gradient clipping

        self.optimizer.step()

        return loss.item()