import numpy as np
from game.lost_cities_env import LostCitiesEnv
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent
import torch

# Hyperparameters 
STATE_SIZE = 221
ACTION_SIZE = 8 * 2 * 6
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.02
VALUE_LOSS_COEF = 0.5
NUM_GAMES = 20000
BATCH_SIZE = 2048
NUM_EPOCHS = 10
PRINT_INTERVAL = 100
SAVE_INTERVAL = 1000 

def load_model(model_path="model_final.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with same architecture as training
    model = LostCitiesNet(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {model_path}")
        
        # Create agent with loaded model
        agent = PPOAgent(
            STATE_SIZE, 
            ACTION_SIZE, 
            HIDDEN_SIZE,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            clip_ratio=CLIP_RATIO,
            entropy_coef=ENTROPY_COEF,
            value_loss_coef=VALUE_LOSS_COEF,
            device=device
        )
        agent.model = model
        return agent
        
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def train():
    env = LostCitiesEnv()
    agent = PPOAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LEARNING_RATE, GAMMA, CLIP_RATIO, ENTROPY_COEF, VALUE_LOSS_COEF)

    for game in range(NUM_GAMES):
        state = env.reset()
        done = False
        states, actions, log_probs, rewards = [], [], [], []
        while not done:
            valid_actions = env.get_valid_actions()
            action, action_index, log_prob = agent.select_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action_index)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = agent.compute_returns(rewards)
        advantages = returns - agent.model(torch.tensor(np.array(states),dtype=torch.float32).to(agent.device))[1].squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update in batches
        for _ in range(NUM_EPOCHS):
          for i in range(0, len(states), BATCH_SIZE):
            batch_states = states[i:i + BATCH_SIZE]
            batch_actions = actions[i:i + BATCH_SIZE]
            batch_log_probs = log_probs[i:i + BATCH_SIZE]
            batch_returns = returns[i:i + BATCH_SIZE]
            batch_advantages = advantages[i:i + BATCH_SIZE]
            loss = agent.update(batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)

        if (game + 1) % PRINT_INTERVAL == 0:
            print(f"Game: {game + 1}, Loss: {loss:.4f}, Reward: {sum(rewards):.2f}, Winner : {env.winner}")
        if (game + 1) % SAVE_INTERVAL == 0:
          torch.save(agent.model.state_dict(), f"model_{game + 1}.pth")
    torch.save(agent.model.state_dict(), "model_final.pth")

if __name__ == "__main__":
    train()