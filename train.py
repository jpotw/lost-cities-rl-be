"""Training script for the Lost Cities reinforcement learning agent."""

from typing import Optional, Union

import numpy as np
import torch
from torch import device as Device

from game.lost_cities_env import LostCitiesEnv
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent

# Hyperparameters
STATE_SIZE = 265
ACTION_SIZE = (
    8 * 2 * 7
)  # 8 cards × 2 actions (play/discard) × 7 draw sources (deck + 6 discard piles)
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


def load_model(
    model_path: str, device: Optional[Union[str, Device]] = None
) -> Optional[PPOAgent]:
    """Load a trained model from the specified path.

    Args:
        model_path: Path to the saved model file.
        device: Device to load the model on ('cuda' or 'cpu').

    Returns:
        PPOAgent: Loaded agent if successful, None otherwise.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

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
            device=device,
        )
        agent.model = model
        return agent

    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def train(
    num_episodes: int = NUM_GAMES,
    batch_size: int = BATCH_SIZE,
    save_interval: int = SAVE_INTERVAL,
    model_path: str = "model_final.pth",
    device: Optional[Union[str, Device]] = None,
) -> None:
    """Train the Lost Cities agent using PPO.

    Args:
        num_episodes: Number of episodes to train for.
        batch_size: Number of episodes per batch.
        save_interval: How often to save model checkpoints.
        model_path: Path to save the trained model.
        device: Device to train on ('cuda' or 'cpu').
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    env = LostCitiesEnv()
    agent = PPOAgent(
        STATE_SIZE,
        ACTION_SIZE,
        HIDDEN_SIZE,
        LEARNING_RATE,
        GAMMA,
        CLIP_RATIO,
        ENTROPY_COEF,
        VALUE_LOSS_COEF,
        device=device,
    )

    # Initialize metrics
    metrics = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_games': 0,
        'total_score': 0,
        'episode_rewards': [],
        'win_rate_history': [],
        'avg_score_history': []
    }

    for game in range(num_episodes):
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

        # Update metrics
        metrics['total_games'] += 1
        metrics['episode_rewards'].append(sum(rewards))
        metrics['total_score'] += sum(rewards)
        
        if env.winner == 0:  # AI wins
            metrics['wins'] += 1
        elif env.winner == 1:  # AI loses
            metrics['losses'] += 1
        else:  # Draw
            metrics['draws'] += 1
            
        # Calculate running statistics
        win_rate = metrics['wins'] / metrics['total_games']
        avg_score = metrics['total_score'] / metrics['total_games']
        metrics['win_rate_history'].append(win_rate)
        metrics['avg_score_history'].append(avg_score)

        returns = agent.compute_returns(rewards)
        advantages = (
            returns
            - agent.model(
                torch.tensor(np.array(states), dtype=torch.float32).to(agent.device)
            )[1]
            .squeeze()
            .detach()
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update in batches
        for _ in range(NUM_EPOCHS):
            for i in range(0, len(states), batch_size):
                batch_states = states[i : i + batch_size]
                batch_actions = actions[i : i + batch_size]
                batch_log_probs = log_probs[i : i + batch_size]
                batch_returns = returns[i : i + batch_size]
                batch_advantages = advantages[i : i + batch_size]
                loss = agent.update(
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_returns,
                    batch_advantages,
                )

        if (game + 1) % PRINT_INTERVAL == 0:
            print(
                f"Game: {game + 1}, Loss: {loss:.4f}, "
                f"Reward: {sum(rewards):.2f}, Winner: {env.winner}\n"
                f"Win Rate: {win_rate:.2%}, Avg Score: {avg_score:.2f}\n"
                f"W/L/D: {metrics['wins']}/{metrics['losses']}/{metrics['draws']}"
            )
        if (game + 1) % save_interval == 0:
            torch.save(agent.model.state_dict(), f"model_{game + 1}.pth")
            # Save metrics
            np.savez(f"metrics_{game + 1}.npz",
                win_rate_history=np.array(metrics['win_rate_history']),
                avg_score_history=np.array(metrics['avg_score_history']),
                episode_rewards=np.array(metrics['episode_rewards'])
            )
    
    torch.save(agent.model.state_dict(), model_path)
    # Save final metrics
    np.savez("metrics_final.npz",
        win_rate_history=np.array(metrics['win_rate_history']),
        avg_score_history=np.array(metrics['avg_score_history']),
        episode_rewards=np.array(metrics['episode_rewards'])
    )


if __name__ == "__main__":
    train()
