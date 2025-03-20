"""FastAPI backend for Lost Cities AI game with PPO agent integration."""

import os
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from game.lost_cities_env import LostCitiesEnv
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent
from train import load_model

app = FastAPI()

# Global variables for model and environment
api_instance = None


class Card(BaseModel):
    """A card in the Lost Cities game with color and value attributes."""

    id: Optional[str] = None
    color: str
    value: Union[str, int]  # "HS" or number 2-10
    isHidden: Optional[bool] = None


class Player(BaseModel):
    """A player in the Lost Cities game with their current game state."""

    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    hand: List[Card]
    expeditions: Dict[str, List[Card]]
    score: Optional[int] = None


class GameState(BaseModel):
    """The complete state of a Lost Cities game."""

    players: List[Player]
    currentPlayerIndex: Optional[int] = None
    currentPlayer: Optional[int] = None  # For backward compatibility
    deck: Optional[List[Card]] = None
    discardPiles: Dict[str, List[Card]]
    selectedCard: Optional[Card] = None
    gamePhase: Optional[str] = None
    isAIThinking: Optional[bool] = None
    lastDiscarded: Optional[Dict] = None
    winner: Optional[Union[str, int]] = None


class AIMoveResponse(BaseModel):
    """Response model for AI move predictions."""

    action: tuple[int, int, int]


# Constants
NUM_COLORS = 6
NUM_VALUES = 11  # 0 (handshake) and 2-10 (number cards)
STATE_SIZE = 265  # Match the saved model's state size
ACTION_SIZE = (
    8 * 2 * 7
)  # 8 cards × 2 actions (play/discard) × 7 draw sources (main deck + 6 discard piles)
HIDDEN_SIZE = 256  # Hidden layer size for the neural network

# Create model and ensure it's on CPU for testing
device = torch.device("cpu")  # Use CPU for testing
model = LostCitiesNet(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
agent = PPOAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, device=device)
agent.model = model  # Set the agent's model.


class LostCitiesAPI:
    """Main API class for handling Lost Cities game logic and AI interactions."""

    def __init__(self, model_path="model_final.pth"):
        """Initialize the API with a model path.

        Args:
            model_path: Path to the trained model file.
        """
        self.agent = None
        self.env = LostCitiesEnv()
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model and initialize the agent.

        Args:
            model_path: Path to the trained model file.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        # Try to load the model, if not found, generate a dummy model
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, generating dummy model...")
            model = LostCitiesNet(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
            torch.save(model.state_dict(), model_path)
            print(f"Dummy model saved to {model_path}")

        self.agent = load_model(model_path)
        if self.agent is not None:
            print("Agent loaded successfully with the following configuration:")
            print(f"Model architecture: {self.agent.model}")
            print(f"Device: {self.agent.device}")
            print(f"Model is in eval mode: {not self.agent.model.training}")
        else:
            print(
                "Failed to load agent. Please ensure the model has been trained first."
            )
        return self.agent is not None

    def get_action(self, state, valid_actions):
        """Get the next action from the model.

        Args:
            state: Current game state.
            valid_actions: List of valid actions.

        Returns:
            tuple: Selected action.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.agent is None:
            raise RuntimeError("Model not loaded. Please load the model first.")
        action, action_index, _ = self.agent.select_action(state, valid_actions)
        return action

    def reset_environment(self):
        """Reset the environment and return initial state.

        Returns:
            Initial environment state.
        """
        return self.env.reset()

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Action to take.

        Returns:
            tuple: (state, reward, done, info).
        """
        return self.env.step(action)

    def initialize_game(self) -> GameState:
        """Initialize a new game and return the game state.

        Returns:
            GameState: Initial game state.
        """
        _ = self.env.reset()  # Reset environment but we don't use the raw state

        # Convert environment state to GameState format
        game_state = GameState(
            players=[
                Player(
                    id="player1",
                    name="Human",
                    type="HUMAN",
                    hand=[
                        Card(
                            id=f"p1_card_{i}",
                            color=self.env.index_to_color(card[0]),
                            value="HS" if card[1] == 0 else str(card[1]),
                        )
                        for i, card in enumerate(self.env.get_player_hand(0))
                    ],
                    expeditions={
                        color: []
                        for color in [
                            "RED",
                            "BLUE",
                            "GREEN",
                            "WHITE",
                            "YELLOW",
                            "PURPLE",
                        ]
                    },
                    score=0,
                ),
                Player(
                    id="player2",
                    name="AI",
                    type="AI",
                    hand=[
                        Card(
                            id=f"p2_card_{i}",
                            color=self.env.index_to_color(card[0]),
                            value="HS" if card[1] == 0 else str(card[1]),
                        )
                        for i, card in enumerate(self.env.get_player_hand(1))
                    ],
                    expeditions={
                        color: []
                        for color in [
                            "RED",
                            "BLUE",
                            "GREEN",
                            "WHITE",
                            "YELLOW",
                            "PURPLE",
                        ]
                    },
                    score=0,
                ),
            ],
            discardPiles={
                color: []
                for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
            },
            currentPlayerIndex=0,
            currentPlayer=0,
            deck=[
                Card(id=f"deck_{i}", color="HIDDEN", value="0")
                for i in range(len(self.env.deck))
            ],  # Use HIDDEN color and "0" value for deck cards
            gamePhase="PLAY",
            isAIThinking=False,
            winner=None,
        )

        return game_state


@app.on_event("startup")
async def startup_event():
    """Initialize the API instance on application startup."""
    global api_instance
    api_instance = LostCitiesAPI()


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "Lost Cities AI API is running"}


@app.post("/reset")
def reset_game():
    """Reset the game to its initial state.

    Returns:
        dict: Initial game state.
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    state = api_instance.reset_environment()
    return {"state": state.tolist() if hasattr(state, "tolist") else state}


@app.post("/step")
def take_step(action: Dict[str, Any]):
    """Take a step in the game with the given action.

    Args:
        action: Action to take.

    Returns:
        dict: New state, reward, done flag, and additional info.
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    state, reward, done, info = api_instance.step(action)
    return {
        "state": state.tolist() if hasattr(state, "tolist") else state,
        "reward": float(reward),
        "done": done,
        "info": info,
    }


@app.post("/get_action")
def get_model_action(state: List[float], valid_actions: List[List[int]]):
    """Get the AI's action for the current state.

    Args:
        state: Current game state.
        valid_actions: List of valid actions.

    Returns:
        dict: Selected action.
    """
    if api_instance is None:
        raise HTTPException(status_code=500, detail="API not initialized")
    action = api_instance.get_action(state, valid_actions)
    return {"action": action}


@app.post("/get_ai_move", response_model=AIMoveResponse)
async def get_ai_move(game_state: GameState):
    """Get the AI's next move based on the current game state.

    This endpoint converts the frontend game state to the environment format,
    processes it through the AI agent, and returns the selected action.

    Args:
        game_state: Current state of the game from the frontend.

    Returns:
        AIMoveResponse: The AI's selected action as a tuple of
            (card_index, action_type, draw_source).

    Raises:
        HTTPException: If the agent is not initialized or other errors occur.
    """
    try:
        # Convert game state and get AI move
        env_state, valid_actions = _convert_game_state_to_env(game_state)

        # Get action from agent
        if api_instance is None or api_instance.agent is None:
            raise HTTPException(status_code=500, detail="Agent not initialized")

        decoded_action, action_index, _ = api_instance.agent.select_action(
            env_state, valid_actions
        )

        # Return the action
        return AIMoveResponse(action=decoded_action)

    except Exception as e:
        print(f"Error in /get_ai_move: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _convert_game_state_to_env(
    game_state: GameState,
) -> tuple[torch.Tensor, List[tuple[int, int, int]]]:
    """Convert frontend game state to environment format.

    Args:
        game_state: Game state from the frontend.

    Returns:
        tuple: (environment state tensor, valid actions list)
    """
    env = LostCitiesEnv()
    env_state = env.reset()

    # Set current player
    current_player = (
        game_state.currentPlayer
        if game_state.currentPlayer is not None
        else game_state.currentPlayerIndex
    )
    if current_player is None:
        current_player = 0
    env.current_player = current_player

    # Clear initial state
    env.player_hands.fill(0)
    env.expeditions.fill(0)
    env.discard_piles.fill(0)

    # Convert player hands and expeditions
    for player_idx, player in enumerate(game_state.players):
        _convert_player_state(env, player, player_idx)

    # Convert discard piles
    _convert_discard_piles(env, game_state.discardPiles)

    # Get the current state and valid actions
    env_state = env._get_state()
    valid_actions = env.get_valid_actions()
    env_state = torch.from_numpy(env_state).float().to(device)

    return env_state, valid_actions


def _convert_player_state(env: LostCitiesEnv, player: Player, player_idx: int) -> None:
    """Convert a player's state to environment format.

    Args:
        env: Game environment.
        player: Player state from frontend.
        player_idx: Index of the player.
    """
    # Convert hand
    for card in player.hand:
        if card.color == "HIDDEN":
            continue
        value = 0 if card.value == "HS" else int(card.value)
        env.player_hands[player_idx][color_to_index(card.color)][value] += 1

    # Convert expeditions
    for color, cards in player.expeditions.items():
        for card in cards:
            if card.color == "HIDDEN":
                continue
            value = 0 if card.value == "HS" else int(card.value)
            env.expeditions[player_idx][color_to_index(color)][value] += 1


def _convert_discard_piles(
    env: LostCitiesEnv, discard_piles: Dict[str, List[Card]]
) -> None:
    """Convert discard piles to environment format.

    Args:
        env: Game environment.
        discard_piles: Discard piles from frontend.
    """
    for color, cards in discard_piles.items():
        for card in cards:
            if card.color == "HIDDEN":
                continue
            value = 0 if card.value == "HS" else int(card.value)
            env.discard_piles[color_to_index(color)][value] += 1


@app.post("/start_game")
async def start_game():
    """Start a new game and return the initial game state.

    Returns:
        GameState: Initial game state.

    Raises:
        HTTPException: If API is not initialized or other errors occur.
    """
    try:
        if api_instance is None:
            raise HTTPException(status_code=500, detail="API not initialized")
        return api_instance.initialize_game()
    except Exception as e:
        print(f"Error in /start_game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def color_to_index(color: str) -> int:
    """Convert color string to index.

    Args:
        color: Color name as string.

    Returns:
        int: Color index.

    Raises:
        ValueError: If color is unknown.
    """
    # Handle special case for hidden cards
    if color.upper() == "HIDDEN":
        return 0  # Default to first color for hidden cards

    colors_upper = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
    colors_lower = ["red", "blue", "green", "white", "yellow", "purple"]

    if color.upper() in colors_upper:
        return colors_upper.index(color.upper())
    elif color.lower() in colors_lower:
        return colors_lower.index(color.lower())
    else:
        raise ValueError(f"Unknown color: {color}")


def index_to_color(index: int) -> str:
    """Convert index to color string.

    Args:
        index: Color index.

    Returns:
        str: Color name.
    """
    colors = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
    return colors[index]


def value_to_index(value: Union[str, int]) -> int:
    """Convert card value to index.

    Args:
        value: Card value ("HS" or number).

    Returns:
        int: Value index.
    """
    if value == "HS":
        return 0
    return int(value)


def index_to_value(index: int) -> Union[str, int]:
    """Convert index to card value.

    Args:
        index: Value index.

    Returns:
        Union[str, int]: "HS" for handshake or number value.
    """
    if index == 0:
        return "HS"
    return index


# For testing the API directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
