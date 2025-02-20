from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent
from game.lost_cities_env import LostCitiesEnv
import torch
from typing import Union, List, Dict, Any
from train import load_model

app = FastAPI()

# Global variables for model and environment
api_instance = None

class Card(BaseModel):
    id: int
    suit: str
    value: Union[str, int]
    isHidden: bool

class PlayerState(BaseModel):
    id: str
    name: str
    type: str
    hand: list[Card]
    expeditions: dict[str, list[Card]]
    score: int

class GameState(BaseModel):
    players: list[PlayerState]
    currentPlayerIndex: int
    deck: list[Card]
    discardPiles: dict[str, list[Card]]
    selectedCard: Union[Card, None]
    gamePhase: str
    isAIThinking: Union[bool, None] = None
    lastDiscarded: Union[dict[str, Union[str, int]], None] = None # str(suit), str(handshake cards) | value(number cards)
    winner: Union[str, None] = None

class AIMoveResponse(BaseModel):
    action: tuple[int, int, int]


# Calculate state size based on environment dimensions
NUM_SUITS = 6
NUM_VALUES = 11  # 0 (handshake) and 2-10 (number cards)
STATE_SIZE = NUM_SUITS * NUM_VALUES * 4 + 1  # 4 flattened arrays (66 each) + deck size
ACTION_SIZE = 8 * 2 * 7  # 8 cards × 2 actions (play/discard) × 7 draw sources
HIDDEN_SIZE = 256

# Create model and ensure it's on CPU for testing
device = torch.device("cpu")  # Use CPU for testing
model = LostCitiesNet(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
agent = PPOAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, device=device)
agent.model = model  # Set the agent's model

class LostCitiesAPI:
    def __init__(self, model_path="model_final.pth"):
        self.agent = None
        self.env = LostCitiesEnv()
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model and initialize the agent"""
        self.agent = load_model(model_path)
        if self.agent is not None:
            print("Agent loaded successfully with the following configuration:")
            print(f"Model architecture: {self.agent.model}")
            print(f"Device: {self.agent.device}")
            print(f"Model is in eval mode: {not self.agent.model.training}")
        else:
            print("Failed to load agent. Please ensure the model has been trained first.")
        return self.agent is not None

    def get_action(self, state, valid_actions):
        """Get the next action from the model"""
        if self.agent is None:
            raise RuntimeError("Model not loaded. Please load the model first.")
        action, action_index, _ = self.agent.select_action(state, valid_actions)
        return action

    def reset_environment(self):
        """Reset the environment and return initial state"""
        return self.env.reset()

    def step(self, action):
        """Take a step in the environment"""
        return self.env.step(action)

    def initialize_game(self) -> GameState:
        """Initialize a new game and return the game state"""
        initial_state = self.env.reset()
        
        # Convert environment state to GameState format
        game_state = GameState(
            players=[
                PlayerState(
                    id="0",
                    name="Player",
                    type="HUMAN",
                    hand=[
                        Card(
                            id=i,
                            suit=self.env.index_to_suit(card[0]),
                            value="HS" if card[1] == 0 else str(card[1]),
                            isHidden=False
                        )
                        for i, card in enumerate(self.env.get_player_hand(0))
                    ],
                    expeditions={suit: [] for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]},
                    score=0
                ),
                PlayerState(
                    id="1",
                    name="AI",
                    type="AI",
                    hand=[
                        Card(
                            id=i + 8,
                            suit=self.env.index_to_suit(card[0]),
                            value="HS" if card[1] == 0 else str(card[1]),
                            isHidden=True
                        )
                        for i, card in enumerate(self.env.get_player_hand(1))
                    ],
                    expeditions={suit: [] for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]},
                    score=0
                )
            ],
            currentPlayerIndex=0,
            deck=[
                Card(
                    id=i + 16,
                    suit=self.env.index_to_suit(card[0]),
                    value="HS" if card[1] == 0 else str(card[1]),
                    isHidden=True
                )
                for i, card in enumerate(self.env.deck)
            ],
            discardPiles={suit: [] for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]},
            selectedCard=None,
            gamePhase="PLAY",
            isAIThinking=False,
            lastDiscarded=None,
            winner=None
        )
        
        return game_state

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    global api_instance
    api_instance = LostCitiesAPI()

@app.get("/")
def read_root():
    return {"status": "Lost Cities AI API is running"}

@app.post("/reset")
def reset_game():
    """Reset the game and return initial state"""
    state = api_instance.reset_environment()
    return {"state": state.tolist() if hasattr(state, 'tolist') else state}

@app.post("/step")
def take_step(action: Dict[str, Any]):
    """Take a step in the game with the given action"""
    state, reward, done, info = api_instance.step(action)
    return {
        "state": state.tolist() if hasattr(state, 'tolist') else state,
        "reward": float(reward),
        "done": done,
        "info": info
    }

@app.post("/get_action")
def get_model_action(state: List[float], valid_actions: List[List[int]]):
    """Get the AI's action for the current state"""
    action = api_instance.get_action(state, valid_actions)
    return {"action": action}

@app.post("/get_ai_move", response_model=AIMoveResponse)
async def get_ai_move(game_state: GameState):
    try:
        # 1. Convert the incoming GameState (from Next.js) to the format
        env = LostCitiesEnv()
        
        # Initialize environment with empty state
        env_state = env.reset()
        
        # Set current player
        env.current_player = game_state.currentPlayerIndex
        
        # Clear initial state
        env.player_hands.fill(0)
        env.expeditions.fill(0)
        env.discard_piles.fill(0)
        
        # Convert player hands and expeditions
        for player_idx, player in enumerate(game_state.players):
            # Convert hand
            for card in player.hand:
                value = 0 if card.value == "HS" else int(card.value)
                env.player_hands[player_idx][suit_to_index(card.suit)][value] += 1
            
            # Convert expeditions
            for suit, cards in player.expeditions.items():
                for card in cards:
                    value = 0 if card.value == "HS" else int(card.value)
                    env.expeditions[player_idx][suit_to_index(suit)][value] += 1

        # Convert discard piles
        for suit, cards in game_state.discardPiles.items():
            for card in cards:
                value = 0 if card.value == "HS" else int(card.value)
                env.discard_piles[suit_to_index(suit)][value] += 1

        # Convert deck
        env.deck = []
        for card in game_state.deck:
            value = 0 if card.value == "HS" else int(card.value)
            env.deck.append((suit_to_index(card.suit), value))

        # Get the current state
        env_state = env._get_state()

        # 2. Get the AI's move
        valid_actions = env.get_valid_actions()
        
        # Convert state to tensor and ensure it's on CPU
        env_state = torch.from_numpy(env_state).float().to(device)
        
        # Get action from agent
        decoded_action, action_index, _ = agent.select_action(env_state, valid_actions)

        # 3. Return the chosen action
        return AIMoveResponse(action=decoded_action)

    except Exception as e:
        # Log the error
        print(f"Error in /get_ai_move: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_game")
async def start_game():
    """Start a new game and return the initial game state"""
    try:
        if api_instance is None:
            raise HTTPException(status_code=500, detail="API not initialized")
        return api_instance.initialize_game()
    except Exception as e:
        print(f"Error in /start_game: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def suit_to_index(suit: str) -> int:
    """Convert suit string to index"""
    suits = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
    return suits.index(suit)

# For testing the API directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)