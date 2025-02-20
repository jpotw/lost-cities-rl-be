from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent
from game.lost_cities_env import LostCitiesEnv
import torch
from typing import Union

app = FastAPI(root_path="/api")

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

    
@app.get("/health")
async def health_check():
    return {"status": "ok"}


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

def suit_to_index(suit: str) -> int:
    """Convert suit string to index"""
    suits = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
    return suits.index(suit)