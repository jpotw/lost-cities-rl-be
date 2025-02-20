from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.model import LostCitiesNet
from models.ppo_agent import PPOAgent

app = FastAPI()

class Card(BaseModel):
    id: int
    color: str
    value: int | str
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
    selectedCard: Card | None
    gamePhase: str
    isAIThinking: bool | None = None
    lastDiscarded: dict[str, str | int] | None = None # str(color), str(handshake cards) | value(number cards)
    winner: str | None

class AIMoveResponse(BaseModel):
    action: tuple[int, int, int]


STATE_SIZE = 221
ACTION_SIZE = 8 * 2 * 7
HIDDEN_SIZE = 256

# Placeholder - create a random model for now
model = LostCitiesNet(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
agent = PPOAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
agent.model = model  # Set the agent's model

    
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/get_ai_move", response_model=AIMoveResponse)
async def get_ai_move(game_state: GameState):
    # Placeholder - return a dummy action for now
    return AIMoveResponse(action=(0, 0, 0))