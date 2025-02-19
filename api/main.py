from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

    
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/get_ai_move", response_model=AIMoveResponse)
async def get_ai_move(game_state: GameState):
    # Placeholder - return a dummy action for now
    return AIMoveResponse(action=(0, 0, 0))