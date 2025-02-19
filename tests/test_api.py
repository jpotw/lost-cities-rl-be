import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_get_ai_move():
    response = client.post(
        "/get_ai_move",
        json={
            "players": [
                {
                    "id": "player1",
                    "name": "Player",
                    "type": "human",
                    "hand": [
                        {"id": 1, "color": "red", "value": 5, "isHidden": False}
                    ],
                    "expeditions": {
                        "red": [{"id": 2, "color": "red", "value": 3, "isHidden": False}]
                    },
                    "score": 10
                }
            ],
            "currentPlayerIndex": 0,
            "deck": [
                {"id": 3, "color": "blue", "value": 7, "isHidden": True}
            ],
            "discardPiles": {
                "red": [{"id": 4, "color": "red", "value": 2, "isHidden": False}]
            },
            "selectedCard": None,
            "gamePhase": "playing",
            "isAIThinking": False,
            "lastDiscarded": None,
            "winner": None
        }
    )
    assert response.status_code == 200
    assert response.json() == {"action": [0, 0, 0]}