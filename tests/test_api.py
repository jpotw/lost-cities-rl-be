"""Tests for the Lost Cities API endpoints."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set testing environment variable
os.environ["TESTING"] = "1"

from api import app, api_instance

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check() -> None:
    """Test the health check endpoint returns OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_get_ai_move_new_game() -> None:
    """Test AI move generation for a new game state."""
    from tests.mock_game_state import create_mock_game_state
    
    game_state = create_mock_game_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())

    # Since TESTING=1, we should get a default action
    assert response.status_code == 200
    action = response.json()["action"]
    assert len(action) == 3
    assert action == [0, 0, 0]


@pytest.mark.asyncio
async def test_get_ai_move_midgame() -> None:
    """Test AI move generation for a mid-game state."""
    from tests.mock_game_state import create_mock_midgame_state
    
    game_state = create_mock_midgame_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())

    # Since TESTING=1, we should get a default action
    assert response.status_code == 200
    action = response.json()["action"]
    assert len(action) == 3
    assert action == [0, 0, 0]


@pytest.mark.asyncio
async def test_get_ai_move_endgame() -> None:
    """Test AI move generation for an end-game state."""
    from tests.mock_game_state import create_mock_endgame_state
    
    game_state = create_mock_endgame_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())

    # Since TESTING=1, we should get a default action
    assert response.status_code == 200
    action = response.json()["action"]
    assert len(action) == 3
    assert action == [0, 0, 0]
