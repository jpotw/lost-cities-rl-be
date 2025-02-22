import pytest
from fastapi.testclient import TestClient
from api import app
from tests.mock_game_state import create_mock_game_state, create_mock_midgame_state, create_mock_endgame_state

client = TestClient(app)

@pytest.mark.asyncio
async def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_get_ai_move_new_game():
    """Test AI move generation for a new game state"""
    game_state = create_mock_game_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())
    
    assert response.status_code == 200
    action = response.json()["action"]
    
    assert isinstance(action, list)
    assert len(action) == 3
    card_index, play_or_discard, draw_source = action
    
    assert isinstance(card_index, int)
    assert isinstance(play_or_discard, int)
    assert isinstance(draw_source, int)
    assert 0 <= card_index <= 7  # Since AI has 8 cards
    assert play_or_discard in [0, 1]  # 0 for play, 1 for discard
    assert 0 <= draw_source <= 6  # 0 for deck, 1-6 for discard piles

@pytest.mark.asyncio
async def test_get_ai_move_midgame():
    """Test AI move generation for a mid-game state"""
    game_state = create_mock_midgame_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())
    
    assert response.status_code == 200
    action = response.json()["action"]
    
    assert isinstance(action, list)
    assert len(action) == 3
    card_index, play_or_discard, draw_source = action
    
    assert isinstance(card_index, int)
    assert isinstance(play_or_discard, int)
    assert isinstance(draw_source, int)
    assert 0 <= card_index <= 7
    assert play_or_discard in [0, 1]
    assert 0 <= draw_source <= 6

@pytest.mark.asyncio
async def test_get_ai_move_endgame():
    """Test AI move generation for an end-game state"""
    game_state = create_mock_endgame_state()
    response = client.post("/get_ai_move", json=game_state.model_dump())
    
    assert response.status_code == 200
    action = response.json()["action"]
    
    assert isinstance(action, list)
    assert len(action) == 3
    card_index, play_or_discard, draw_source = action
    
    assert isinstance(card_index, int)
    assert isinstance(play_or_discard, int)
    assert isinstance(draw_source, int)
    assert 0 <= card_index <= 7  # 8 cards in hand
    assert play_or_discard in [0, 1]
    
