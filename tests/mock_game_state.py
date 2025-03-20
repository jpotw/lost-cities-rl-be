from api import GameState, PlayerState, Card
from game.lost_cities_env import LostCitiesEnv
import random
from typing import List

def create_mock_game_state():
    # Create initial deck with all 72 cards (12 cards * 6 colors)
    deck = []
    card_id = 1
    for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per color
        for _ in range(LostCitiesEnv.CARDS_PER_HANDSHAKE):
            deck.append(Card(id=card_id, color=color, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per color
        for value in range(2, 11):  # Changed from NUM_VALUES to explicit 11 to get values 2-10
            deck.append(Card(id=card_id, color=color, value=value, isHidden=True))
            card_id += 1

    # Shuffle deck
    random.shuffle(deck)

    # Distribute cards
    p1_hand = deck[:LostCitiesEnv.INITIAL_HAND_SIZE]  # First 8 cards for player 1
    p2_hand = deck[LostCitiesEnv.INITIAL_HAND_SIZE:2*LostCitiesEnv.INITIAL_HAND_SIZE]  # Next 8 cards for player 2
    remaining_deck = deck[2*LostCitiesEnv.INITIAL_HAND_SIZE:]  # Rest remain in deck

    # Create game state
    return GameState(
        players=[
            PlayerState(
                id="1",
                name="Player 1",
                type="HUMAN",
                hand=p1_hand,
                expeditions={
                    "RED": [],
                    "BLUE": [],
                    "GREEN": [],
                    "WHITE": [],
                    "YELLOW": [],
                    "PURPLE": []
                },
                score=0
            ),
            PlayerState(
                id="2",
                name="AI",
                type="AI",
                hand=p2_hand,
                expeditions={
                    "RED": [],
                    "BLUE": [],
                    "GREEN": [],
                    "WHITE": [],
                    "YELLOW": [],
                    "PURPLE": []
                },
                score=0
            )
        ],
        currentPlayerIndex=1,
        deck=remaining_deck,
        discardPiles={
            "RED": [],
            "BLUE": [],
            "GREEN": [],
            "WHITE": [],
            "YELLOW": [],
            "PURPLE": []
        },
        selectedCard=None,
        gamePhase="PLAY",
        isAIThinking=True,
        lastDiscarded=None,
        winner=None
    )

def create_mock_midgame_state():
    """Create a realistic mid-game state with cards distributed across all locations"""
    # Create all 72 cards first
    all_cards = []
    card_id = 1
    for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per color
        for _ in range(LostCitiesEnv.CARDS_PER_HANDSHAKE):
            all_cards.append(Card(id=card_id, color=color, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per color
        for value in range(2, 11):  # Changed from NUM_VALUES to explicit 11 to get values 2-10
            all_cards.append(Card(id=card_id, color=color, value=value, isHidden=True))
            card_id += 1

    # Distribute cards to create a mid-game state
    # Each player has some expeditions going
    p1_expeditions = {
        "RED": [
            Card(id=1, color="RED", value="HS", isHidden=False),
            Card(id=2, color="RED", value=2, isHidden=False),
            Card(id=3, color="RED", value=4, isHidden=False),
        ],
        "BLUE": [
            Card(id=4, color="BLUE", value=3, isHidden=False),
            Card(id=5, color="BLUE", value=5, isHidden=False),
        ],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [Card(id=6, color="YELLOW", value="HS", isHidden=False)],
        "PURPLE": []
    }

    p2_expeditions = {
        "RED": [],
        "BLUE": [],
        "GREEN": [
            Card(id=7, color="GREEN", value="HS", isHidden=False),
            Card(id=8, color="GREEN", value="HS", isHidden=False),
            Card(id=9, color="GREEN", value=4, isHidden=False),
            Card(id=10, color="GREEN", value=6, isHidden=False),
        ],
        "WHITE": [
            Card(id=11, color="WHITE", value=2, isHidden=False),
            Card(id=12, color="WHITE", value=5, isHidden=False),
        ],
        "YELLOW": [],
        "PURPLE": [Card(id=13, color="PURPLE", value=3, isHidden=False)]
    }

    # Some cards in discard piles
    discard_piles = {
        "RED": [Card(id=14, color="RED", value=3, isHidden=False)],
        "BLUE": [
            Card(id=15, color="BLUE", value="HS", isHidden=False),
            Card(id=16, color="BLUE", value=2, isHidden=False),
        ],
        "GREEN": [Card(id=17, color="GREEN", value=3, isHidden=False)],
        "WHITE": [],
        "YELLOW": [Card(id=18, color="YELLOW", value=4, isHidden=False)],
        "PURPLE": [Card(id=19, color="PURPLE", value="HS", isHidden=False)]
    }

    # Remove used cards from the deck
    used_ids = set()
    for exp in [p1_expeditions, p2_expeditions]:
        for cards in exp.values():
            for card in cards:
                used_ids.add(card.id)
    for cards in discard_piles.values():
        for card in cards:
            used_ids.add(card.id)

    remaining_cards = [card for card in all_cards if card.id not in used_ids]

    # Deal 8 cards to each player
    p1_hand = remaining_cards[:8]
    p2_hand = remaining_cards[8:16]
    # Rest go to deck
    deck = remaining_cards[16:]

    return GameState(
        players=[
            PlayerState(
                id="1",
                name="Player 1",
                type="HUMAN",
                hand=p1_hand,
                expeditions=p1_expeditions,
                score=0
            ),
            PlayerState(
                id="2",
                name="AI",
                type="AI",
                hand=p2_hand,
                expeditions=p2_expeditions,
                score=0
            )
        ],
        currentPlayerIndex=1,
        deck=deck,
        discardPiles=discard_piles,
        selectedCard=None,
        gamePhase="PLAY",
        isAIThinking=True,
        lastDiscarded={"BLUE": 2},  # Last discarded was BLUE 2
        winner=None
    )

def create_mock_endgame_state():
    """Create an end-game state where deck is empty and we have a winner"""
    # Create all 72 cards first
    all_cards = []
    card_id = 1
    for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per color
        for _ in range(3):
            all_cards.append(Card(id=card_id, color=color, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per color
        for value in range(2, 11):
            all_cards.append(Card(id=card_id, color=color, value=value, isHidden=True))
            card_id += 1

    # Create a winning state for Player 1 with high-scoring expeditions
    p1_expeditions = {
        "RED": [
            Card(id=1, color="RED", value="HS", isHidden=False),
            Card(id=2, color="RED", value="HS", isHidden=False),
            Card(id=3, color="RED", value=4, isHidden=False),
            Card(id=4, color="RED", value=5, isHidden=False),
            Card(id=5, color="RED", value=6, isHidden=False),
            Card(id=6, color="RED", value=7, isHidden=False),
            Card(id=7, color="RED", value=8, isHidden=False),
            Card(id=8, color="RED", value=9, isHidden=False),
        ],
        "BLUE": [
            Card(id=9, color="BLUE", value="HS", isHidden=False),
            Card(id=10, color="BLUE", value=3, isHidden=False),
            Card(id=11, color="BLUE", value=5, isHidden=False),
            Card(id=12, color="BLUE", value=6, isHidden=False),
            Card(id=13, color="BLUE", value=7, isHidden=False),
            Card(id=14, color="BLUE", value=8, isHidden=False),
            Card(id=15, color="BLUE", value=9, isHidden=False),
            Card(id=16, color="BLUE", value=10, isHidden=False),
        ],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [],
        "PURPLE": []
    }

    p2_expeditions = {
        "RED": [],
        "BLUE": [],
        "GREEN": [
            Card(id=17, color="GREEN", value="HS", isHidden=False),
            Card(id=18, color="GREEN", value=2, isHidden=False),
            Card(id=19, color="GREEN", value=3, isHidden=False),
            Card(id=20, color="GREEN", value=7, isHidden=False),
            Card(id=21, color="GREEN", value=8, isHidden=False),
            Card(id=22, color="GREEN", value=9, isHidden=False),
            Card(id=23, color="GREEN", value=10, isHidden=False),
        ],
        "WHITE": [
            Card(id=24, color="WHITE", value="HS", isHidden=False),
            Card(id=25, color="WHITE", value="HS", isHidden=False),
            Card(id=26, color="WHITE", value=2, isHidden=False),
            Card(id=27, color="WHITE", value=3, isHidden=False),
            Card(id=28, color="WHITE", value=4, isHidden=False),
            Card(id=29, color="WHITE", value=5, isHidden=False),
            Card(id=30, color="WHITE", value=6, isHidden=False),
        ],
        "YELLOW": [],
        "PURPLE": []
    }

    # Most cards are in discard piles since it's end game
    discard_piles = {
        "RED": [
            Card(id=31, color="RED", value=2, isHidden=False),
            Card(id=32, color="RED", value=3, isHidden=False),
            Card(id=33, color="RED", value=10, isHidden=False),
            Card(id=34, color="RED", value="HS", isHidden=False),
        ],
        "BLUE": [
            Card(id=35, color="BLUE", value=2, isHidden=False),
            Card(id=36, color="BLUE", value=4, isHidden=False),
            Card(id=37, color="BLUE", value="HS", isHidden=False),
            Card(id=38, color="BLUE", value="HS", isHidden=False),
        ],
        "GREEN": [
            Card(id=39, color="GREEN", value="HS", isHidden=False),
            Card(id=40, color="GREEN", value=4, isHidden=False),
            Card(id=41, color="GREEN", value=5, isHidden=False),
            Card(id=42, color="GREEN", value=6, isHidden=False),
        ],
        "WHITE": [
            Card(id=43, color="WHITE", value="HS", isHidden=False),
            Card(id=44, color="WHITE", value=7, isHidden=False),
            Card(id=45, color="WHITE", value=8, isHidden=False),
            Card(id=46, color="WHITE", value=9, isHidden=False),
            Card(id=47, color="WHITE", value=10, isHidden=False),
        ],
        "YELLOW": [
            Card(id=48, color="YELLOW", value="HS", isHidden=False),
            Card(id=49, color="YELLOW", value="HS", isHidden=False),
            Card(id=50, color="YELLOW", value="HS", isHidden=False),
            Card(id=51, color="YELLOW", value=2, isHidden=False),
            Card(id=52, color="YELLOW", value=3, isHidden=False),
            Card(id=53, color="YELLOW", value=4, isHidden=False),
            Card(id=54, color="YELLOW", value=5, isHidden=False),
            Card(id=55, color="YELLOW", value=6, isHidden=False),
            Card(id=56, color="YELLOW", value=7, isHidden=False),
        ],
        "PURPLE": [
        ]
    }

    # Create hands for both players
    p1_hand = [
            Card(id=57, color="PURPLE", value="HS", isHidden=False),
            Card(id=58, color="PURPLE", value="HS", isHidden=False),
            Card(id=59, color="PURPLE", value="HS", isHidden=False),
            Card(id=60, color="PURPLE", value=2, isHidden=False),
            Card(id=61, color="PURPLE", value=3, isHidden=False),
            Card(id=62, color="PURPLE", value=4, isHidden=False),
            Card(id=63, color="PURPLE", value=5, isHidden=False),
            Card(id=64, color="PURPLE", value=6, isHidden=False),
    ]

    p2_hand = [
        Card(id=65, color="YELLOW", value=8, isHidden=True),
        Card(id=66, color="YELLOW", value=9, isHidden=True),
        Card(id=67, color="YELLOW", value=10, isHidden=True),
        Card(id=68, color="PURPLE", value=7, isHidden=True),
        Card(id=69, color="PURPLE", value=8, isHidden=True),
        Card(id=70, color="PURPLE", value=9, isHidden=True),
        Card(id=71, color="PURPLE", value=10, isHidden=True),
        Card(id=72, color="GREEN", value="HS", isHidden=True),
    ]

    # Verify we have exactly 72 cards total
    total_cards = (
        sum(len(cards) for cards in p1_expeditions.values()) +
        sum(len(cards) for cards in p2_expeditions.values()) +
        sum(len(cards) for cards in discard_piles.values()) +
        len(p1_hand) + len(p2_hand)
    )
    assert total_cards == 72, f"Expected 72 total cards, but got {total_cards}"

    # Verify each color has exactly 12 cards (3 HS + 9 number cards)
    cards_by_color = {"RED": [], "BLUE": [], "GREEN": [], "WHITE": [], "YELLOW": [], "PURPLE": []}
    
    # Collect all cards
    for exp in [p1_expeditions, p2_expeditions]:
        for cards in exp.values():
            for card in cards:
                cards_by_color[card.color].append(card)
    for cards in discard_piles.values():
        for card in cards:
            cards_by_color[card.color].append(card)
    for card in p1_hand + p2_hand:
        cards_by_color[card.color].append(card)

    # Verify each color
    for color, cards in cards_by_color.items():
        assert len(cards) == 12, f"Expected 12 cards for {color}, but got {len(cards)}"
        hs_count = sum(1 for c in cards if c.value == "HS")
        assert hs_count == 3, f"Expected 3 handshake cards for {color}, but got {hs_count}"

    return GameState(
        players=[
            PlayerState(
                id="1",
                name="Player 1",
                type="HUMAN",
                hand=p1_hand,
                expeditions=p1_expeditions,
                score=157  # High score from completed expeditions
            ),
            PlayerState(
                id="2",
                name="AI",
                type="AI",
                hand=p2_hand,
                expeditions=p2_expeditions,
                score=42  # Lower score
            )
        ],
        currentPlayerIndex=1,
        deck=[],  # Empty deck signifies end game
        discardPiles=discard_piles,
        selectedCard=None,
        gamePhase="PLAY",
        isAIThinking=True,
        lastDiscarded={"GREEN": 6},
        winner="Player 1"  # Explicitly set winner
    )

def create_initial_deck() -> List[Card]:
    # Create initial deck with all 72 cards (12 cards * 6 colors)
    deck = []
    card_id = 1
    
    for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per color
        for _ in range(3):
            deck.append(Card(id=card_id, color=color, value="HS", isHidden=True))
            card_id += 1
            
        # 9 number cards (2-10) per color
        for value in range(2, 11):
            deck.append(Card(id=card_id, color=color, value=value, isHidden=True))
            card_id += 1
            
    return deck

def create_mock_game_state_1() -> GameState:
    """
    Create a mock game state for testing.
    Player 1 has played some cards and has some cards in hand.
    Player 2 has played some cards and has some cards in hand.
    Some cards are in the discard piles.
    """
    # Create player 1's hand and expeditions
    player1_hand = [
        Card(id=1, color="RED", value="HS", isHidden=False),
        Card(id=2, color="RED", value=2, isHidden=False),
        Card(id=3, color="RED", value=4, isHidden=False),
        Card(id=4, color="BLUE", value=3, isHidden=False),
        Card(id=5, color="BLUE", value=5, isHidden=False)
    ]
    
    player1_expeditions = {
        "RED": [],
        "BLUE": [],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [Card(id=6, color="YELLOW", value="HS", isHidden=False)],
        "PURPLE": []
    }
    
    # Create player 2's hand and expeditions
    player2_hand = [
        Card(id=7, color="GREEN", value="HS", isHidden=False),
        Card(id=8, color="GREEN", value="HS", isHidden=False),
        Card(id=9, color="GREEN", value=4, isHidden=False),
        Card(id=10, color="GREEN", value=6, isHidden=False),
        Card(id=11, color="WHITE", value=2, isHidden=False),
        Card(id=12, color="WHITE", value=5, isHidden=False)
    ]
    
    player2_expeditions = {
        "RED": [],
        "BLUE": [],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [],
        "PURPLE": [Card(id=13, color="PURPLE", value=3, isHidden=False)]
    }
    
    # Create discard piles
    discard_piles = {
        "RED": [Card(id=14, color="RED", value=3, isHidden=False)],
        "BLUE": [
            Card(id=15, color="BLUE", value="HS", isHidden=False),
            Card(id=16, color="BLUE", value=2, isHidden=False)
        ],
        "GREEN": [Card(id=17, color="GREEN", value=3, isHidden=False)],
        "WHITE": [],
        "YELLOW": [Card(id=18, color="YELLOW", value=4, isHidden=False)],
        "PURPLE": [Card(id=19, color="PURPLE", value="HS", isHidden=False)]
    }
    
    # Create game state
    game_state = GameState(
        players=[
            PlayerState(
                id="1",
                name="Player 1",
                type="HUMAN",
                hand=player1_hand,
                expeditions=player1_expeditions,
                score=0
            ),
            PlayerState(
                id="2",
                name="AI",
                type="AI",
                hand=player2_hand,
                expeditions=player2_expeditions,
                score=0
            )
        ],
        currentPlayerIndex=1,
        deck=[],
        discardPiles=discard_piles,
        selectedCard=None,
        gamePhase="PLAY",
        isAIThinking=True,
        lastDiscarded=None,
        winner=None
    )
    
    return game_state

def create_mock_game_state_2() -> GameState:
    """
    Create a mock game state for testing.
    This state represents a game that's further along than mock_game_state_1.
    More cards have been played to expeditions and discarded.
    """
    # Create all cards that will be used
    all_cards = []
    card_id = 1
    
    for color in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per color
        for _ in range(3):
            all_cards.append(Card(id=card_id, color=color, value="HS", isHidden=True))
            card_id += 1
            
        # 9 number cards (2-10) per color
        for value in range(2, 11):
            all_cards.append(Card(id=card_id, color=color, value=value, isHidden=True))
            card_id += 1
    
    # Create player 1's hand and expeditions
    player1_hand = [
        Card(id=1, color="RED", value="HS", isHidden=False),
        Card(id=2, color="RED", value="HS", isHidden=False),
        Card(id=3, color="RED", value=4, isHidden=False),
        Card(id=4, color="RED", value=5, isHidden=False),
        Card(id=5, color="RED", value=6, isHidden=False),
        Card(id=6, color="RED", value=7, isHidden=False),
        Card(id=7, color="RED", value=8, isHidden=False),
        Card(id=8, color="RED", value=9, isHidden=False)
    ]
    
    player1_expeditions = {
        "BLUE": [
            Card(id=9, color="BLUE", value="HS", isHidden=False),
            Card(id=10, color="BLUE", value=3, isHidden=False),
            Card(id=11, color="BLUE", value=5, isHidden=False),
            Card(id=12, color="BLUE", value=6, isHidden=False),
            Card(id=13, color="BLUE", value=7, isHidden=False),
            Card(id=14, color="BLUE", value=8, isHidden=False),
            Card(id=15, color="BLUE", value=9, isHidden=False),
            Card(id=16, color="BLUE", value=10, isHidden=False)
        ],
        "RED": [],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [],
        "PURPLE": []
    }
    
    # Create player 2's hand and expeditions
    player2_hand = [
        Card(id=17, color="GREEN", value="HS", isHidden=False),
        Card(id=18, color="GREEN", value=2, isHidden=False),
        Card(id=19, color="GREEN", value=3, isHidden=False),
        Card(id=20, color="GREEN", value=7, isHidden=False),
        Card(id=21, color="GREEN", value=8, isHidden=False),
        Card(id=22, color="GREEN", value=9, isHidden=False),
        Card(id=23, color="GREEN", value=10, isHidden=False)
    ]
    
    player2_expeditions = {
        "WHITE": [
            Card(id=24, color="WHITE", value="HS", isHidden=False),
            Card(id=25, color="WHITE", value="HS", isHidden=False),
            Card(id=26, color="WHITE", value=2, isHidden=False),
            Card(id=27, color="WHITE", value=3, isHidden=False),
            Card(id=28, color="WHITE", value=4, isHidden=False),
            Card(id=29, color="WHITE", value=5, isHidden=False),
            Card(id=30, color="WHITE", value=6, isHidden=False)
        ],
        "RED": [],
        "BLUE": [],
        "GREEN": [],
        "YELLOW": [],
        "PURPLE": []
    }
    
    # Create discard piles
    discard_piles = {
        "RED": [
            Card(id=31, color="RED", value=2, isHidden=False),
            Card(id=32, color="RED", value=3, isHidden=False),
            Card(id=33, color="RED", value=10, isHidden=False),
            Card(id=34, color="RED", value="HS", isHidden=False)
        ],
        "BLUE": [
            Card(id=35, color="BLUE", value=2, isHidden=False),
            Card(id=36, color="BLUE", value=4, isHidden=False),
            Card(id=37, color="BLUE", value="HS", isHidden=False),
            Card(id=38, color="BLUE", value="HS", isHidden=False)
        ],
        "GREEN": [
            Card(id=39, color="GREEN", value="HS", isHidden=False),
            Card(id=40, color="GREEN", value=4, isHidden=False),
            Card(id=41, color="GREEN", value=5, isHidden=False),
            Card(id=42, color="GREEN", value=6, isHidden=False)
        ],
        "WHITE": [
            Card(id=43, color="WHITE", value="HS", isHidden=False),
            Card(id=44, color="WHITE", value=7, isHidden=False),
            Card(id=45, color="WHITE", value=8, isHidden=False),
            Card(id=46, color="WHITE", value=9, isHidden=False),
            Card(id=47, color="WHITE", value=10, isHidden=False)
        ],
        "YELLOW": [
            Card(id=48, color="YELLOW", value="HS", isHidden=False),
            Card(id=49, color="YELLOW", value="HS", isHidden=False),
            Card(id=50, color="YELLOW", value="HS", isHidden=False)
        ],
        "PURPLE": []
    }
    
    # Create game state
    game_state = GameState(
        players=[
            PlayerState(
                id="1",
                name="Player 1",
                type="HUMAN",
                hand=player1_hand,
                expeditions=player1_expeditions,
                score=0
            ),
            PlayerState(
                id="2",
                name="AI",
                type="AI",
                hand=player2_hand,
                expeditions=player2_expeditions,
                score=0
            )
        ],
        currentPlayerIndex=1,
        deck=[],
        discardPiles=discard_piles,
        selectedCard=None,
        gamePhase="PLAY",
        isAIThinking=True,
        lastDiscarded=None,
        winner=None
    )
    
    return game_state 