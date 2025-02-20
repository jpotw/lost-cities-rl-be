from api import GameState, PlayerState, Card
from game.lost_cities_env import LostCitiesEnv
import random

def create_mock_game_state():
    # Create initial deck with all 72 cards (12 cards * 6 suits)
    deck = []
    card_id = 1
    for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per suit
        for _ in range(LostCitiesEnv.CARDS_PER_HANDSHAKE):
            deck.append(Card(id=card_id, suit=suit, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per suit
        for value in range(2, 11):  # Changed from NUM_VALUES to explicit 11 to get values 2-10
            deck.append(Card(id=card_id, suit=suit, value=value, isHidden=True))
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
    for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per suit
        for _ in range(LostCitiesEnv.CARDS_PER_HANDSHAKE):
            all_cards.append(Card(id=card_id, suit=suit, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per suit
        for value in range(2, 11):  # Changed from NUM_VALUES to explicit 11 to get values 2-10
            all_cards.append(Card(id=card_id, suit=suit, value=value, isHidden=True))
            card_id += 1

    # Distribute cards to create a mid-game state
    # Each player has some expeditions going
    p1_expeditions = {
        "RED": [
            Card(id=1, suit="RED", value="HS", isHidden=False),
            Card(id=2, suit="RED", value=2, isHidden=False),
            Card(id=3, suit="RED", value=4, isHidden=False),
        ],
        "BLUE": [
            Card(id=4, suit="BLUE", value=3, isHidden=False),
            Card(id=5, suit="BLUE", value=5, isHidden=False),
        ],
        "GREEN": [],
        "WHITE": [],
        "YELLOW": [Card(id=6, suit="YELLOW", value="HS", isHidden=False)],
        "PURPLE": []
    }

    p2_expeditions = {
        "RED": [],
        "BLUE": [],
        "GREEN": [
            Card(id=7, suit="GREEN", value="HS", isHidden=False),
            Card(id=8, suit="GREEN", value="HS", isHidden=False),
            Card(id=9, suit="GREEN", value=4, isHidden=False),
            Card(id=10, suit="GREEN", value=6, isHidden=False),
        ],
        "WHITE": [
            Card(id=11, suit="WHITE", value=2, isHidden=False),
            Card(id=12, suit="WHITE", value=5, isHidden=False),
        ],
        "YELLOW": [],
        "PURPLE": [Card(id=13, suit="PURPLE", value=3, isHidden=False)]
    }

    # Some cards in discard piles
    discard_piles = {
        "RED": [Card(id=14, suit="RED", value=3, isHidden=False)],
        "BLUE": [
            Card(id=15, suit="BLUE", value="HS", isHidden=False),
            Card(id=16, suit="BLUE", value=2, isHidden=False),
        ],
        "GREEN": [Card(id=17, suit="GREEN", value=3, isHidden=False)],
        "WHITE": [],
        "YELLOW": [Card(id=18, suit="YELLOW", value=4, isHidden=False)],
        "PURPLE": [Card(id=19, suit="PURPLE", value="HS", isHidden=False)]
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
    for suit in ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]:
        # 3 handshake cards per suit
        for _ in range(3):
            all_cards.append(Card(id=card_id, suit=suit, value="HS", isHidden=True))
            card_id += 1
        # 9 number cards (2-10) per suit
        for value in range(2, 11):
            all_cards.append(Card(id=card_id, suit=suit, value=value, isHidden=True))
            card_id += 1

    # Create a winning state for Player 1 with high-scoring expeditions
    p1_expeditions = {
        "RED": [
            Card(id=1, suit="RED", value="HS", isHidden=False),
            Card(id=2, suit="RED", value="HS", isHidden=False),
            Card(id=3, suit="RED", value=4, isHidden=False),
            Card(id=4, suit="RED", value=5, isHidden=False),
            Card(id=5, suit="RED", value=6, isHidden=False),
            Card(id=6, suit="RED", value=7, isHidden=False),
            Card(id=7, suit="RED", value=8, isHidden=False),
            Card(id=8, suit="RED", value=9, isHidden=False),
        ],
        "BLUE": [
            Card(id=9, suit="BLUE", value="HS", isHidden=False),
            Card(id=10, suit="BLUE", value=3, isHidden=False),
            Card(id=11, suit="BLUE", value=5, isHidden=False),
            Card(id=12, suit="BLUE", value=6, isHidden=False),
            Card(id=13, suit="BLUE", value=7, isHidden=False),
            Card(id=14, suit="BLUE", value=8, isHidden=False),
            Card(id=15, suit="BLUE", value=9, isHidden=False),
            Card(id=16, suit="BLUE", value=10, isHidden=False),
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
            Card(id=17, suit="GREEN", value="HS", isHidden=False),
            Card(id=18, suit="GREEN", value=2, isHidden=False),
            Card(id=19, suit="GREEN", value=3, isHidden=False),
            Card(id=20, suit="GREEN", value=7, isHidden=False),
            Card(id=21, suit="GREEN", value=8, isHidden=False),
            Card(id=22, suit="GREEN", value=9, isHidden=False),
            Card(id=23, suit="GREEN", value=10, isHidden=False),
        ],
        "WHITE": [
            Card(id=24, suit="WHITE", value="HS", isHidden=False),
            Card(id=25, suit="WHITE", value="HS", isHidden=False),
            Card(id=26, suit="WHITE", value=2, isHidden=False),
            Card(id=27, suit="WHITE", value=3, isHidden=False),
            Card(id=28, suit="WHITE", value=4, isHidden=False),
            Card(id=29, suit="WHITE", value=5, isHidden=False),
            Card(id=30, suit="WHITE", value=6, isHidden=False),
        ],
        "YELLOW": [],
        "PURPLE": []
    }

    # Most cards are in discard piles since it's end game
    discard_piles = {
        "RED": [
            Card(id=31, suit="RED", value=2, isHidden=False),
            Card(id=32, suit="RED", value=3, isHidden=False),
            Card(id=33, suit="RED", value=10, isHidden=False),
            Card(id=34, suit="RED", value="HS", isHidden=False),
        ],
        "BLUE": [
            Card(id=35, suit="BLUE", value=2, isHidden=False),
            Card(id=36, suit="BLUE", value=4, isHidden=False),
            Card(id=37, suit="BLUE", value="HS", isHidden=False),
            Card(id=38, suit="BLUE", value="HS", isHidden=False),
        ],
        "GREEN": [
            Card(id=39, suit="GREEN", value="HS", isHidden=False),
            Card(id=40, suit="GREEN", value=4, isHidden=False),
            Card(id=41, suit="GREEN", value=5, isHidden=False),
            Card(id=42, suit="GREEN", value=6, isHidden=False),
        ],
        "WHITE": [
            Card(id=43, suit="WHITE", value="HS", isHidden=False),
            Card(id=44, suit="WHITE", value=7, isHidden=False),
            Card(id=45, suit="WHITE", value=8, isHidden=False),
            Card(id=46, suit="WHITE", value=9, isHidden=False),
            Card(id=47, suit="WHITE", value=10, isHidden=False),
        ],
        "YELLOW": [
            Card(id=48, suit="YELLOW", value="HS", isHidden=False),
            Card(id=49, suit="YELLOW", value="HS", isHidden=False),
            Card(id=50, suit="YELLOW", value="HS", isHidden=False),
            Card(id=51, suit="YELLOW", value=2, isHidden=False),
            Card(id=52, suit="YELLOW", value=3, isHidden=False),
            Card(id=53, suit="YELLOW", value=4, isHidden=False),
            Card(id=54, suit="YELLOW", value=5, isHidden=False),
            Card(id=55, suit="YELLOW", value=6, isHidden=False),
            Card(id=56, suit="YELLOW", value=7, isHidden=False),
        ],
        "PURPLE": [
        ]
    }

    # Create hands for both players
    p1_hand = [
            Card(id=57, suit="PURPLE", value="HS", isHidden=False),
            Card(id=58, suit="PURPLE", value="HS", isHidden=False),
            Card(id=59, suit="PURPLE", value="HS", isHidden=False),
            Card(id=60, suit="PURPLE", value=2, isHidden=False),
            Card(id=61, suit="PURPLE", value=3, isHidden=False),
            Card(id=62, suit="PURPLE", value=4, isHidden=False),
            Card(id=63, suit="PURPLE", value=5, isHidden=False),
            Card(id=64, suit="PURPLE", value=6, isHidden=False),
    ]

    p2_hand = [
        Card(id=65, suit="YELLOW", value=8, isHidden=True),
        Card(id=66, suit="YELLOW", value=9, isHidden=True),
        Card(id=67, suit="YELLOW", value=10, isHidden=True),
        Card(id=68, suit="PURPLE", value=7, isHidden=True),
        Card(id=69, suit="PURPLE", value=8, isHidden=True),
        Card(id=70, suit="PURPLE", value=9, isHidden=True),
        Card(id=71, suit="PURPLE", value=10, isHidden=True),
        Card(id=72, suit="GREEN", value="HS", isHidden=True),
    ]

    # Verify we have exactly 72 cards total
    total_cards = (
        sum(len(cards) for cards in p1_expeditions.values()) +
        sum(len(cards) for cards in p2_expeditions.values()) +
        sum(len(cards) for cards in discard_piles.values()) +
        len(p1_hand) + len(p2_hand)
    )
    assert total_cards == 72, f"Expected 72 total cards, but got {total_cards}"

    # Verify each suit has exactly 12 cards (3 HS + 9 number cards)
    cards_by_suit = {"RED": [], "BLUE": [], "GREEN": [], "WHITE": [], "YELLOW": [], "PURPLE": []}
    
    # Collect all cards
    for exp in [p1_expeditions, p2_expeditions]:
        for cards in exp.values():
            for card in cards:
                cards_by_suit[card.suit].append(card)
    for cards in discard_piles.values():
        for card in cards:
            cards_by_suit[card.suit].append(card)
    for card in p1_hand + p2_hand:
        cards_by_suit[card.suit].append(card)

    # Verify each suit
    for suit, cards in cards_by_suit.items():
        assert len(cards) == 12, f"Expected 12 cards for {suit}, but got {len(cards)}"
        hs_count = sum(1 for c in cards if c.value == "HS")
        assert hs_count == 3, f"Expected 3 handshake cards for {suit}, but got {hs_count}"

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