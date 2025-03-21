"""Utility functions for the Lost Cities game environment."""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def create_deck(num_colors: int, cards_per_handshake: int) -> NDArray[np.int8]:
    """Create a deck of cards for Lost Cities.

    Args:
        num_colors: Number of different card colors.
        cards_per_handshake: Number of handshake cards per color.

    Returns:
        NDArray[np.int8]: Array of shape (60, 2) where each row is [color, value].
    """
    deck = []

    # Add handshake cards (0) and number cards (2-10) for each color
    for color in range(num_colors):
        deck.extend([[color, 0]] * cards_per_handshake)
        # Add number cards
        deck.extend([[color, val] for val in range(2, 11)])

    return np.array(deck, dtype=np.int8)


def is_valid_play(expedition: NDArray[np.int8], color: int, value: int) -> bool:
    """Check if a card can be played on an expedition.

    Args:
        expedition: Array representing cards in the expedition.
        color: Card color index.
        value: Card value (0 for handshake, 2-10 for number cards).

    Returns:
        bool: True if the card can be played on the expedition.

    Note:
        A card can only be played if its value is greater than the last card played.
        Handshake cards (value=0) can only be played when no other cards have been played.
    """
    # Get all cards played in this expedition
    played_indices = np.where(expedition > 0)[0]

    if len(played_indices) == 0:
        return True  # Can play any card on empty expedition

    if value == 0:  # Handshake card
        return False  # Can't play handshake after other cards

    # Get the highest value card played so far
    highest_played_value = max(played_indices)  # Since indices match card values

    return value > highest_played_value


def calculate_score(expedition: NDArray[np.int8]) -> int:
    """Calculate the score for a single expedition.

    Args:
        expedition: Array representing cards in the expedition.

    Returns:
        int: Score for the expedition, calculated as:
            (sum of card values - 20) * (1 + number of handshake cards)
    """
    # Get all cards played in this expedition
    played_cards = np.where(expedition > 0)[0]

    if len(played_cards) == 0:
        return 0

    # Calculate base score
    base_score = sum(played_cards[played_cards > 0])  # Sum all non-handshake cards

    # Apply multiplier based on handshake cards
    multiplier = 1 + np.sum(expedition[0])  # Count of handshake cards

    # Apply expedition cost (-20)
    final_score = (base_score - 20) * multiplier

    return final_score


def get_valid_actions(
    hand: NDArray[np.int8],
    expeditions: NDArray[np.int8],
    discard_piles: NDArray[np.int8],
    current_player: int,
    num_colors: int,
    num_values: int,
) -> List[Tuple[int, int, int]]:
    """Get all valid actions for the current player.

    Args:
        hand: Array representing all players' hands.
        expeditions: Array representing all players' expeditions.
        discard_piles: Array representing discard piles.
        current_player: Index of the current player.
        num_colors: Number of different card colors.
        num_values: Number of different card values.

    Returns:
        List[Tuple[int, int, int]]: List of valid actions, each represented as
            (card_index, play_or_discard, draw_source) where:
            - card_index: Index of the card in hand (0 to N-1 where N is number of cards)
            - play_or_discard: 0 for play, 1 for discard
            - draw_source: 0 for deck, 1-6 for discard piles
    """
    valid_actions = []

    # Get indices of cards in hand
    card_indices = np.where(hand[current_player] > 0)
    
    # Create a mapping of (color, value) to card_index
    card_mapping = {}
    for idx in range(len(card_indices[0])):
        color, value = card_indices[0][idx], card_indices[1][idx]
        card_mapping[(color, value)] = idx

    # For each card in hand
    for idx in range(len(card_indices[0])):
        color, value = card_indices[0][idx], card_indices[1][idx]
        card_index = card_mapping[(color, value)]

        # Check if card can be played
        if is_valid_play(expeditions[current_player, color], color, value):
            # Can draw from deck
            valid_actions.append((card_index, 0, 0))

            # Can draw from any non-empty discard pile
            for draw_color in range(num_colors):
                if np.sum(discard_piles[draw_color]) > 0:
                    valid_actions.append((card_index, 0, draw_color + 1))

        # Can always discard
        # Can draw from deck
        valid_actions.append((card_index, 1, 0))

        # Can draw from any non-empty discard pile
        for draw_color in range(num_colors):
            if np.sum(discard_piles[draw_color]) > 0:
                valid_actions.append((card_index, 1, draw_color + 1))

    return valid_actions
