import numpy as np
from typing import Tuple, List

def create_deck(num_colors: int, cards_per_handshake: int) -> np.ndarray:
    """
    Creates a deck of cards for Lost Cities.
    
    Returns: Array of shape (60, 2) where each row is [color, value]
    """
    deck = []
    
    # Add handshake cards (0) and number cards (2-10) for each color
    for color in range(num_colors):
        deck.extend([[color, 0]] * cards_per_handshake)
        # Add number cards
        deck.extend([[color, val] for val in range(2, 11)])
    
    return np.array(deck)

def is_valid_play(expedition: np.ndarray, color: int, value: int) -> bool:
    """
    Check if a card can be played on an expedition.
    A card can only be played if its value is greater than the last card played.
    Handshake cards (value=0) can only be played when no other cards have been played.
    """
    # Get all cards played in this expedition
    played_cards = np.where(expedition > 0)[0]
    
    if len(played_cards) == 0:
        return True
    
    if value == 0:  # Handshake card
        return False
    
    # Get the highest value card played
    highest_value = played_cards[-1]
    
    return value > highest_value

def calculate_score(expedition: np.ndarray) -> int:
    """
    Calculate the score for a single expedition.
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
    hand: np.ndarray,
    expeditions: np.ndarray,
    discard_piles: np.ndarray,
    current_player: int,
    num_colors: int, num_values: int) -> List[Tuple[int, int, int]]:
    """
    Get all valid actions for the current player.
    Returns: List of tuples (card_index, play_or_discard, draw_source)
    """
    valid_actions = []
    
    # Get indices of cards in hand
    card_indices = np.where(hand[current_player] > 0)
    
    # For each card in hand
    for idx in range(len(card_indices[0])):
        color, value = card_indices[0][idx], card_indices[1][idx]
        card_index = len(np.where(hand.flatten()[:color * num_values + value] > 0)[0])
        
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
