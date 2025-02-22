import numpy as np
from typing import Tuple, List

def create_deck(num_suits: int, cards_per_handshake: int) -> np.ndarray:
    """
    Creates the deck of cards using NumPy arrays.
    Returns: Array of shape (60, 2) where each row is [suit, value]
    Value mapping:
    - 0: handshake
    - 2-10: number cards (stored at indices 2-10)
    """
    deck = []
    # Add handshake cards (value 0)
    for suit in range(num_suits):
        deck.extend([[suit, 0]] * cards_per_handshake)
        # Add number cards (values 2-10)
        deck.extend([[suit, val] for val in range(2, 11)])
    return np.array(deck, dtype=np.int8)

def is_valid_play(expedition: np.ndarray, suit: int, value: int) -> bool:
    """
    Check if playing a card is valid.
    Values:
    - 0: handshake
    - 2-10: number cards (stored at indices 2-10)
    """
    if np.sum(expedition) == 0:
        return True  # Empty expedition
        
    if value == 0:  # Handshake card
        # Can only play if no number cards are present
        return np.sum(expedition[2:]) == 0
        
    # Get the highest value card in the expedition
    number_cards = np.nonzero(expedition[2:])[0]
    if len(number_cards) == 0:
        return True
        
    # Since indices 2-10 correspond to actual values 2-10, we can use the index directly
    highest_value = number_cards[-1] + 2
    return value > highest_value

def calculate_score(expedition: np.ndarray) -> int:
    """
    Calculate score for a single expedition.
    """
    if np.sum(expedition) == 0:
        return 0
        
    # Calculate base score
    # Use actual card values (2-10) for score calculation
    values = np.arange(2, 11)
    card_sum = np.sum(values * expedition[2:])
    
    # Calculate multiplier (handshake cards + 1)
    multiplier = expedition[0] + 1
    
    # Calculate expedition score
    score = -20 + card_sum
    score *= multiplier
    
    # Bonus for 8 or more cards
    if np.sum(expedition) >= 8:
        score += 20
        
    return score

def get_valid_actions(player_hands: np.ndarray, expeditions: np.ndarray, 
                     discard_piles: np.ndarray, current_player: int,
                     num_suits: int, num_values: int) -> List[Tuple[int, int, int]]:
    """
    Generate valid actions using NumPy operations.
    Returns list of tuples (card_index, play_or_discard, draw_source)
    """
    valid_actions = []
    hand = player_hands[current_player]
    
    # Get indices of cards in hand
    card_indices = np.where(hand > 0)
    for idx in range(len(card_indices[0])):
        suit, value = card_indices[0][idx], card_indices[1][idx]
        card_index = len(np.where(hand.flatten()[:suit * num_values + value] > 0)[0])
        
        # Check play actions
        if is_valid_play(expeditions[current_player, suit], suit, value):
            # Add valid draw sources
            valid_actions.extend([
                (card_index, 0, 0)  # Draw from deck
            ])
            # Add valid discard pile draws
            for draw_suit in range(num_suits):
                if np.sum(discard_piles[draw_suit]) > 0:
                    valid_actions.append((card_index, 0, draw_suit + 1))
        
        # Discard actions (always valid)
        valid_actions.append((card_index, 1, 0))  # Draw from deck
        for draw_suit in range(num_suits):
            if np.sum(discard_piles[draw_suit]) > 0:
                valid_actions.append((card_index, 1, draw_suit + 1))
    
    return valid_actions
