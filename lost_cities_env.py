import numpy as np
from typing import Tuple, Dict, List, Optional

class LostCitiesEnv:
    """
    Lost Cities game environment optimized with NumPy arrays
    """
    NUM_SUITS = 6
    NUM_VALUES = 11  # 0 (handshake) and 2-10 (number cards)
    CARDS_PER_HANDSHAKE = 3
    INITIAL_HAND_SIZE = 8
    
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the game state.

        """
        # Initialize the game state with NumPy arrays
        self.deck = self._create_deck()
        np.random.shuffle(self.deck)
        
        # 2 for each player
        # Player hands: shape (2, 6(NUM_SUITS), 10(NUM_VALUES)) - count of each card type
        self.player_hands = np.zeros((2, self.NUM_SUITS, self.NUM_VALUES), dtype=np.int8)
        
        # Expeditions: shape (2, 6(NUM_SUITS), 10(NUM_VALUES)) - count of each card type
        self.expeditions = np.zeros((2, self.NUM_SUITS, self.NUM_VALUES), dtype=np.int8)
        
        # 2 players share the same discard piles
        # Discard piles: shape (6(NUM_SUITS), 10(NUM_VALUES)) - count of each card type
        self.discard_piles = np.zeros((self.NUM_SUITS, self.NUM_VALUES), dtype=np.int8)
        
        # Deal initial hands
        for _ in range(self.INITIAL_HAND_SIZE):
            for player in range(2):
                card = self.deck[0]
                self.deck = self.deck[1:]
                self.player_hands[player, card[0], card[1]] += 1
        
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        return self._get_state()

    def _create_deck(self) -> np.ndarray:
        """
        Creates the deck of cards using NumPy arrays.
        Returns: Array of shape (60, 2) where each row is [suit, value]
        Value mapping:
        - 0: handshake
        - 2-10: number cards (stored at indices 2-10)
        """
        deck = []
        # Add handshake cards (value 0)
        for suit in range(self.NUM_SUITS):
            deck.extend([[suit, 0]] * self.CARDS_PER_HANDSHAKE)
            # Add number cards (values 2-10)
            deck.extend([[suit, val] for val in range(2, 11)])
        return np.array(deck, dtype=np.int8)

    def _get_state(self) -> np.ndarray:
        """
        Returns a numerical representation of the game state.
        NUM_SUITS = 6, NUM_VALUES = 10
        # Output shape: (60) + (60) + (60) + (60) + 1
        
        This includes: 
        1) the current player's hand in a flattened array
        2) the current player's expeditions in a flattened array
        3) the opponent's expeditions in a flattened array
        4) the discard piles in a flattened array
        5) the size of the deck normalized by the total number of cards (72)
        """
        # Flatten and concatenate all state components
        state = []
        
        # Current player's hand
        state.extend(self.player_hands[self.current_player].flatten())
        
        # Current player's expeditions
        state.extend(self.expeditions[self.current_player].flatten())
        
        # Opponent's expeditions
        state.extend(self.expeditions[1 - self.current_player].flatten())
        
        # Discard piles
        state.extend(self.discard_piles.flatten())
        
        # Deck size (normalized)
        state.append(len(self.deck) / 72.0)
        
        return np.array(state, dtype=np.float32)

    def _is_valid_play(self, suit: int, value: int, player: int) -> bool:
        """
        Check if playing a card is valid.
        Values:
        - 0: handshake
        - 2-10: number cards (stored at indices 2-10)
        """
        expedition = self.expeditions[player, suit]
        
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

    def _calculate_score(self, player: int) -> int:
        """
        Calculate score for a player.
        """
        total_score = 0
        for suit in range(self.NUM_SUITS):
            expedition = self.expeditions[player, suit]
            if np.sum(expedition) == 0:
                continue
                
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
                
            total_score += score
            
        return total_score

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return the next state.
        action: (card_index, play_or_discard, draw_source)
        """

        card_index, play_or_discard, draw_source = action
        
        # Convert card_index to suit and value
        hand = self.player_hands[self.current_player]
        flat_index = np.where(hand.flatten() > 0)[0][card_index]
        suit, value = np.unravel_index(flat_index, hand.shape)
        
        # Remove card from hand
        self.player_hands[self.current_player, suit, value] -= 1
        
        # Play or discard
        if play_or_discard == 0:  # Play
            if not self._is_valid_play(suit, value, self.current_player):
                self.player_hands[self.current_player, suit, value] += 1
                raise ValueError("Invalid play")
            self.expeditions[self.current_player, suit, value] += 1
        else:  # Discard
            self.discard_piles[suit, value] += 1
        
        # Draw
        if draw_source == 0:  # Draw from deck
            if len(self.deck) == 0:
                self.game_over = True
            else:
                new_card = self.deck[0]
                self.deck = self.deck[1:]
                self.player_hands[self.current_player, new_card[0], new_card[1]] += 1
        else:  # Draw from discard pile
            discard_suit = draw_source - 1
            if np.sum(self.discard_piles[discard_suit]) == 0:
                # Reverse the play/discard
                if play_or_discard == 0:
                    self.expeditions[self.current_player, suit, value] -= 1
                else:
                    self.discard_piles[suit, value] -= 1
                self.player_hands[self.current_player, suit, value] += 1
                raise ValueError("Cannot draw from empty discard pile")
                
            # Find the top card in the discard pile
            flat_index = np.where(self.discard_piles[discard_suit] > 0)[0][-1]
            self.discard_piles[discard_suit, flat_index] -= 1
            self.player_hands[self.current_player, discard_suit, flat_index] += 1
        
        # Check for game over
        if len(self.deck) == 0:
            self.game_over = True
        
        # Calculate reward
        reward = 0
        if self.game_over:
            player_score = self._calculate_score(self.current_player)
            opponent_score = self._calculate_score(1 - self.current_player)
            reward = int(player_score - opponent_score)  # Convert numpy int to Python int
            self.winner = 0 if player_score > opponent_score else (1 if opponent_score > player_score else -1)
        
        # Switch players
        self.current_player = 1 - self.current_player
        
        return self._get_state(), reward, self.game_over, {}

    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        """
        Generate valid actions using NumPy operations.
        """
        valid_actions = []
        hand = self.player_hands[self.current_player]
        
        # Get indices of cards in hand
        card_indices = np.where(hand > 0)
        for idx in range(len(card_indices[0])):
            suit, value = card_indices[0][idx], card_indices[1][idx]
            card_index = len(np.where(hand.flatten()[:suit * self.NUM_VALUES + value] > 0)[0])
            
            # Check play actions
            if self._is_valid_play(suit, value, self.current_player):
                # Add valid draw sources
                valid_actions.extend([
                    (card_index, 0, 0)  # Draw from deck
                ])
                # Add valid discard pile draws
                for draw_suit in range(self.NUM_SUITS):
                    if np.sum(self.discard_piles[draw_suit]) > 0:
                        valid_actions.append((card_index, 0, draw_suit + 1))
            
            # Discard actions (always valid)
            valid_actions.append((card_index, 1, 0))  # Draw from deck
            for draw_suit in range(self.NUM_SUITS):
                if np.sum(self.discard_piles[draw_suit]) > 0:
                    valid_actions.append((card_index, 1, draw_suit + 1))
        
        return valid_actions

    def render(self):
        """
        Text-based visualization of the game state.
        """
        print(f"\nPlayer {self.current_player}'s turn:")
        
        print("\nHands:")
        for player in range(2):
            cards = []
            for suit in range(self.NUM_SUITS):
                for value in range(self.NUM_VALUES):
                    count = self.player_hands[player, suit, value]
                    if count > 0:
                        cards.extend([(suit, value)] * count)
            print(f"Player {player}: {cards}")
        
        print("\nExpeditions:")
        for player in range(2):
            exps = []
            for suit in range(self.NUM_SUITS):
                cards = []
                for value in range(self.NUM_VALUES):
                    count = self.expeditions[player, suit, value]
                    if count > 0:
                        cards.extend([(suit, value)] * count)
                exps.append(cards)
            print(f"Player {player}: {exps}")
        
        print("\nDiscard Piles:")
        for suit in range(self.NUM_SUITS):
            cards = []
            for value in range(self.NUM_VALUES):
                count = self.discard_piles[suit, value]
                if count > 0:
                    cards.extend([(suit, value)] * count)
            print(f"Suit {suit}: {cards}")
        
        print(f"\nDeck size: {len(self.deck)}")
        
        if self.game_over:
            if self.winner != -1:
                print(f"Winner: Player {self.winner}")
            else:
                print("Game ended in a draw")
