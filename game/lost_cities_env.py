import numpy as np
from typing import Tuple, Dict, List, Optional
from game import utils

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
        self.deck = utils.create_deck(self.NUM_SUITS, self.CARDS_PER_HANDSHAKE)
        np.random.shuffle(self.deck)
        
        # Player hands: shape (2, 6(NUM_SUITS), 11(NUM_VALUES)) - count of each card type
        self.player_hands = np.zeros((2, self.NUM_SUITS, self.NUM_VALUES), dtype=np.int8)
        
        # Expeditions: shape (2, 6(NUM_SUITS), 11(NUM_VALUES)) - count of each card type
        self.expeditions = np.zeros((2, self.NUM_SUITS, self.NUM_VALUES), dtype=np.int8)
        
        # Discard piles: shape (6(NUM_SUITS), 11(NUM_VALUES)) - count of each card type
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

    def get_player_hand(self, player: int) -> List[Tuple[int, int]]:
        """
        Return the hand of the specified player as a list of (suit, value) tuples.
        """
        hand = []
        for suit in range(self.NUM_SUITS):
            for value in range(self.NUM_VALUES):
                count = self.player_hands[player, suit, value]
                if count > 0:
                    hand.extend([(suit, value)] * count)
        return hand

    def index_to_suit(self, index: int) -> str:
        """
        Convert suit index to string representation.
        """
        suits = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
        return suits[index]

    def _get_state(self) -> np.ndarray:
        """
        Returns a numerical representation of the game state.
        """
        state = []
        state.extend(self.player_hands[self.current_player].flatten())
        state.extend(self.expeditions[self.current_player].flatten())
        state.extend(self.expeditions[1 - self.current_player].flatten())
        state.extend(self.discard_piles.flatten())
        state.append(len(self.deck) / 72.0)
        return np.array(state, dtype=np.float32)

    def _is_valid_play(self, suit: int, value: int, player: int) -> bool:
        return utils.is_valid_play(self.expeditions[player, suit], suit, value)

    def _calculate_score(self, player: int) -> int:
        """
        Calculate score for a player.
        """
        total_score = 0
        for suit in range(self.NUM_SUITS):
            expedition = self.expeditions[player, suit]
            total_score += utils.calculate_score(expedition)
        return total_score

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return the next state.
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
                if play_or_discard == 0:
                    self.expeditions[self.current_player, suit, value] -= 1
                else:
                    self.discard_piles[suit, value] -= 1
                self.player_hands[self.current_player, suit, value] += 1
                raise ValueError("Cannot draw from empty discard pile")
                
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
            reward = int(player_score - opponent_score)
            self.winner = 0 if player_score > opponent_score else (1 if opponent_score > player_score else -1)
        
        # Switch players
        self.current_player = 1 - self.current_player
        
        return self._get_state(), reward, self.game_over, {}

    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        return utils.get_valid_actions(
            self.player_hands, self.expeditions, self.discard_piles,
            self.current_player, self.NUM_SUITS, self.NUM_VALUES
        )

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