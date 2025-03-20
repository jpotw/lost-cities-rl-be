import numpy as np
from typing import Tuple, Dict, List, Optional
from game import utils

class LostCitiesEnv:
    """
    Lost Cities game environment optimized with NumPy arrays
    """
    NUM_COLORS = 6
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
        self.deck = utils.create_deck(self.NUM_COLORS, self.CARDS_PER_HANDSHAKE)
        np.random.shuffle(self.deck)
        
        # 2 for each player
        # Player hands: shape (2, 6(NUM_COLORS), 10(NUM_VALUES)) - count of each card type
        self.player_hands = np.zeros((2, self.NUM_COLORS, self.NUM_VALUES), dtype=np.int8)
        
        # Expeditions: shape (2, 6(NUM_COLORS), 10(NUM_VALUES)) - count of each card type
        self.expeditions = np.zeros((2, self.NUM_COLORS, self.NUM_VALUES), dtype=np.int8)
        
        # 2 players share the same discard piles
        # Discard piles: shape (6(NUM_COLORS), 10(NUM_VALUES)) - count of each card type
        self.discard_piles = np.zeros((self.NUM_COLORS, self.NUM_VALUES), dtype=np.int8)
        
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

    def _get_state(self) -> np.ndarray:
        """
        Returns a numerical representation of the game state.
        NUM_COLORS = 6, NUM_VALUES = 10
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

    def _is_valid_play(self, color: int, value: int, player: int) -> bool:
        return utils.is_valid_play(self.expeditions[player, color], color, value)

    def _calculate_score(self, player: int) -> int:
        """
        Calculate score for a player.
        """
        total_score = 0
        for color in range(self.NUM_COLORS):
            expedition = self.expeditions[player, color]
            total_score += utils.calculate_score(expedition)
        return total_score

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return the next state.
        action: (card_index, play_or_discard, draw_source)
        """

        card_index, play_or_discard, draw_source = action
        
        # Convert card_index to color and value
        hand = self.player_hands[self.current_player]
        flat_index = np.where(hand.flatten() > 0)[0][card_index]
        color, value = np.unravel_index(flat_index, hand.shape)
        
        # Remove card from hand
        self.player_hands[self.current_player, color, value] -= 1
        
        # Play or discard
        if play_or_discard == 0:  # Play
            if not self._is_valid_play(color, value, self.current_player):
                self.player_hands[self.current_player, color, value] += 1
                raise ValueError("Invalid play")
            self.expeditions[self.current_player, color, value] += 1
        else:  # Discard
            self.discard_piles[color, value] += 1
        
        # Draw
        if draw_source == 0:  # Draw from deck
            if len(self.deck) == 0:
                self.game_over = True
            else:
                new_card = self.deck[0]
                self.deck = self.deck[1:]
                self.player_hands[self.current_player, new_card[0], new_card[1]] += 1
        else:  # Draw from discard pile
            discard_color = draw_source - 1
            if np.sum(self.discard_piles[discard_color]) == 0:
                # Reverse the play/discard
                if play_or_discard == 0:
                    self.expeditions[self.current_player, color, value] -= 1
                else:
                    self.discard_piles[color, value] -= 1
                self.player_hands[self.current_player, color, value] += 1
                raise ValueError("Cannot draw from empty discard pile")
                
            # Find the top card in the discard pile
            flat_index = np.where(self.discard_piles[discard_color] > 0)[0][-1]
            self.discard_piles[discard_color, flat_index] -= 1
            self.player_hands[self.current_player, discard_color, flat_index] += 1
        
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
        return utils.get_valid_actions(
            self.player_hands, self.expeditions, self.discard_piles,
            self.current_player, self.NUM_COLORS, self.NUM_VALUES
        )

    def render(self):
        """
        Text-based visualization of the game state.
        """
        print(f"\nPlayer {self.current_player}'s turn:")
        
        print("\nHands:")
        for player in range(2):
            cards = []
            for color in range(self.NUM_COLORS):
                for value in range(self.NUM_VALUES):
                    count = self.player_hands[player, color, value]
                    if count > 0:
                        cards.extend([(color, value)] * count)
            print(f"Player {player}: {cards}")
        
        print("\nExpeditions:")
        for player in range(2):
            exps = []
            for color in range(self.NUM_COLORS):
                cards = []
                for value in range(self.NUM_VALUES):
                    count = self.expeditions[player, color, value]
                    if count > 0:
                        cards.extend([(color, value)] * count)
                exps.append(cards)
            print(f"Player {player}: {exps}")
        
        print("\nDiscard Piles:")
        for color in range(self.NUM_COLORS):
            cards = []
            for value in range(self.NUM_VALUES):
                count = self.discard_piles[color, value]
                if count > 0:
                    cards.extend([(color, value)] * count)
            print(f"Color {color}: {cards}")
        
        print(f"\nDeck size: {len(self.deck)}")
        
        if self.game_over:
            if self.winner != -1:
                print(f"Winner: Player {self.winner}")
            else:
                print("Game ended in a draw")
                
    def get_player_hand(self, player_idx):
        """
        Return a list of cards in the player's hand.
        Each card is represented as (color, value).
        """
        cards = []
        for color in range(self.NUM_COLORS):
            for value in range(self.NUM_VALUES):
                count = self.player_hands[player_idx, color, value]
                if count > 0:
                    cards.extend([(color, value)] * count)
        return cards
    
    def index_to_color(self, index):
        """Convert color index to string"""
        colors = ["RED", "BLUE", "GREEN", "WHITE", "YELLOW", "PURPLE"]
        return colors[index]
