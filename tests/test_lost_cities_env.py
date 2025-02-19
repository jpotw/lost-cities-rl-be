import unittest
import numpy as np
from game.lost_cities_env import LostCitiesEnv
from game.utils import create_deck

class TestLostCitiesEnv(unittest.TestCase):
    def setUp(self):
        """Set up a new environment before each test."""
        self.env = LostCitiesEnv()
        
    def test_initialization(self):
        """Test if the environment is properly initialized."""
        state = self.env.reset()
        
        # Test state shape
        self.assertEqual(state.shape[0], 6 * 11 * 4 + 1)  # hands + expeditions + discard + deck size
        
        # Test initial hands
        for player in range(2):
            self.assertEqual(np.sum(self.env.player_hands[player]), 8)  # Each player starts with 8 cards
            
        # Test deck size
        self.assertEqual(len(self.env.deck), 56)  # 72 total - 16 dealt cards (8 per player)
        
        # Test empty expeditions and discard piles
        self.assertTrue(np.all(self.env.expeditions == 0))
        self.assertTrue(np.all(self.env.discard_piles == 0))

    def test_deck_creation(self):
        """Test if the deck is created with correct number and types of cards."""
        deck = create_deck(self.env.NUM_SUITS, self.env.CARDS_PER_HANDSHAKE)
        
        # Test deck size
        self.assertEqual(len(deck), 72)
        
        # Test number of handshake cards per suit
        for suit in range(self.env.NUM_SUITS):
            handshake_count = np.sum((deck[:, 0] == suit) & (deck[:, 1] == 0))
            self.assertEqual(handshake_count, 3)
            
            # Test number cards (2-10) per suit
            for value in range(2, 11):
                card_count = np.sum((deck[:, 0] == suit) & (deck[:, 1] == value))
                self.assertEqual(card_count, 1)

    def test_valid_play_rules(self):
        """Test the rules for valid card plays."""
        # Test empty expedition
        self.assertTrue(self.env._is_valid_play(0, 5, 0))
        
        # Set up an expedition with a handshake
        self.env.expeditions[0, 0, 0] = 1
        
        # Test playing number card after handshake
        self.assertTrue(self.env._is_valid_play(0, 5, 0))
        
        # Set up expedition with number cards
        self.env.expeditions[0, 0, 5] = 1  # Card value 5
        
        # Test invalid plays
        self.assertFalse(self.env._is_valid_play(0, 4, 0))  # Lower value
        self.assertFalse(self.env._is_valid_play(0, 0, 0))  # Handshake after number
        
        # Test valid plays
        self.assertTrue(self.env._is_valid_play(0, 6, 0))  # Higher value

    def test_score_calculation(self):
        """Test score calculation logic."""
        # Set up a simple expedition
        self.env.expeditions[0, 0, 5] = 1  # Value 5
        self.env.expeditions[0, 0, 6] = 1  # Value 6
        
        # Calculate score: -20 + (5 + 6) = -9
        self.assertEqual(self.env._calculate_score(0), -9)
        
        # Add handshake
        self.env.expeditions[0, 0, 0] = 1
        # Score with multiplier: (-20 + 11) * 2 = -18
        self.assertEqual(self.env._calculate_score(0), -18)
        
        # Test bonus for 8+ cards
        self.env.expeditions[0, 0, :] = 1  # Fill expedition
        score = self.env._calculate_score(0)
        self.assertTrue(score > 0)  # Should be positive with bonus

    def test_valid_actions_generation(self):
        """Test generation of valid actions."""
        # Reset to known state
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        
        # Should have actions for each card in hand
        self.assertTrue(len(valid_actions) > 0)
        
        # Test action format
        for action in valid_actions:
            self.assertEqual(len(action), 3)
            card_index, play_or_discard, draw_source = action
            self.assertIsInstance(card_index, (int, np.integer))
            self.assertIn(play_or_discard, [0, 1])
            self.assertGreaterEqual(draw_source, 0)
            self.assertLess(draw_source, 6)

    def test_game_flow(self):
        """Test complete game flow."""
        self.env.reset()
        done = False
        moves = 0
        
        while not done and moves < 100:  # Prevent infinite loop in test
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
                
            action = valid_actions[0]  # Take first valid action
            state, reward, done, _ = self.env.step(action)
            
            # Test state validity
            self.assertIsInstance(state, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            
            moves += 1
        
        # Game should end
        self.assertTrue(done or moves == 100)
        
        if done:
            # Winner should be set
            self.assertIn(self.env.winner, [-1, 0, 1])

    def test_invalid_actions(self):
        """Test handling of invalid actions."""
        self.env.reset()
        
        # Test invalid card index
        with self.assertRaises(IndexError):
            self.env.step((100, 0, 0))
            
        # Test invalid draw from empty discard
        with self.assertRaises(ValueError):
            self.env.step((0, 0, 1))  # Try to draw from empty discard pile

if __name__ == '__main__':
    unittest.main()
