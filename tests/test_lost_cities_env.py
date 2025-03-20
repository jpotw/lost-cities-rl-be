import unittest
import numpy as np
from game.lost_cities_env import LostCitiesEnv
from game.utils import create_deck

class TestLostCitiesEnv(unittest.TestCase):
    def setUp(self):
        """Set up a new environment before each test."""
        self.env = LostCitiesEnv()
        
    def test_reset(self):
        """Test that reset initializes the environment correctly"""
        state = self.env.reset()
        
        # Test state shape
        self.assertEqual(len(state), 241)  # 4 * 60 + 1
        
        # Test player hands
        self.assertEqual(np.sum(self.env.player_hands[0]), 8)  # Each player starts with 8 cards
        self.assertEqual(np.sum(self.env.player_hands[1]), 8)
        
        # Test expeditions and discard piles are empty
        self.assertEqual(np.sum(self.env.expeditions), 0)
        self.assertEqual(np.sum(self.env.discard_piles), 0)
        
        # Test deck size
        self.assertEqual(len(self.env.deck), 56)  # 72 - 16 (initial hands)

    def test_deck_creation(self):
        """Test that deck is created with correct number of cards"""
        deck = create_deck(self.env.NUM_COLORS, self.env.CARDS_PER_HANDSHAKE)
        
        # Test total number of cards
        self.assertEqual(len(deck), 72)  # 6 colors * (3 handshake + 9 number cards)
        
        # Test number of handshake cards per color
        for color in range(self.env.NUM_COLORS):
            handshake_count = np.sum((deck[:, 0] == color) & (deck[:, 1] == 0))
            self.assertEqual(handshake_count, 3)
            
            # Test number cards (2-10) per color
            for value in range(2, 11):
                card_count = np.sum((deck[:, 0] == color) & (deck[:, 1] == value))
                self.assertEqual(card_count, 1)

    def test_valid_play(self):
        """Test valid play conditions"""
        # Test empty expedition
        self.assertTrue(self.env._is_valid_play(np.zeros(11), 0, 0))
        
        # Test handshake on empty expedition
        expedition = np.zeros(11)
        self.assertTrue(self.env._is_valid_play(expedition, 0, 0))
        
        # Test handshake after number card (invalid)
        expedition = np.zeros(11)
        expedition[5] = 1  # Add a 5
        self.assertFalse(self.env._is_valid_play(expedition, 0, 0))
        
        # Test ascending number cards
        expedition = np.zeros(11)
        expedition[5] = 1  # Add a 5
        self.assertTrue(self.env._is_valid_play(expedition, 0, 6))  # Add a 6
        self.assertFalse(self.env._is_valid_play(expedition, 0, 4))  # Try to add a 4

    def test_score_calculation(self):
        """Test expedition scoring"""
        # Empty expedition
        expedition = np.zeros(11)
        self.assertEqual(self.env._calculate_score(expedition), 0)
        
        # Single number card
        expedition = np.zeros(11)
        expedition[5] = 1  # Add a 5
        self.assertEqual(self.env._calculate_score(expedition), -15)  # 5 - 20
        
        # Multiple number cards
        expedition = np.zeros(11)
        expedition[5] = 1  # Add a 5
        expedition[6] = 1  # Add a 6
        self.assertEqual(self.env._calculate_score(expedition), -9)  # (5 + 6) - 20
        
        # With handshake multiplier
        expedition = np.zeros(11)
        expedition[0] = 1  # Add a handshake
        expedition[5] = 1  # Add a 5
        expedition[6] = 1  # Add a 6
        self.assertEqual(self.env._calculate_score(expedition), -18)  # ((5 + 6) - 20) * 2

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
