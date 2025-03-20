import torch
import torch.nn as nn
import numpy as np

NUM_COLORS = 6
NUM_VALUES = 11  # 0 (handshake) and 2-10 (number cards)
STATE_SIZE = NUM_COLORS * NUM_VALUES * 4 + 1  # 4 flattened arrays (66 each) + deck size
ACTION_SIZE = 8 * 2 * 7  # 8 cards × 2 actions (play/discard) × 7 draw sources
HIDDEN_SIZE = 256

# Create a simple neural network
model = nn.Sequential(
    nn.Linear(STATE_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
)

# Save the model
torch.save(model.state_dict(), "model.pth")
print("Dummy model generated and saved as model.pth") 