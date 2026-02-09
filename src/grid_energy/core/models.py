import torch
import torch.nn.functional as F
from torch import nn

HIDDEN_DIM = 512
CHANNELS = 256
GRID_SIZE = 5
HINT_DIM = 20
HINT_ENCODER_LAYER_1_OUTPUT = HIDDEN_DIM
HINT_ENCODER_LAYER_2_OUTPUT = HIDDEN_DIM
HINT_ENCODER_OUTPUT_FEATURES = GRID_SIZE * GRID_SIZE
CONV1_IN_CHANNELS = 2
CONV1_OUT_CHANNELS = CHANNELS
CONV1_KERNEL_SIZE = 3
CONV1_PADDING = 1
CONV2_IN_CHANNELS = CHANNELS
CONV2_OUT_CHANNELS = CHANNELS
CONV2_KERNEL_SIZE = 3
CONV2_PADDING = 1
CONV3_IN_CHANNELS = CHANNELS
CONV3_OUT_CHANNELS = CHANNELS
CONV3_KERNEL_SIZE = 3
CONV3_PADDING = 1
CONV4_IN_CHANNELS = CHANNELS
CONV4_OUT_CHANNELS = 1
CONV4_KERNEL_SIZE = 1
CONV4_PADDING = 0
ACTIVATION_FUNCTION = F.relu
SCALE_INITIAL_VALUE = 1.0


class NonogramCNN(nn.Module):
    def __init__(self, grid_size=GRID_SIZE, hint_dim=HINT_DIM):
        super().__init__()
        self.grid_size = grid_size
        self.hint_dim = hint_dim
        
        self.hint_encoder = nn.Sequential(
            nn.Linear(self.hint_dim, HINT_ENCODER_LAYER_1_OUTPUT),
            nn.ReLU(),
            nn.Linear(HINT_ENCODER_LAYER_1_OUTPUT, HINT_ENCODER_LAYER_2_OUTPUT),
            nn.ReLU(),
            nn.Linear(HINT_ENCODER_LAYER_2_OUTPUT, self.grid_size * self.grid_size),
        )
        
        self.conv1 = nn.Conv2d(
            CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS,
            kernel_size=CONV1_KERNEL_SIZE, padding=CONV1_PADDING
        )
        self.conv2 = nn.Conv2d(
            CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS,
            kernel_size=CONV2_KERNEL_SIZE, padding=CONV2_PADDING
        )
        self.conv3 = nn.Conv2d(
            CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS,
            kernel_size=CONV3_KERNEL_SIZE, padding=CONV3_PADDING
        )
        self.conv4 = nn.Conv2d(
            CONV4_IN_CHANNELS, CONV4_OUT_CHANNELS,
            kernel_size=CONV4_KERNEL_SIZE, padding=CONV4_PADDING
        )

        self.scale = nn.Parameter(torch.tensor(SCALE_INITIAL_VALUE))

    def forward(self, grid, hints):
        batch_size = grid.size(0)
        
        hints_flat = hints.view(batch_size, -1)
        if hints_flat.size(1) != self.hint_dim:
            raise ValueError(f"Hint dimension mismatch. Expected {self.hint_dim}, got {hints_flat.size(1)}")
        
        h_feat = self.hint_encoder(hints_flat)
        h_grid = h_feat.view(batch_size, 1, self.grid_size, self.grid_size)

        x = torch.cat([grid, h_grid], dim=1)
        x = ACTIVATION_FUNCTION(self.conv1(x))
        x = ACTIVATION_FUNCTION(self.conv2(x))
        x = ACTIVATION_FUNCTION(self.conv3(x))
        x = self.conv4(x)

        return x.mean(dim=(1, 2, 3)) * self.scale
    
    def predict_grid(self, hints):
        batch_size = hints.size(0)
        hints_flat = hints.view(batch_size, -1)
        h_feat = self.hint_encoder(hints_flat)
        grid_pred = torch.sigmoid(h_feat.view(batch_size, 1, self.grid_size, self.grid_size) * 3.0)
        return (grid_pred - 0.5) * 2.0