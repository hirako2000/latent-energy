import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 512
CHANNELS = 256

class NonogramCNN(nn.Module):
    def __init__(self, grid_size=12):
        super().__init__()
        self.grid_size = grid_size
        
        self.hint_encoder = nn.Sequential(
            nn.Linear(96, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, grid_size * grid_size),
        )
        
        self.conv1 = nn.Conv2d(2, CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(CHANNELS, 1, kernel_size=1)
        
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, grid, hints):
        batch_size = grid.size(0)
        
        h_feat = self.hint_encoder(hints.view(batch_size, -1))
        h_grid = h_feat.view(batch_size, 1, self.grid_size, self.grid_size)
        
        x = torch.cat([grid, h_grid], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        # scaling for better gradient flow
        return x.mean(dim=(1, 2, 3)) * self.scale