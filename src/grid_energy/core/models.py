import torch
import torch.nn as nn
import torch.nn.functional as F

class NonogramCNN(nn.Module):
    def __init__(self, grid_size=12):
        super().__init__()
        self.grid_size = grid_size
        
        self.hint_encoder = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, grid_size * grid_size),
        )
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, grid, hints):
        batch_size = grid.size(0)
        
        h_feat = self.hint_encoder(hints.view(batch_size, -1))
        h_grid = h_feat.view(batch_size, 1, self.grid_size, self.grid_size)
        
        x = torch.cat([grid, h_grid], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x.mean(dim=(1, 2, 3))