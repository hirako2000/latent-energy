import torch
from torch.utils.data import Dataset, DataLoader
from grid_energy.utils.config import settings

class CroissantDataset(Dataset):
    def __init__(self):
        path = settings.GOLD_DIR / "croissant_dataset.pt"
        if not path.exists():
            raise FileNotFoundError(f"Gold dataset missing at {path}")
        
        
        data = torch.load(path, weights_only=False)
        
        self.ids = data["ids"]
        self.grids = data["grids"].float()
        self.row_hints = data["row_hints"].float()
        self.col_hints = data["col_hints"].float()
        self.metadata = data["metadata"]
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        target_grid = self.grids[idx].unsqueeze(0)
        hints = torch.stack([self.row_hints[idx], self.col_hints[idx]], dim=0)
        return {
            "initial_state": torch.randn_like(target_grid),
            "target_grid": target_grid,
            "hints": hints,
            "id": self.ids[idx]
        }

def get_dataloader(batch_size=32, shuffle=True):
    dataset = CroissantDataset()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0 # would be >0 for production, 0 is safer for debugging
    )