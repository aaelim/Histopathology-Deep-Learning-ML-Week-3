# src/dataset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class PCamDataset(Dataset):
    def __init__(self, df, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.tf = transform or (lambda x: x)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_   = row["id"]
        label = row["label"] if "label" in row else 0
        img   = Image.open(self.img_dir/f"{id_}.tif")
        return self.tf(img), torch.tensor(label, dtype=torch.float32)
