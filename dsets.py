from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# picklable identity transform
tf_identity = transforms.Lambda(lambda x: x)

class PCamDataset(Dataset):
    """Handles train/val (with label) and test (id-only)."""
    def __init__(self, df, img_dir, transform=None):
        self.df  = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tf  = transform if transform is not None else tf_identity

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_   = row["id"]
        label = row["label"] if "label" in row else 0
        img   = Image.open(self.dir / f"{id_}.tif")
        return self.tf(img), torch.tensor(label, dtype=torch.float32)
