"""Reusable torch-Dataset utilities.

Provides:
    PCamDataset – thin wrapper around a (id, label) dataframe that
    loads images from disk, applies an optional transform and returns
    (image_tensor, label_tensor).

The class is intentionally lightweight so it can be copy-pasted into
any vision project that stores images as <id>.tif or <id>.png files.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class PCamDataset(Dataset):
    """Generic image-classification Dataset.

    Args:
        df (pd.DataFrame): must contain an 'id' column and,
            optionally, a 'label' column. Keeping the label optional
            makes the same class usable for *test-time* datasets
            where ground-truth is absent.
        img_dir (Path | str): root directory that holds image files.
        transform (callable, optional): torchvision-style transform
            applied to the PIL image before returning.
    """
    def __init__(self, df, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)      # ensure 0-based index
        self.img_dir = Path(img_dir)
        self.tf = transform or (lambda x: x)     # identity if no tf given

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # -- fetch row information
        row = self.df.iloc[idx]
        id_   = row["id"]
        # if labels are absent (e.g., Kaggle test set) default to 0
        label = row["label"] if "label" in row else 0

        # -- load the raw image
        img   = Image.open(self.img_dir / f"{id_}.tif")

        # -- return transformed image + label as float tensor
        return self.tf(img), torch.tensor(label, dtype=torch.float32)
