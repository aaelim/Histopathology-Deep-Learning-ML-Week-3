"""Standalone inference / submission generator.

Example:
    python infer.py --data_dir data --batch 256 --workers 8
"""
import argparse, torch, pandas as pd, multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import PCamDataset
from model   import get_model


def main(args):
    # Force 'spawn' so that DataLoader workers behave the same on all OSes
    mp.set_start_method("spawn", force=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Paths 
    DATA_DIR = Path(args.data_dir)
    TEST_DIR = DATA_DIR / "test"
    sub      = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # Dataset / Loader 
    tf = transforms.Compose([transforms.ToTensor()])  # keep preprocessing minimal
    test_ds = PCamDataset(sub[['id']], TEST_DIR, tf)
    test_dl = DataLoader(
        test_ds,
        batch_size     = args.batch,
        shuffle        = False,  # preserve row order for Kaggle submission
        num_workers    = args.workers,
        pin_memory     = True,
        persistent_workers = True,
    )

    #  Model 
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("best.pt", map_location=DEVICE))
    model.eval()

    #  Predict 
    preds = []
    with torch.no_grad():
        for x, _ in tqdm(test_dl):
            preds.append(model(x.to(DEVICE)).sigmoid().cpu())

    # Stack → numpy so that pandas writes float values directly
    sub["label"] = torch.cat(preds).numpy()
    sub.to_csv("submission.csv", index=False)
    print("saved submission.csv")


if __name__ == "__main__":
    # suppress torchvision PNG warning noise
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)

    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", default="data")
    arg.add_argument("--batch",   type=int, default=512)
    arg.add_argument("--workers", type=int, default=4)
    main(arg.parse_args())

