"""End-to-end training script with mixed-precision & cosine LR.

Generic enough to drop into any binary image-classification project:
just adjust the transforms + get_model() call where necessary.
"""
import os, argparse, time, multiprocessing as mp, torch, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from dataset import PCamDataset
from model   import get_model


def main(args):
    torch.manual_seed(42)  # ensure deterministic train/val split

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DATA: split train/val stratified 80/20
    DATA_DIR  = Path(args.data_dir)
    TRAIN_DIR = DATA_DIR / "train"
    df = pd.read_csv(DATA_DIR / "train_labels.csv")

    #  basic aug for small 96×96 tiles
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
    ])
    tf_valid = transforms.Compose([transforms.ToTensor()])

    # stratified split keeps class balance intact
    val_df = df.groupby('label', group_keys=False, as_index=False)\
               .apply(lambda x: x.sample(int(0.2 * len(x)), random_state=42))
    train_df = df.drop(val_df.index)

    train_ds = PCamDataset(train_df, TRAIN_DIR, tf_train)
    val_ds   = PCamDataset(val_df,   TRAIN_DIR, tf_valid)

    # torch DataLoader workers need 'spawn' on Windows / Jupyter
    mp.set_start_method("spawn", force=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_dl   = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 2) MODEL + OPTIMISER
    model  = get_model().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()  # mixed precision → faster on Ampere GPUs
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 3) TRAIN / EVAL LOOP
    def run_epoch(dl, train=True):
        """One pass over `dl`. Returns (mean_loss, roc_auc)."""
        model.train(train)
        tot, ys, ps = 0, [], []
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)  # BCE expects shape (N, 1)

            with torch.set_grad_enabled(train):
                with autocast():            # ← half-precision region
                    out  = model(x)
                    loss = loss_fn(out, y)

            if train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            # accumulate for metrics
            tot += loss.item() * len(x)
            ys.append(y.detach().cpu())
            ps.append(out.sigmoid().detach().cpu())

        auc = roc_auc_score(torch.cat(ys), torch.cat(ps))
        return tot / len(dl.dataset), auc

    best_auc = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(train_dl, train=True)
        val_loss, val_auc = run_epoch(val_dl, train=False)
        sch.step()  # cosine schedule

        print(f"E{ep:02d} {time.time() - t0:4.0f}s "
              f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        # save the best checkpoint for later inference
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--batch",    type=int, default=512)
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--epochs",   type=int, default=5)
    main(p.parse_args())

