import os, argparse
import torch
from torch.utils.data import DataLoader, random_split
from monai.transforms import Compose
from utils.dataset import PatchFolder
from utils.transforms import train_transforms, val_transforms
from utils.model import build_model

def main(args):
    ds = PatchFolder(args.data_dir, transform=None)
    n = len(ds)
    n_val = max(1, int(0.2*n))
    n_train = n - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])

    t_train = train_transforms(args.imgsz)
    t_val = val_transforms(args.imgsz)

    # Wrap to apply transforms
    class Wrap(torch.utils.data.Dataset):
        def __init__(self, base, T):
            self.base = base; self.T = T
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            sample = self.base[i]
            return self.T(sample)

    dl_train = DataLoader(Wrap(ds_train, t_train), batch_size=args.batch, shuffle=True, num_workers=2)
    dl_val = DataLoader(Wrap(ds_val, t_val), batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.backbone, num_classes=args.num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs('models', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0; cnt = 0
        for batch in dl_train:
            x = batch["img"].to(device)
            y = batch["label"].to(device)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            tot += loss.item()*x.size(0); cnt += x.size(0)
        tr_loss = tot/cnt if cnt else 0.0

        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["img"].to(device)
                y = batch["label"].to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total if total else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={tr_loss:.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "models/best.pth")
            print("Saved models/best.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/patches")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
