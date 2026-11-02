"""Train a linear probe or fine-tune layer4, logging every epoch.

Reproduce the final model (run 7):
    python -m src.train --arch resnet50 --stage probe --epochs 5 --name resnet50_linear_probe
    python -m src.train --arch resnet50 --stage finetune --init data/resnet50_linear_probe.pt \
        --lr-layer4 1e-5 --epochs 10 --name resnet50_ft_layer4_lr1e-5
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.data import DATA_DIR, check_split, load_pets, load_split, make_loaders
from src.engine import evaluate, train_one_epoch
from src.model import (build_model, count_params, load_checkpoint, pick_device,
                       set_train_mode, unfreeze_layer4)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--stage", required=True, choices=["probe", "finetune"])
    ap.add_argument("--init", help="checkpoint to start from (the probe, for finetune)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True, help="run name; checkpoint is data/<name>.pt")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "finetune" and not args.init:
        raise SystemExit("--stage finetune needs --init (the probe checkpoint)")

    torch.manual_seed(args.seed)
    device = pick_device()

    train_ix, val_ix = load_split()
    check_split(train_ix, val_ix, load_pets("trainval")._labels)
    train_loader, val_loader = make_loaders(train_ix, val_ix, batch_size=args.batch_size)

    model = build_model(args.arch)
    if args.init:
        load_checkpoint(model, args.init)
    if args.stage == "finetune":
        unfreeze_layer4(model)
    model = model.to(device)

    if args.stage == "probe":
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr_head)
        set_mode = lambda m: m.eval()     # frozen BatchNorm keeps its ImageNet stats
    else:
        optimizer = torch.optim.Adam([
            {"params": model.layer4.parameters(), "lr": args.lr_layer4},
            {"params": model.fc.parameters(),     "lr": args.lr_head},
        ])
        set_mode = set_train_mode

    trainable, total = count_params(model)
    print(f"{args.name}: {args.arch} {args.stage} on {device}, "
          f"trainable params {trainable:,} / {total:,}")

    loss_fn = nn.CrossEntropyLoss()
    ckpt = DATA_DIR / f"{args.name}.pt"
    RUNS_DIR.mkdir(exist_ok=True)
    epoch_log = RUNS_DIR / f"{args.name}.csv"

    best = {"val_acc": 0.0, "epoch": 0, "val_loss": None}
    with open(epoch_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                              device, set_mode)
            va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
            secs = time.time() - t0
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                             f"{va_loss:.4f}", f"{va_acc:.4f}", f"{secs:.0f}"])
            f.flush()

            saved = ""
            if va_acc > best["val_acc"]:
                best = {"val_acc": va_acc, "epoch": epoch, "val_loss": va_loss}
                torch.save(model.state_dict(), ckpt)
                saved = "  * saved"
            print(f"epoch {epoch}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f}  ({secs:.0f}s){saved}")

    # one summary row per run, so runs can be compared later
    summary = RUNS_DIR / "runs.csv"
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "name", "best_val_acc", "best_epoch", "best_val_loss",
                             "train_size", "val_size", "config"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), args.name,
                         f"{best['val_acc']:.4f}", best["epoch"], f"{best['val_loss']:.4f}",
                         len(train_ix), len(val_ix), json.dumps(vars(args))])

    print(f"best val acc {best['val_acc']:.4f} at epoch {best['epoch']}; saved {ckpt}")


if __name__ == "__main__":
    main()
