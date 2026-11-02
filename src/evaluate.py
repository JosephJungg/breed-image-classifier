"""Score a checkpoint on the validation or test split and print the error analysis.

    python -m src.evaluate --checkpoint data/resnet50_ft_layer4_lr1e-5.pt --split val

The test split is for the final, one-time evaluation only. Choosing models by test
score turns it into another validation set.
"""

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.analysis import report
from src.data import eval_transform, load_pets, load_split
from src.engine import collect_predictions, evaluate
from src.model import build_model, load_checkpoint, pick_device


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = pick_device()

    if args.split == "val":
        _, val_ix = load_split()
        dataset = Subset(load_pets("trainval", eval_transform()), val_ix)
        classes = dataset.dataset.classes
    else:
        dataset = load_pets("test", eval_transform())
        classes = dataset.classes
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    model = load_checkpoint(build_model(args.arch), args.checkpoint).to(device)
    loss, acc = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    y_true, y_pred, _ = collect_predictions(model, loader, device)

    print(f"{args.split}: {len(dataset)} images, loss {loss:.3f}\n")
    print(report(y_true, y_pred, classes))


if __name__ == "__main__":
    main()
