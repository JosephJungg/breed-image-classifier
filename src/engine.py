"""Training and evaluation loops."""

import numpy as np
import torch


def train_one_epoch(model, loader, loss_fn, optimizer, device, set_mode):
    """set_mode puts the model in the right train/eval mix before the epoch.

    The linear probe uses model.eval() so frozen BatchNorm stats stay fixed;
    fine-tuning uses src.model.set_train_mode.
    """
    set_mode(model)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                 # [B, num_classes] raw scores
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_predictions(model, loader, device):
    """True labels, predicted labels and confidence for every image, in loader order."""
    model.eval()
    y_true, y_pred, conf = [], [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        p, pred = probs.max(dim=1)
        y_true.append(y.numpy())
        y_pred.append(pred.cpu().numpy())
        conf.append(p.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(conf)
