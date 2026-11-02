import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import collect_predictions, evaluate, train_one_epoch


class FixedModel(nn.Module):
    """Always returns the given logits, one row per input."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits[x[:, 0].long()]


def make_loader(n=10, batch_size=4):
    x = torch.arange(n, dtype=torch.float32).unsqueeze(1)   # x[i] = i, used as a row index
    y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_evaluate_counts_accuracy_across_uneven_batches():
    logits = torch.zeros(10, 3)
    logits[:7, 0] = 5.0      # first 7 predict class 0 (correct)
    logits[7:, 1] = 5.0      # last 3 predict class 1 (wrong)
    _, acc = evaluate(FixedModel(logits), make_loader(), nn.CrossEntropyLoss(), "cpu")
    assert acc == 0.7


def test_evaluate_loss_is_a_per_image_average():
    logits = torch.zeros(10, 3)   # uniform over 3 classes: loss ln(3) for every image
    loss, _ = evaluate(FixedModel(logits), make_loader(), nn.CrossEntropyLoss(), "cpu")
    assert abs(loss - np.log(3)) < 1e-6


def test_collect_predictions_keeps_loader_order():
    logits = torch.zeros(10, 3)
    logits[torch.arange(10), torch.arange(10) % 3] = 2.0
    y_true, y_pred, conf = collect_predictions(FixedModel(logits), make_loader(), "cpu")
    assert y_pred.tolist() == [i % 3 for i in range(10)]
    assert y_true.tolist() == [0] * 10
    assert np.all((conf > 1 / 3) & (conf <= 1))


def test_train_one_epoch_reduces_loss():
    torch.manual_seed(0)
    x = torch.randn(64, 5)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = nn.Linear(5, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    first, _ = train_one_epoch(model, loader, loss_fn, optimizer, "cpu", lambda m: m.train())
    for _ in range(5):
        last, acc = train_one_epoch(model, loader, loss_fn, optimizer, "cpu", lambda m: m.train())
    assert last < first
    assert acc > 0.9


def test_train_one_epoch_applies_set_mode():
    calls = []
    model = nn.Linear(1, 2)
    loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long)),
                        batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, "cpu",
                    lambda m: calls.append(m))
    assert calls == [model]
