import numpy as np
import pytest
import torch
from PIL import Image

from src.data import (DATA_DIR, NUM_CLASSES, check_split, eval_transform, load_pets,
                      load_split, make_split, train_transform)


def fake_labels(per_class=100, num_classes=NUM_CLASSES):
    return np.repeat(np.arange(num_classes), per_class)


def test_make_split_sizes_and_disjoint():
    labels = fake_labels()
    train_ix, val_ix = make_split(labels)
    assert len(train_ix) == 2960 and len(val_ix) == 740
    assert len(np.intersect1d(train_ix, val_ix)) == 0


def test_make_split_is_stratified():
    labels = fake_labels()
    train_ix, val_ix = make_split(labels)
    assert set(np.bincount(labels[train_ix])) == {80}
    assert set(np.bincount(labels[val_ix])) == {20}


def test_make_split_is_deterministic():
    labels = fake_labels()
    a, _ = make_split(labels, seed=42)
    b, _ = make_split(labels, seed=42)
    assert np.array_equal(a, b)


def test_check_split_accepts_valid_split():
    labels = fake_labels()
    check_split(*make_split(labels), labels)


def test_check_split_rejects_identical_train_and_val():
    # regression: train_ix.npy was once saved with the val indices
    labels = fake_labels()
    _, val_ix = make_split(labels)
    with pytest.raises(ValueError, match="share"):
        check_split(val_ix, val_ix, labels)


def test_check_split_rejects_missing_images():
    labels = fake_labels()
    train_ix, val_ix = make_split(labels)
    with pytest.raises(ValueError, match="covers"):
        check_split(train_ix[:-5], val_ix, labels)


def test_check_split_rejects_missing_class():
    labels = fake_labels()
    train_ix, val_ix = make_split(labels)
    val_without_0 = val_ix[labels[val_ix] != 0]
    train_with_0 = np.concatenate([train_ix, val_ix[labels[val_ix] == 0]])
    with pytest.raises(ValueError, match="val is missing"):
        check_split(train_with_0, val_without_0, labels)


def test_transforms_output_shape():
    img = Image.new("RGB", (500, 375), color=(120, 80, 40))
    assert train_transform()(img).shape == (3, 224, 224)
    assert eval_transform()(img).shape == (3, 224, 224)


def test_eval_transform_is_deterministic():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (375, 500, 3), dtype=np.uint8))
    assert torch.equal(eval_transform()(img), eval_transform()(img))


# checks on the real data; skipped when the dataset or split files are absent
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "train_ix.npy").exists()
    or not (DATA_DIR / "oxford-iiit-pet" / "annotations").exists(),
    reason="dataset or saved split not found in data/")


@needs_data
def test_saved_split_is_valid():
    labels = np.array(load_pets("trainval")._labels)
    train_ix, val_ix = load_split()
    check_split(train_ix, val_ix, labels)
    assert len(train_ix) == 2944 and len(val_ix) == 736


@needs_data
def test_saved_split_matches_make_split():
    labels = np.array(load_pets("trainval")._labels)
    train_ix, val_ix = load_split()
    expected_train, expected_val = make_split(labels)
    assert np.array_equal(train_ix, expected_train)
    assert np.array_equal(val_ix, expected_val)
