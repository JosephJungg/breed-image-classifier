"""Dataset, split and transforms for Oxford-IIIT Pet."""

from pathlib import Path

import numpy as np
# torch must be imported before sklearn: in the reverse order their two OpenMP runtimes
# conflict on macOS and the first image transform segfaults
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split  # noqa: E402  (after torch, see above)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 37

CATS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
        "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}


def train_transform(size=224):
    # original augmentation (run 7); rotation and color jitter did not help
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_pets(split, transform=None, root=DATA_DIR, download=False):
    """split is "trainval" or "test", the two official splits."""
    return OxfordIIITPet(root=root, split=split, target_types="category",
                         transform=transform, download=download)


def make_split(labels, val_size=0.2, seed=42):
    """Stratified train/val indices into the trainval split."""
    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))
    train_ix, val_ix = train_test_split(all_idx, test_size=val_size, stratify=labels,
                                        random_state=seed)
    return train_ix, val_ix


def save_split(train_ix, val_ix, data_dir=DATA_DIR):
    np.save(Path(data_dir) / "train_ix.npy", train_ix)
    np.save(Path(data_dir) / "val_ix.npy", val_ix)


def load_split(data_dir=DATA_DIR):
    train_ix = np.load(Path(data_dir) / "train_ix.npy")
    val_ix   = np.load(Path(data_dir) / "val_ix.npy")
    return train_ix, val_ix


def check_split(train_ix, val_ix, labels):
    """Raise if the split is broken: overlap, wrong total, or a class missing from either side."""
    labels = np.asarray(labels)
    overlap = np.intersect1d(train_ix, val_ix)
    if len(overlap):
        raise ValueError(f"train and val share {len(overlap)} indices")
    if len(train_ix) + len(val_ix) != len(labels):
        raise ValueError(f"split covers {len(train_ix) + len(val_ix)} of {len(labels)} images")
    for name, ix in [("train", train_ix), ("val", val_ix)]:
        missing = set(np.unique(labels)) - set(np.unique(labels[ix]))
        if missing:
            raise ValueError(f"{name} is missing classes {sorted(missing)}")


def make_loaders(train_ix, val_ix, batch_size=32, root=DATA_DIR, num_workers=0):
    # same images on disk, two different transforms
    base_train = load_pets("trainval", train_transform(), root)
    base_eval  = load_pets("trainval", eval_transform(), root)
    train_loader = DataLoader(Subset(base_train, train_ix), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(Subset(base_eval, val_ix), batch_size=64,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def make_test_loader(root=DATA_DIR, num_workers=0):
    return DataLoader(load_pets("test", eval_transform(), root), batch_size=64,
                      shuffle=False, num_workers=num_workers)
