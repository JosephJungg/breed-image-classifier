"""Predict the breed of a single image with a trained checkpoint."""

import time

import torch

from src.data import CATS, CLASSES, eval_transform
from src.model import build_model, load_checkpoint, pick_device

# below this top probability the app says it is not sure. On the validation set, run 7
# flags 5.6% of images at 0.7; those are right only 66% of the time (vs 97% above it)
CONFIDENCE_THRESHOLD = 0.7


def top_k(probs, classes=CLASSES, k=3):
    """[(breed, probability)] for the k most likely breeds, highest first."""
    values, indices = torch.topk(probs, k)
    return [(classes[i], float(v)) for i, v in zip(indices.tolist(), values)]


def species(breed, cats=CATS):
    return "cat" if breed in cats else "dog"


class Predictor:
    def __init__(self, checkpoint, arch="resnet50", device=None,
                 threshold=CONFIDENCE_THRESHOLD):
        self.device = device or pick_device()
        self.model = load_checkpoint(build_model(arch, pretrained=False), checkpoint)
        self.model = self.model.to(self.device).eval()
        self.transform = eval_transform()   # must match evaluation, or accuracy quietly drops
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image, k=3):
        """image is a PIL image. Returns a dict with the top breeds and timing."""
        t0 = time.perf_counter()
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        top = top_k(probs, k=k)
        return {
            "top": top,
            "species": species(top[0][0]),
            "confident": top[0][1] >= self.threshold,
            "ms": (time.perf_counter() - t0) * 1000,
        }
