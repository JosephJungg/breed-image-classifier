import pytest
import torch
from PIL import Image

from src.data import CLASSES, DATA_DIR, NUM_CLASSES, load_pets
from src.model import build_model
from src.predict import Predictor, species, top_k


def test_top_k_highest_first():
    probs = torch.zeros(NUM_CLASSES)
    probs[CLASSES.index("Birman")] = 0.6
    probs[CLASSES.index("Ragdoll")] = 0.3
    probs[CLASSES.index("Siamese")] = 0.1
    assert top_k(probs) == [("Birman", pytest.approx(0.6)), ("Ragdoll", pytest.approx(0.3)),
                            ("Siamese", pytest.approx(0.1))]


def test_species():
    assert species("Russian Blue") == "cat"
    assert species("Beagle") == "dog"


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save(build_model("resnet18", pretrained=False).state_dict(), path)
    return path


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_predict_handles_any_image_mode(checkpoint, mode):
    predictor = Predictor(checkpoint, arch="resnet18", device="cpu")
    result = predictor.predict(Image.new(mode, (500, 375)))
    assert len(result["top"]) == 3
    assert all(breed in CLASSES for breed, _ in result["top"])
    probs = [p for _, p in result["top"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-6
    assert result["species"] in {"cat", "dog"}


def test_threshold_decides_confidence(checkpoint):
    image = Image.new("RGB", (300, 300))
    assert Predictor(checkpoint, "resnet18", "cpu", threshold=0.0).predict(image)["confident"]
    assert not Predictor(checkpoint, "resnet18", "cpu", threshold=1.01).predict(image)["confident"]


@pytest.mark.skipif(not (DATA_DIR / "oxford-iiit-pet" / "annotations").exists(),
                    reason="dataset not found in data/")
def test_classes_match_dataset_label_order():
    assert CLASSES == load_pets("trainval").classes
