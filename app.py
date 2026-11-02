"""Gradio demo: upload a cat or dog photo, get the top 3 breeds.

    python app.py            # then open http://127.0.0.1:7860

MODEL_CHECKPOINT overrides the checkpoint path. Each prediction is logged to
logs/predictions.csv (breeds, probabilities, timing; never the image itself).
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import gradio as gr

from src.predict import Predictor

ROOT = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", ROOT / "data" / "resnet50_ft_layer4_lr1e-5.pt")
LOG_FILE = ROOT / "logs" / "predictions.csv"

predictor = Predictor(CHECKPOINT)


def log_prediction(result):
    LOG_FILE.parent.mkdir(exist_ok=True)
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "top1", "p1", "top2", "p2", "top3", "p3",
                             "confident", "ms"])
        flat = [v for breed, p in result["top"] for v in (breed, f"{p:.4f}")]
        writer.writerow([datetime.now().isoformat(timespec="seconds"), *flat,
                         result["confident"], f"{result['ms']:.0f}"])


def classify(image):
    if image is None:
        return {}, "Upload a photo of a cat or dog."
    result = predictor.predict(image)
    log_prediction(result)

    breed, p = result["top"][0]
    if result["confident"]:
        note = f"Most likely a **{breed}** ({result['species']})."
    else:
        note = (f"**Not sure.** Best guess is {breed} ({p:.0%}), but the model is unsure. "
                "It may be a mixed breed, a breed it wasn't trained on, or not a cat or dog.")
    return {b: prob for b, prob in result["top"]}, note


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Cat or dog photo"),
    outputs=[gr.Label(num_top_classes=3, label="Top 3 breeds"), gr.Markdown()],
    title="Pet Breed Classifier",
    description=("Identifies 37 cat and dog breeds from the Oxford-IIIT Pet dataset. "
                 "ResNet-50, 92.5% test accuracy. Other breeds and mixed breeds will still "
                 "get a (wrong) breed name."),
    flagging_mode="never",   # flagging would save uploaded photos to disk
)

if __name__ == "__main__":
    demo.launch()
