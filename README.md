# Breed Image Classifier

Breed classification (37 cat and dog breeds) on the
[Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Data split

The official `trainval` split is divided 80/20, stratified by breed (`random_state=42`):
2,944 train and 736 validation images. The official `test` split is held out for a
single final evaluation.

## Final result

**92.5% test accuracy** (274 mistakes out of 3,669; test loss 0.267) with ResNet-50,
`layer4` fine-tuned (run 7), evaluated once on the official test split at 224×224.

The test score is 2.9% below run 7's validation score (95.4%). Run 7 was chosen as the best
of many noisy validation scores, which builds in some luck. The test set is also five times
larger, so its estimate is more precise.

Evaluating run 7 at higher input resolutions (crops of 256, 288 and 352) did not clearly
improve validation accuracy, so the final evaluation uses 224.

On the test set (about 100 images per breed), the largest confusion is American Pit Bull
Terrier ↔ Staffordshire Bull Terrier (23 + 14 mistakes; Pit Bull accuracy 0.61). With only
about 20 validation images per breed, this pair never stood out during model selection.
Next are Birman ↔ Ragdoll (14 + 9), Abyssinian → Bengal (11) and Maine Coon → Ragdoll (11).

## Usage

The code lives in `src/`; `notebooks/explore1.ipynb` holds the original exploration.

```bash
# reproduce the final model (run 7)
python -m src.train --arch resnet50 --stage probe --epochs 5 --name resnet50_linear_probe
python -m src.train --arch resnet50 --stage finetune --init data/resnet50_linear_probe.pt \
    --lr-layer4 1e-5 --epochs 10 --name resnet50_ft_layer4_lr1e-5

# score a checkpoint and print the error analysis (val by default; test only once, at the end)
python -m src.evaluate --checkpoint data/resnet50_ft_layer4_lr1e-5.pt --split val

# tests
python -m pytest
```

### Demo app

```bash
python app.py     # then open http://127.0.0.1:7860
```

Upload a cat or dog photo to see the top 3 breeds. The app uses the run 7 checkpoint
(override with `MODEL_CHECKPOINT`) and the same preprocessing as evaluation. When the top
probability is below 0.7 it says it is not sure: on the validation set that flags 5.6% of
images, which are right only 66% of the time, versus 97% above the threshold. Each
prediction is logged to `logs/predictions.csv` (breeds, probabilities, timing). Uploaded
images are never stored, and Gradio's flagging is turned off for that reason.

Known limitation: the model only knows the 37 breeds and maps anything else to one of
them, often confidently. A blank grey image is predicted as a Bombay cat at 88%. The
confidence threshold catches uncertain pet photos, not inputs that aren't pets.

Training checks the train/val split before it starts and saves the best epoch to
`data/<name>.pt`. Each epoch is logged to `runs/<name>.csv`, and each run adds a summary
row with its full configuration to `runs/runs.csv`.

| Module | Contents |
|---|---|
| `src/data.py` | Transforms, dataset loading, the stratified split and its validity check |
| `src/model.py` | ResNet-18/50 with a 37-class head, freezing and BatchNorm mode helpers |
| `src/engine.py` | Training and evaluation loops, prediction collection |
| `src/analysis.py` | Per-breed accuracy, confused pairs, cat/dog mix-ups |
| `src/predict.py` | Single-image prediction with top 3 breeds and the confidence threshold |
| `app.py` | Gradio demo app |
| `src/train.py`, `src/evaluate.py` | Command-line entry points |

## Future work

Planned extensions (data catalog and validation, cloud storage, vector search with non-pet
detection, RAG) are listed in [ROADMAP.md](ROADMAP.md).

## Results

Validation accuracy on the 736-image validation split. One image is about 0.14%, so
differences under about 0.5% are within noise.

| # | Model | Trained layers | Epochs | Optimizer | Val loss | Val acc | Mistakes |
|---|---|---|---|---|---|---|---|
| 1 | ResNet-18 (ImageNet), linear probe | Final layer only | 5 | Adam, lr 1e-3 | 0.316 | 91.7% | 61 |
| 2 | Run 1, then fine-tune `layer4` | `layer4` + final layer | 10 (best: 10) | Adam, lr 1e-4 (`layer4`) / 1e-3 (final) | 0.295 | 92.3%* | 57 |
| 3 | Run 2 + rotation (±12°), brightness/contrast jitter (0.2) | `layer4` + final layer | 10 (best: 4) | Adam, lr 1e-4 (`layer4`) / 1e-3 (final) | 0.288 | 91.8%* | 60 |
| 4 | Run 3 with lower `layer4` learning rate | `layer4` + final layer | 10 (best: 5) | Adam, lr 1e-5 (`layer4`) / 1e-3 (final) | 0.263 | 92.3%* | 57 |
| 5 | ResNet-50 (ImageNet), linear probe† | Final layer only | 5 | Adam, lr 1e-3 | 0.287 | 95.1% | 36 |
| 6 | Run 5, then fine-tune `layer4` (run 4 settings) | `layer4` + final layer | 10 (best: 8) | Adam, lr 1e-5 (`layer4`) / 1e-3 (final) | 0.178 | 94.8%* | 38 |
| 7 | Run 6 with original augmentation (no rotation or jitter) | `layer4` + final layer | 10 (best: 10) | Adam, lr 1e-5 (`layer4`) / 1e-3 (final) | 0.174 | 95.4%* | 34 |

\* Best of 10 epochs, so slightly optimistic. Run 2's val accuracy ranged 90.4–92.3% across
epochs. Run 3 was trained twice with the same setup: best 91.7% and 91.8%, with epoch 10 at
89.3% and 90.8%. So run-to-run noise is about 1.5% at a given epoch, and runs 1–3 are within
noise of each other.

† Run 5 used run 3's augmentation (rotation and jitter), while run 1 used the original, so
runs 1 and 5 differ in both backbone and augmentation. The backbone likely explains most of
the +3.4%.

## Error analysis

### Run 1: linear probe

- Weakest breeds are mostly cats: British Shorthair 0.65, Abyssinian 0.70, Ragdoll 0.70.
  ImageNet has about 120 dog breeds but only about 5 cat classes, so the frozen features
  separate cat breeds less well.
- Top confusions are look-alikes: British Shorthair ↔ Russian Blue (8), Ragdoll → Birman (4),
  Abyssinian → Bengal (3), Bengal → Egyptian Mau (3), Boxer → American Bulldog (3).
- 1 cat/dog mix-up.

### Run 2: fine-tuned `layer4`

- Clear overfitting: train accuracy reached 99.5% while val stayed around 91–92% and val
  loss stayed flat (0.26–0.32). With about 80 training images per breed, `layer4`'s 8.4M
  weights memorize training photos.
- British Shorthair ↔ Russian Blue dropped from 8 to at most 4; British Shorthair rose from
  0.65 to at least 0.90.
- Errors shifted to colorpoint cats: Ragdoll → Birman (5), Siamese → Birman (2),
  Maine Coon → Ragdoll (2). Siamese fell to 0.80.
- Cat/dog mix-ups rose from 1 to 3.
- Net effect: 61 → 57 mistakes, about the same as the linear probe.

### Run 3: fine-tuned `layer4` with extra augmentation

- Augmentation slowed memorization (final train loss 0.040 vs 0.018) but train accuracy
  still reached 99%, and val loss drifted upward after epoch 2.
- Errors are more evenly spread: no breed below 0.80 (runs 1–2 had breeds at 0.65–0.70).
- Ragdoll ↔ Birman is now the largest confusion and goes both ways (4 + 3). Birmans'
  white paws may be cropped out or too small at 224×224.
- Boxer ↔ American Bulldog is also two-way (3 + 2). British Shorthair ↔ Russian Blue stayed
  improved (at most 5, vs 8 in run 1). Cat/dog mix-ups: 3.
- Net effect: 60 mistakes. Augmentation changed which images are wrong, not how many.

### Run 4: run 3 with `layer4` learning rate 1e-5

- Memorization slowed: train accuracy reached only 97.4% by epoch 10 (99% by epoch 6 in
  run 3). Val loss fell steadily to 0.249 and did not drift upward, the best of any run.
- Val accuracy did not improve beyond noise (90.9–92.3% over epochs 5–10).
- Most evenly spread errors so far: the worst breed is 0.84, and no confusion pair exceeds 3.
- Ragdoll ↔ Birman fell to 5 (3 + 2) from 7 in run 3. British Shorthair → Russian Blue: 3.
- Cat/dog mix-ups: 4, including Shiba Inu → Abyssinian (2), which share a reddish, fox-like look.
- Net effect: 57 mistakes. Runs 1–4 all land around 91–92%, which suggests a ceiling for
  ResNet-18 at 224×224 with about 80 training images per breed.

### Run 5: ResNet-50 linear probe

- 95.1% val accuracy, about 25 fewer mistakes than run 1: the first change clearly beyond
  noise. Val accuracy and loss were still improving at epoch 5.
- Small train/val gap (96.7% vs 95.1%).
- Breed-level error analysis: pending.

### Run 6: ResNet-50 fine-tuned `layer4`

- No accuracy gain over run 5 (94.8%, best of 10, vs 95.1%), but much lower val loss
  (0.178 at the best epoch, lowest 0.173, vs 0.287): more confident when right, less
  confidently wrong.
- Still memorizes: train accuracy reached 99.5% even at lr 1e-5 (`layer4` has about 15M
  weights). Val accuracy dipped to 93.3% in epoch 1 before recovering.
- First run with 0 cat/dog mix-ups.
- The same hard pairs persist with fewer mistakes: Ragdoll → Birman (4),
  British Shorthair → Russian Blue (4), Abyssinian → Bengal (3), Boxer → American Bulldog (3).
- Bengal absorbs errors from Abyssinian (3), Maine Coon (2) and Egyptian Mau (2).
- Beagle / Basset Hound flipped direction: Basset Hound → Beagle (3).
- Net effect: 38 mistakes. Across both backbones, fine-tuning `layer4` improves confidence
  but not accuracy with about 80 images per breed.

### Run 7: run 6 with the original augmentation

- Same starting probe and settings as run 6; only the augmentation differs.
- Highest score so far (95.4%, 34 mistakes), but +0.6% over run 6 is within noise.
- Slightly more memorization without rotation and jitter (train 99.7% vs 99.0%), with no
  cost on validation. Val loss about the same (0.174 vs 0.178).
- Errors into Bengal fell from 7 to 2 (only Egyptian Mau → Bengal remains). Hypothesis:
  brightness/contrast jitter blurs coat-pattern cues between tabby breeds. Needs a repeat
  run to confirm.
- Hard pairs are all small: British Shorthair ↔ Russian Blue (3 + 2), Ragdoll ↔ Birman
  (2 + 2), Boxer → American Bulldog (3). Cat/dog mix-ups: 0.
