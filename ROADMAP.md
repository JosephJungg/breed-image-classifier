# Roadmap

Possible extensions, recorded 2026-11-02. None are started.

## Recommended first

1. **Data catalog, validation and cloud storage** (data engineering)
2. **Vector search with non-pet detection** (retrieval). Fixes a known flaw: a blank grey
   image is predicted as a Bombay cat at 88% confidence.

## Data engineering

- **Metadata catalog.** One row per image (path, breed, species, width, height, color mode,
  file size, content hash, split), stored as Parquet and queried with DuckDB. The split
  becomes a versioned table instead of loose `.npy` files.
- **Data validation.** Automated checks before training: corrupt or non-RGB images,
  duplicate images across train/val/test (by content hash), label counts per breed.
  Extends the existing `check_split`.
- **Pipeline.** Repeatable stages: download → validate → catalog → split → train → evaluate.
  Start with scripts or a Makefile; Prefect or Dagster if it grows.
- **Data versioning (DVC).** Tie each checkpoint to the exact data and split that produced it.
- **Feedback loop.** App prediction logs and "was this right?" answers flow into a labeled
  store, pass the same validation, and feed periodic retraining.

## Cloud data platforms

- **Cloud storage** (S3 or GCS) for the dataset, checkpoints and logs. Also fixes the fact
  that cloning the repo gives no checkpoint.
- **Cloud warehouse** (BigQuery, Snowflake or MotherDuck) for the catalog and prediction
  logs, queryable with SQL.
- **Public deployment** (Hugging Face Spaces or Cloud Run) for a shareable app link. Needs
  the run 7 checkpoint (about 95 MB) uploaded.

## Retrieval and RAG

- **Vector search over image embeddings.** Store the ResNet-50 features (2,048 numbers per
  image) for all training images in a vector index (FAISS or a vector database).
  - "Show similar images": display the nearest training photos and their breeds next to a
    prediction.
  - Non-pet detection: reject uploads that are far from every training image.
- **RAG breed explainer (optional).** After classifying, retrieve facts from a small curated
  knowledge base (temperament, care, health) and have a language model answer questions
  using only those sources. Works as a product feature, but it sits beside the classifier
  rather than improving it.

## Modeling (lower priority)

- Target the largest test-set confusion: American Pit Bull Terrier ↔ Staffordshire Bull
  Terrier (37 mistakes).
- Repeat run 7 to test the hypothesis that brightness/contrast jitter blurs tabby coat
  patterns (errors into Bengal fell from 7 to 2 without it).
- Fill in run 5's breed-level error analysis in the README.
