# OCT A-line Classification

Deep-learning pipeline that classifies oral tissue from OCT B-scan images using single-column intensity profiles (**A-lines**).

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `oct_aline_extraction.py` | Detect the tissue surface in each B-scan and extract fixed-depth A-lines into one `.npz` dataset |
| 2 | `oct_aline_classifier.py` | Train and evaluate 1-D models on the A-lines, with A-line and B-scan level metrics, plots and CSVs |

## Tasks

| Task | Classes |
|------|---------|
| Binary | Non_Cancer, OSCC |
| Multiclass | Normal, CIS, WD_OSCC, PD_OSCC |

## Installation

```bash
pip install numpy pandas scipy opencv-python matplotlib seaborn scikit-learn tqdm torch
pip install umap-learn   # optional, for UMAP feature plots
```

## Data layout

```
Binary_Classification_Data/
    Non_Cancer/*.jpg
    OSCC/*.jpg

Multiclass_Classification_Data/
    Normal/*.jpg
    CIS/*.jpg
    WD_OSCC/*.jpg
    PD_OSCC/*.jpg
```

## Step 1: Extract A-lines

Set `CLASSIFICATION_TASK` (`'binary'` or `'multiclass'`) at the top of the script, then run:

```bash
python oct_aline_extraction.py
```

For each image it detects the tissue surface per column, then takes an A-line of `EXTRACTION_DEPTH_PIXELS` below the surface every `COLUMN_STEP` columns. A-lines are padded to a common length and normalized to [0, 1].

**Key settings:** `COLUMN_STEP` (5), `EXTRACTION_DEPTH_PIXELS` (500), `NORMALIZE` (True), `CROP` (None), `SAVE_VISUALIZATIONS` (True).

**Output:** `5px_extracted_alines_<task>/combined_OCT_dataset.npz`

| Key | Content |
|-----|---------|
| `alines` | float32 array, shape `(N, 500)` |
| `labels` | int32 array, shape `(N,)` |
| `metadata` | per-A-line dict: `image_name`, `class_label`, `class_name`, `column_index`, `padded_length` |

> Keep `metadata`. The classifier uses it for B-scan level splitting and aggregation.

## Step 2: Train and evaluate

Edit the CONFIG block at the top of `oct_aline_classifier.py`, then run:

```bash
python oct_aline_classifier.py
```

**Must match your data:**
- `BINARY`: `True` for Non_Cancer vs OSCC, `False` for the four-class task.
- `DATA_PATH`: path to the `.npz` from Step 1 (for example `./5px_extracted_alines_multiclass/combined_OCT_dataset.npz`).
- `INPUT_SIZE`: must equal the A-line length (500 by default).

**Models** (`cnn_lstm`, `cnn_1d`, `lstm_only`, `cnn_gru`, `inception_1d`, `transformer_1d`):
- `RUN_ALL_MODELS = False` runs only `MODEL_TYPE`.
- `RUN_ALL_MODELS = True` runs every model in `MODEL_TYPES` one after another and writes a comparison summary.
- `EVALUATE_MODEL = '<path>.pt'` skips training and evaluates a saved checkpoint.

**Training setup:** AdamW, linear warmup then cosine LR schedule, class-weighted loss with label smoothing, early stopping on validation loss, and optional train-only A-line augmentation.

**Recommended:** set `GROUP_SPLIT_BY_BSCAN = True`. This keeps all A-lines of a B-scan in one split. With `False`, A-lines from the same B-scan can appear in both train and test, which inflates scores.

## Outputs

Each model writes to `<OUTPUT_DIR>/<model>_<task>_<timestamp>/`:

| Category | Files |
|----------|-------|
| Model and config | `best_model.pt`, `args.json` |
| Metrics | `test_metrics.json/.csv`, `classification_report.csv`, `ascan_predictions.csv`, `training_history.json/.csv` |
| B-scan level | `bscan_metrics.json/.csv`, `bscan_predictions.csv`, `bscan_classification_report.csv` |
| Plots (600 DPI) | training history, confusion matrices (A-line and B-scan), ROC, PR, calibration curve, prediction distribution, per-class metrics, class distribution, A-line vs B-scan comparison, voting agreement, sample A-line waveforms |
| UMAP | true-label, correctness and K-means cluster maps, plus embedding and cluster CSVs |

A multi-model run also creates `ALL_MODELS_<timestamp>/` with `all_models_summary.csv/.json` and `cross_model_comparison.png`.

## Notes

- Seeds are fixed (`SEED = 42`) for reproducible splits and training.
- Class weights are computed from the training split only.
- Validation and test data are never augmented.
- B-scan aggregation: `mean_prob` (average probabilities) or `majority_vote`, set via `BSCAN_AGG_METHOD`.
