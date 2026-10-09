"""
OCT A-line Classifier (CNN / LSTM / GRU / Inception / Transformer)
==================================================================

Trains and evaluates 1-D deep-learning models that classify OCT A-lines
(single image columns) for oral tissue analysis, using the
``combined_OCT_dataset.npz`` file produced by the A-line extraction script.

Tasks
-----
    BINARY = True   ->  Non_Cancer vs OSCC
    BINARY = False  ->  Normal / CIS / WD-OSCC / PD-OSCC

Available architectures (set in MODEL_TYPE / MODEL_TYPES)
---------------------------------------------------------
    cnn_lstm, cnn_1d, lstm_only, cnn_gru, inception_1d, transformer_1d

Main features
-------------
1.  Group-aware (B-scan level) train/val/test splitting, so A-lines from the
    same B-scan never appear in more than one split.
2.  Class-weighted, label-smoothed loss; AdamW; warmup + cosine LR schedule;
    early stopping on validation loss; optional train-only A-line augmentation.
3.  A-line level metrics AND B-scan (image) level metrics, obtained by
    aggregating A-line predictions per B-scan (mean probability or majority vote).
4.  Evaluation plots: confusion matrices, ROC, PR, calibration (with ECE),
    prediction distributions, per-class metrics, class distribution,
    qualitative A-line examples and UMAP of the learned features
    (with K-means clustering).
5.  All numeric results are saved as .json and .csv; all figures are saved
    at FIG_DPI (600) with large, bold fonts.
6.  Optional multi-model run: train every architecture back-to-back, each into
    its own timestamped folder, followed by a cross-model comparison CSV/plot.

Usage
-----
Edit the CONFIG section below, then run:  ``python oct_aline_classifier.py``
"""

import json
import math
import os
import time
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Silence the (version-dependent) PyTorch scheduler "epoch" deprecation warning.
try:
    from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING
    warnings.filterwarnings("ignore", category=UserWarning, message=str(EPOCH_DEPRECATION_WARNING))
except ImportError:  # constant removed in newer PyTorch versions
    pass


# =====================================================================================
#  CONFIG - edit these values directly, then run the script
# =====================================================================================
# The .npz dataset must be the output of the extraction script run on:
#   Binary_Classification_Data/{Non_Cancer, OSCC}                    -> BINARY = True
#   Multiclass_Classification_Data/{Normal, CIS, WD_OSCC, PD_OSCC}   -> BINARY = False

# --- Task, paths and run mode -------------------------------------------------------
BINARY = False                 # True: Non_Cancer vs OSCC | False: Normal/CIS/WD-OSCC/PD-OSCC
DATA_PATH = (
    './extracted_alines_binary/combined_OCT_dataset.npz' if BINARY
    else './edited_extracted_alines_multiclass/combined_OCT_dataset.npz'
)
OUTPUT_DIR = './output_b_all_multiclass'
SEED = 42
EVALUATE_MODEL = None          # path to a saved .pt checkpoint for evaluation only; None = train

# --- Model selection ----------------------------------------------------------------
# Single-model run (RUN_ALL_MODELS = False): only MODEL_TYPE is used.
# Multi-model run  (RUN_ALL_MODELS = True) : every entry of MODEL_TYPES is run back-to-back,
#   each into its own timestamped sub-folder, followed by 'all_models_summary.csv'
#   and 'cross_model_comparison.png'.
MODEL_TYPE = 'cnn_lstm'
RUN_ALL_MODELS = True
MODEL_TYPES = ['cnn_lstm', 'cnn_1d', 'lstm_only', 'cnn_gru', 'inception_1d', 'transformer_1d']
STOP_ON_ERROR = False          # True: a failed model aborts the multi-model run;
                               # False: log the error and continue with the next model

# --- Training hyperparameters -------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 3e-3            # increased regularization to limit train/val divergence
EARLY_STOP_PATIENCE = 7        # epochs without val-loss improvement before stopping
NUM_WORKERS = 4
WARMUP_STEPS = 500             # linear LR warmup steps before cosine annealing

# --- Overfitting / class-imbalance controls -----------------------------------------
USE_CLASS_WEIGHTS = True       # inverse-frequency class weights in the loss
LABEL_SMOOTHING = 0.05         # multiclass only; softens one-hot targets
USE_ALINE_AUGMENTATION = True  # train-only augmentation (val/test are never augmented)
AUG_NOISE_STD = 0.02           # additive Gaussian noise std
AUG_MAX_SHIFT = 10             # max circular depth shift (samples)
AUG_SCALE_RANGE = (0.95, 1.05) # random amplitude scaling range

# --- Model hyperparameters ----------------------------------------------------------
INPUT_SIZE = 500               # A-line length in the .npz (= extraction depth in pixels)
CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV3_FILTERS = 128
LSTM_HIDDEN_SIZE = 96          # also used as the GRU hidden size
LSTM_NUM_LAYERS = 2
FC_SIZE = 384
DROPOUT_RATE = 0.5
NUM_INCEPTION_BLOCKS = 2
TRANSFORMER_DIM = 64           # embedding dimension
TRANSFORMER_NHEAD = 4          # attention heads
TRANSFORMER_LAYERS = 2         # encoder layers

# --- Data splitting -----------------------------------------------------------------
# True  : split by B-scan (image) so A-lines of one B-scan never leak across splits.
#         Recommended - otherwise val/test scores are inflated and B-scan-level
#         evaluation is not trustworthy.
# False : plain stratified split at the A-line level.
GROUP_SPLIT_BY_BSCAN = False
VAL_SIZE = 0.15                # fraction held out for validation
TEST_SIZE = 0.15               # fraction held out for testing

# --- Evaluation / analysis toggles --------------------------------------------------
ENABLE_BSCAN_AGGREGATION = True   # aggregate A-line predictions -> per-B-scan metrics
BSCAN_AGG_METHOD = 'mean_prob'    # 'mean_prob' or 'majority_vote'
ENABLE_UMAP = True                # UMAP of learned features (requires `umap-learn`)
UMAP_MAX_SAMPLES = 4000           # subsample test A-lines for UMAP above this size
UMAP_N_CLUSTERS = None            # None -> number of classes; or set an integer k
ENABLE_CALIBRATION_PLOT = True    # reliability diagram + Expected Calibration Error
ENABLE_SAMPLE_ALINE_PLOTS = True  # grid of correct vs misclassified A-line waveforms
SAMPLE_ALINES_PER_CLASS = 4

# --- Figure export ------------------------------------------------------------------
FIG_DPI = 600                     # resolution of every saved figure

# =====================================================================================


# =====================================================================================
#  REPRODUCIBILITY & PLOT STYLE
# =====================================================================================

def set_seed(seed=42):
    """Seed NumPy and PyTorch (CPU and GPU) and force deterministic cuDNN behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Seed set to {seed} for reproducibility.")


def setup_plot_style():
    """Apply a global Matplotlib style: large, bold fonts and high-resolution export."""
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
        'font.size': 22,
        'axes.titlesize': 30,
        'axes.titleweight': 'bold',
        'axes.labelsize': 25,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 32,
        'lines.linewidth': 2.8,
    })


def _save_figure(fig, output_dir, filename):
    """Tighten layout, save a figure at FIG_DPI into `output_dir`, and close it."""
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


# =====================================================================================
#  DATASET
# =====================================================================================

class OCTAlineDataset(Dataset):
    """PyTorch Dataset of OCT A-lines with optional training-time augmentation.

    Parameters
    ----------
    alines : np.ndarray
        Array of shape (N, length) with the A-line intensities.
    labels : np.ndarray
        Integer class label per A-line.
    image_ids : array-like, optional
        Source B-scan identifier per A-line. It is kept (and filtered consistently
        with the binary mask) so predictions can later be grouped per B-scan.
        If omitted, every A-line gets its own unique ID.
    binary : bool
        If True, keep only classes 0 and 1.
    augment : bool
        If True, apply light random augmentation each time a sample is fetched
        (a fresh perturbation every epoch). Use for the TRAIN set only; val/test
        must stay deterministic.
    aug_noise_std, aug_max_shift, aug_scale_range
        Augmentation strengths (see ``_augment_aline``).
    """

    def __init__(self, alines, labels, image_ids=None, binary=False, augment=False,
                 aug_noise_std=0.02, aug_max_shift=10, aug_scale_range=(0.95, 1.05)):
        self.alines = torch.tensor(alines, dtype=torch.float32)
        self.binary = binary
        self.augment = augment
        self.aug_noise_std = aug_noise_std
        self.aug_max_shift = aug_max_shift
        self.aug_scale_range = aug_scale_range

        if image_ids is None:
            image_ids = np.array([f"aline_{i}" for i in range(len(labels))], dtype=object)
        else:
            image_ids = np.asarray(image_ids, dtype=object)

        if binary:
            mask = (labels == 0) | (labels == 1)
            self.alines = self.alines[mask]
            self.labels = torch.tensor(labels[mask], dtype=torch.long)
            self.image_ids = image_ids[mask]
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.image_ids = image_ids

    def __len__(self):
        return len(self.alines)

    def _augment_aline(self, aline):
        """Apply light, label-preserving perturbations to one A-line:
        additive Gaussian noise, a small random circular depth shift, and a
        small random amplitude scaling. They mimic scan-to-scan variation
        without changing tissue identity.
        """
        if self.aug_noise_std > 0:
            aline = aline + torch.randn_like(aline) * self.aug_noise_std
        if self.aug_max_shift > 0:
            shift = int(torch.randint(-self.aug_max_shift, self.aug_max_shift + 1, (1,)).item())
            if shift != 0:
                aline = torch.roll(aline, shifts=shift, dims=0)
        if self.aug_scale_range and self.aug_scale_range != (1.0, 1.0):
            lo, hi = self.aug_scale_range
            aline = aline * (lo + (hi - lo) * torch.rand(1).item())
        return aline

    def __getitem__(self, idx):
        aline = self.alines[idx]
        if self.augment:
            aline = self._augment_aline(aline)
        return aline.unsqueeze(0), self.labels[idx]  # add channel dimension -> (1, length)


# =====================================================================================
#  MODEL ARCHITECTURES
#  Every model takes input of shape (batch, 1, length) and returns raw scores:
#  (batch, num_classes) for multiclass, or (batch, 1) logits for binary.
# =====================================================================================

class CNN_1D(nn.Module):
    """Light pure 1-D CNN baseline.

    Two Conv-BN-ReLU-MaxPool blocks extract hierarchical local features
    (tissue layers / textures), followed by global average pooling and a
    single linear classifier.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.binary = binary
        self.output_size = 1 if binary else num_classes

        c1, c2 = 16, 32  # few filters -> lightweight model

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Classifier head: global average pooling -> linear layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(c2, self.output_size)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.avgpool(x)       # [batch, 32, 1]
        x = self.flatten(x)       # [batch, 32]
        return self.classifier(x)


class LSTM_Only(nn.Module):
    """Pure bidirectional LSTM that treats the A-line as a 1-feature sequence.

    Tests whether the sequential structure alone, without CNN feature extraction,
    carries enough information to classify the tissue.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.binary = binary
        self.output_size = 1 if binary else num_classes

        hidden_size = kwargs.get('lstm_hidden_size', 128)
        num_layers = kwargs.get('lstm_num_layers', 2)
        dropout = kwargs.get('dropout_rate', 0.5)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, self.output_size)  # x2 for bidirectional
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)                      # (batch, 1, len) -> (batch, len, 1)
        _, (h_n, _) = self.lstm(x)
        # Concatenate the last layer's final forward and backward hidden states
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.classifier(hidden)


class CNN_GRU(nn.Module):
    """Hybrid 1-D CNN + bidirectional GRU.

    Three conv blocks extract local features; the GRU models how those features
    evolve with depth. GRUs are lighter than LSTMs with similar performance.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.binary = binary
        self.output_size = 1 if binary else num_classes

        c1 = kwargs.get('conv1_filters', 32)
        c2 = kwargs.get('conv2_filters', 64)
        c3 = kwargs.get('conv3_filters', 96)
        gru_hidden = kwargs.get('lstm_hidden_size', 96)  # shares the LSTM hidden-size setting
        fc_size = kwargs.get('fc_size', 384)
        dropout = kwargs.get('dropout_rate', 0.6)

        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(input_size=c3, hidden_size=gru_hidden,
                          batch_first=True, bidirectional=True)

        # After 3 poolings the length is input_size // 8; the GRU output is 2 * hidden
        flattened_size = gru_hidden * 2 * (input_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, self.output_size),
        )

    def forward(self, x):
        x = self.cnn_feature_extractor(x)   # [batch, channels, len/8]
        x = x.permute(0, 2, 1)              # [batch, len/8, channels]
        x, _ = self.gru(x)
        return self.classifier(x)


class InceptionBlock1D(nn.Module):
    """1-D Inception block: four parallel branches (1x1, 3x3, 5x5, pooling),
    each producing `out_channels_per_branch` channels, concatenated along channels."""

    def __init__(self, in_channels, out_channels_per_branch):
        super().__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(out_channels_per_branch, out_channels_per_branch, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(out_channels_per_branch, out_channels_per_branch, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1x1(x), self.branch3x3(x),
                          self.branch5x5(x), self.branch_pool(x)], dim=1)


class InceptionNet_1D(nn.Module):
    """Inception-style 1-D network.

    Parallel branches with different kernel sizes capture fine textures and
    broader structures at the same time.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.binary = binary
        self.output_size = 1 if binary else num_classes

        num_blocks = kwargs.get('num_inception_blocks', 2)
        dropout = kwargs.get('dropout_rate', 0.5)

        self.pre_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = 64
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(InceptionBlock1D(in_channels, 32))  # 4 branches x 32 = 128 channels
            in_channels = 128

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, self.output_size),
        )

    def forward(self, x):
        x = self.pre_layers(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs of shape (seq_len, batch, d_model)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer_1D(nn.Module):
    """Transformer encoder for 1-D A-line classification.

    A strided Conv1d "patch embedding" shortens the sequence (500 -> 99 tokens),
    positional encoding is added, and self-attention layers model global
    relationships along the depth axis. The first token's output is classified.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.binary = binary
        self.output_size = 1 if binary else num_classes

        d_model = kwargs.get('transformer_dim', 64)
        nhead = kwargs.get('transformer_nhead', 4)
        num_layers = kwargs.get('transformer_layers', 2)
        dropout = kwargs.get('dropout_rate', 0.2)

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=10, stride=5)  # 500 -> 99 tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(nn.Linear(d_model, self.output_size))

    def forward(self, x):
        x = self.patch_embed(x)        # (batch, d_model, seq_len)
        x = x.permute(0, 2, 1)         # (batch, seq_len, d_model)

        # PositionalEncoding expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)
        x = x[:, 0, :]                 # first token acts like a [CLS] token
        return self.classifier(x)


class CNN_LSTM(nn.Module):
    """1-D CNN + bidirectional LSTM with a residual connection after the first block.

    Three conv blocks extract local features; a 1x1-conv "residual adapter" adds a
    skip connection around the first conv block; the LSTM models depth-wise
    dependencies; and a two-layer fully connected head classifies the result.
    """

    def __init__(self, num_classes, input_size=500, binary=False, **kwargs):
        super().__init__()
        self.output_size = 1 if binary else num_classes
        self.binary = binary

        c1 = kwargs.get('conv1_filters', 32)
        c2 = kwargs.get('conv2_filters', 48)
        c3 = kwargs.get('conv3_filters', 64)
        lstm_hidden = kwargs.get('lstm_hidden_size', 48)
        fc_size = kwargs.get('fc_size', 384)
        dropout = kwargs.get('dropout_rate', 0.6)

        # CNN layers
        self.conv1 = nn.Conv1d(1, c1, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Residual adapter: projects the raw input to c1 channels for the skip connection
        self.res_adapter = nn.Conv1d(in_channels=1, out_channels=c1, kernel_size=1)

        # Bidirectional LSTM over the CNN feature sequence
        self.lstm = nn.LSTM(input_size=c3, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=True)

        # Classifier head. Length after 3 poolings = input_size // 8; BiLSTM output = 2 * hidden.
        self.dropout = nn.Dropout(dropout)
        flattened_size = (input_size // 8) * lstm_hidden * 2
        self.fc1 = nn.Linear(flattened_size, fc_size)
        self.fc2 = nn.Linear(fc_size, self.output_size)

    def forward(self, x):
        # Residual path: project input and resample to the post-pool1 length
        res = self.res_adapter(x)
        res = F.interpolate(res, size=x.size(2) // 2, mode='linear', align_corners=False)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x + res                       # add residual connection

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.permute(0, 2, 1)            # [batch, seq_len, channels]
        x, _ = self.lstm(x)

        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =====================================================================================
#  DATA LOADING & SPLITTING
# =====================================================================================

def load_data(npz_path, binary=False, group_split=True, val_size=0.15, test_size=0.15, seed=42):
    """Load the extraction-script .npz file and split it into train/val/test.

    Each A-line's parent B-scan ID is recovered from the saved ``metadata`` array.
    It is used (a) for group-aware splitting, so A-lines of one B-scan never leak
    across splits, and (b) to aggregate A-line predictions per B-scan later.
    If metadata is missing, every A-line gets its own ID and these two features
    are effectively disabled.

    Parameters
    ----------
    npz_path : str
        Path to ``combined_OCT_dataset.npz``.
    binary : bool
        If True, keep only classes 0 and 1.
    group_split : bool
        True: split by B-scan (recommended). False: stratified A-line-level split.
    val_size, test_size : float
        Fractions of the data held out for validation and testing.
    seed : int
        Random seed for the split.

    Returns
    -------
    tuple
        (train_alines, val_alines, test_alines,
         train_labels, val_labels, test_labels,
         train_ids, val_ids, test_ids)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"FATAL: Dataset file not found: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=True)
        alines = data['alines']
        labels = data['labels']
        metadata = data['metadata'] if 'metadata' in data.files else None
    except Exception as e:
        raise IOError(f"FATAL: Error loading data from {npz_path}. Ensure it is a valid .npz file "
                      f"with 'alines' and 'labels' keys. Details: {e}")

    print(f"Original dataset: {alines.shape} A-lines, {len(np.unique(labels))} classes")

    # Per-A-line B-scan ID. The class label is prefixed so that identical filenames
    # in different class folders can never collide.
    if metadata is not None:
        try:
            image_ids = np.array([f"{m['class_label']}__{m['image_name']}" for m in metadata],
                                 dtype=object)
        except Exception:
            print("WARNING: Could not parse 'image_name'/'class_label' from metadata; falling back "
                  "to per-A-line IDs (B-scan aggregation and group-aware splitting disabled).")
            image_ids = np.array([f"aline_{i}" for i in range(len(labels))], dtype=object)
    else:
        print("WARNING: No 'metadata' array in the .npz file; falling back to per-A-line IDs "
              "(B-scan aggregation and group-aware splitting disabled). Re-run the extraction "
              "script to regenerate metadata if you need these features.")
        image_ids = np.array([f"aline_{i}" for i in range(len(labels))], dtype=object)

    if binary:
        mask = np.isin(labels, [0, 1])
        alines, labels, image_ids = alines[mask], labels[mask], image_ids[mask]
        print(f"Binary mode: Filtered to {alines.shape[0]} A-lines, 2 classes")
        if len(np.unique(labels)) < 2:
            raise ValueError(f"FATAL: Binary mode requires classes 0 and 1, "
                             f"but found only: {np.unique(labels)}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Class distribution in loaded data:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({count / len(labels) * 100:.2f}%)")

    n_unique_images = len(np.unique(image_ids))
    print(f"A-lines come from {n_unique_images} unique B-scans.")

    if group_split and n_unique_images >= 3:
        # Outer split: train vs (val + test); inner split: val vs test. Both by B-scan.
        holdout_frac = val_size + test_size
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
        train_idx, holdout_idx = next(gss_outer.split(alines, labels, groups=image_ids))

        gss_inner = GroupShuffleSplit(n_splits=1, test_size=test_size / holdout_frac,
                                      random_state=seed)
        val_rel_idx, test_rel_idx = next(gss_inner.split(
            alines[holdout_idx], labels[holdout_idx], groups=image_ids[holdout_idx]))
        val_idx, test_idx = holdout_idx[val_rel_idx], holdout_idx[test_rel_idx]
        print("Split performed at the B-scan (group) level: no A-lines from the same B-scan "
              "appear in more than one split.")
    else:
        if group_split:
            print("WARNING: Not enough distinct B-scans for group splitting; falling back to a "
                  "stratified A-line-level split (B-scan-level evaluation may be biased).")
        all_idx = np.arange(len(labels))
        train_idx, holdout_idx = train_test_split(
            all_idx, test_size=val_size + test_size, stratify=labels, random_state=seed)
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=test_size / (val_size + test_size),
            stratify=labels[holdout_idx], random_state=seed)

    train_alines, val_alines, test_alines = alines[train_idx], alines[val_idx], alines[test_idx]
    train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]
    train_ids, val_ids, test_ids = image_ids[train_idx], image_ids[val_idx], image_ids[test_idx]

    print(f"Data split: Train={train_alines.shape}, Validation={val_alines.shape}, "
          f"Test={test_alines.shape}")
    print(f"  Unique B-scans -> Train={len(np.unique(train_ids))}, "
          f"Val={len(np.unique(val_ids))}, Test={len(np.unique(test_ids))}")

    return (train_alines, val_alines, test_alines,
            train_labels, val_labels, test_labels,
            train_ids, val_ids, test_ids)


def compute_class_weights(labels, num_classes, binary=False):
    """Compute inverse-frequency class weights to counter class imbalance.

    Returns
    -------
    torch.Tensor
        Binary: scalar ``pos_weight`` (= #negatives / #positives) for BCEWithLogitsLoss.
        Multiclass: per-class weights normalized to mean 1.0, for CrossEntropyLoss(weight=...).
    """
    labels = np.asarray(labels)
    counts = np.array([max(1, np.sum(labels == c)) for c in range(num_classes)], dtype=np.float64)
    if binary:
        return torch.tensor(counts[0] / counts[1], dtype=torch.float32)
    inv_freq = counts.sum() / (num_classes * counts)
    return torch.tensor(inv_freq / inv_freq.mean(), dtype=torch.float32)


# =====================================================================================
#  PLOTTING FUNCTIONS
# =====================================================================================

def plot_history(history, output_dir):
    """Plot train vs validation Accuracy, Loss, Precision, Recall and F1 per epoch."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    metrics = [('Accuracy', 'acc'), ('Loss', 'loss'), ('Precision', 'precision'),
               ('Recall', 'recall'), ('F1 Score', 'f1')]

    for i, (title, key) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.plot(history[f'train_{key}'], label='Train', linewidth=3.0, marker='o', markersize=4)
        ax.plot(history[f'val_{key}'], label='Validation', linewidth=3.0, marker='s', markersize=4)
        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=21)
        ax.set_ylabel(title, fontsize=21)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=18)

    axes[2, 1].axis('off')  # 5 plots on a 3x2 grid -> hide the unused panel
    _save_figure(fig, output_dir, 'training_history.png')


def plot_confusion_matrix(y_true, y_pred, output_dir, class_names, filename_prefix='', title_suffix=''):
    """Save a row-normalized confusion matrix (percent of each true class).

    `filename_prefix` lets A-line level ('') and B-scan level ('bscan_') versions coexist.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 22, 'fontweight': 'bold'})
    plt.xlabel('Predicted', fontsize=22, fontweight='bold')
    plt.ylabel('True', fontsize=22, fontweight='bold')
    plt.title(f'Normalized Confusion Matrix{title_suffix}', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    _save_figure(fig, output_dir, f'{filename_prefix}confusion_matrix_normalized.png')


def plot_confusion_matrix_counts(y_true, y_pred, output_dir, class_names, filename_prefix='', title_suffix=''):
    """Save a confusion matrix showing raw counts."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 22, 'fontweight': 'bold'}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=22, fontweight='bold')
    ax.set_ylabel('True', fontsize=22, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Counts){title_suffix}', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    _save_figure(fig, output_dir, f'{filename_prefix}confusion_matrix_counts.png')


def plot_roc_curve(y_true, y_scores, output_dir, binary=False, class_names=None):
    """Save the ROC curve (binary: single curve; multiclass: one-vs-rest per class)."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    if binary:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, color=colors[0], lw=3.5, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(y_scores.shape[1])))
        for i in range(y_scores.shape[1]):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            label = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=3.5,
                    label=f'{label} (AUC = {auc(fpr, tpr):.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=3)  # chance line
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate', fontsize=24, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=24, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=27, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc="lower right", fontsize=20)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
    _save_figure(fig, output_dir, 'roc_curve.png')

    # This plot uses a serif font; restore the global style for subsequent figures.
    plt.rcdefaults()
    setup_plot_style()


def plot_pr_curve(y_true, y_scores, output_dir, binary=False, class_names=None):
    """Save the Precision-Recall curve (binary, or one-vs-rest per class)."""
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    if binary:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        ax.plot(recall, precision, color=colors[0], lw=3.5, label=f'PR curve (AP = {ap:.3f})')
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(y_scores.shape[1])))
        for i in range(y_scores.shape[1]):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
            label = class_names[i] if class_names else f'Class {i}'
            ax.plot(recall, precision, color=colors[i % len(colors)], lw=3.5,
                    label=f'{label} (AP = {ap:.3f})')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall', fontsize=24, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=24, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=27, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(loc="best", fontsize=20)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
    _save_figure(fig, output_dir, 'pr_curve.png')


def plot_prediction_distribution(y_scores, y_true, output_dir, binary=False):
    """Save histograms of the model's prediction scores for each true class.

    Binary: score of the positive class, split by true class.
    Multiclass: probability assigned to the true class, per class.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if binary:
        ax.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label='Class 0 (Negative)',
                color='#1f77b4', edgecolor='black')
        ax.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label='Class 1 (Positive)',
                color='#ff7f0e', edgecolor='black')
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i in range(len(np.unique(y_true))):
            ax.hist(y_scores[y_true == i, i], bins=40, alpha=0.6, label=f'Class {i}',
                    color=colors[i % len(colors)], edgecolor='black')

    ax.set_xlabel('Prediction Score', fontsize=24, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
    ax.set_title('Distribution of Prediction Scores by Class', fontsize=27, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
    _save_figure(fig, output_dir, 'prediction_distribution.png')


def plot_class_metrics(y_true, y_pred, output_dir, class_names):
    """Save a grouped bar chart of per-class Precision, Recall and F1."""
    classes = np.unique(y_true)
    precision_scores = [precision_score(y_true == c, y_pred == c) for c in classes]
    recall_scores = [recall_score(y_true == c, y_pred == c) for c in classes]
    f1_scores = [f1_score(y_true == c, y_pred == c) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, precision_scores, width, label='Precision', color='#1f77b4', edgecolor='black')
    ax.bar(x, recall_scores, width, label='Recall', color='#ff7f0e', edgecolor='black')
    ax.bar(x + width, f1_scores, width, label='F1 Score', color='#2ca02c', edgecolor='black')

    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=27, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5, axis='y')

    # Value labels above each bar
    for offset, values in ((-width, precision_scores), (0, recall_scores), (width, f1_scores)):
        for i, v in enumerate(values):
            ax.text(i + offset, v + 0.02, f'{v:.2f}', ha='center', fontsize=16, fontweight='bold')

    _save_figure(fig, output_dir, 'class_metrics.png')


def plot_class_distribution(y_train, y_val, y_test, output_dir, class_names, binary=False):
    """Save bar charts of the class counts in the train, validation and test sets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    datasets = [('Training Set', y_train), ('Validation Set', y_val), ('Test Set', y_test)]

    for ax, (title, y_data) in zip(axes, datasets):
        unique, counts = np.unique(y_data, return_counts=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(unique)]
        bars = ax.bar(range(len(unique)), counts, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels([class_names[i] for i in unique], fontsize=18)
        ax.set_ylabel('Count', fontsize=20, fontweight='bold')
        ax.set_title(title, fontsize=21, fontweight='bold')
        ax.tick_params(axis='y', labelsize=17)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                    ha='center', va='bottom', fontsize=17, fontweight='bold')

    _save_figure(fig, output_dir, 'class_distribution.png')


def plot_calibration_curve(y_true, y_scores, output_dir, binary=False, n_bins=10):
    """Save a reliability diagram and compute the Expected Calibration Error (ECE).

    Within each confidence bin, the diagram compares the model's average confidence
    with its observed accuracy. Bin statistics and ECE are also saved to disk.

    Returns
    -------
    float
        Expected Calibration Error (lower is better calibrated).
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if binary:
        confidences = np.where(y_scores > 0.5, y_scores, 1 - y_scores)
        predictions = (y_scores > 0.5).astype(int)
    else:
        confidences = y_scores.max(axis=1)
        predictions = y_scores.argmax(axis=1)
    correctness = (predictions == y_true).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.sum() > 0:
            bin_acc.append(correctness[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Perfect calibration')
    valid = ~np.isnan(bin_conf)
    ax.plot(np.array(bin_conf)[valid], np.array(bin_acc)[valid], marker='o', markersize=10,
            lw=3.5, color='#1f77b4', label='Model')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Confidence', fontsize=24, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=24, fontweight='bold')
    ax.set_title('Reliability Diagram (Calibration)', fontsize=27, fontweight='bold')
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    _save_figure(fig, output_dir, 'calibration_curve.png')

    # ECE = sum over bins of (bin weight) * |accuracy - confidence|
    counts = np.array(bin_counts)
    ece = float(np.nansum(counts[valid] * np.abs(np.array(bin_acc)[valid] - np.array(bin_conf)[valid]))
                / len(confidences))
    pd.DataFrame({'bin_confidence': bin_conf, 'bin_accuracy': bin_acc, 'bin_count': bin_counts}).to_csv(
        os.path.join(output_dir, 'calibration_bins.csv'), index=False)
    with open(os.path.join(output_dir, 'calibration_ece.json'), 'w') as f:
        json.dump({'expected_calibration_error': ece}, f, indent=4)
    return ece


def plot_sample_alines_grid(alines_data, y_true, y_pred, output_dir, class_names, n_per_class=4):
    """Qualitative check: for each class, plot a few correctly classified and a few
    misclassified A-line waveforms side by side."""
    alines_data = np.array(alines_data)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)

    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    rng = np.random.RandomState(42)
    for c in range(n_classes):
        correct_idx = np.where((y_true == c) & (y_pred == c))[0]
        wrong_idx = np.where((y_true == c) & (y_pred != c))[0]
        rng.shuffle(correct_idx)
        rng.shuffle(wrong_idx)

        for col, (idx_list, label, color) in enumerate(
                [(correct_idx, 'Correctly Classified', None), (wrong_idx, 'Misclassified', 'crimson')]):
            ax = axes[c, col]
            for idx in idx_list[:n_per_class]:
                kwargs = {'color': color} if color else {}
                ax.plot(alines_data[idx], alpha=0.8, linewidth=2.0, **kwargs)
            ax.set_title(f'{class_names[c]} \u2014 {label} (n={len(idx_list)})',
                         fontsize=21, fontweight='bold')
            ax.set_xlabel('Depth (pixels)', fontsize=18)
            ax.set_ylabel('Normalized Intensity', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, alpha=0.3)

    _save_figure(fig, output_dir, 'sample_alines_correct_vs_misclassified.png')


def plot_cross_model_comparison(results_df, output_dir):
    """Bar chart comparing test Accuracy/Precision/Recall/F1 across all models of a
    multi-model run."""
    metrics_cols = [c for c in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
                    if c in results_df.columns]
    if not metrics_cols or results_df.empty:
        return
    labels_pretty = {'test_accuracy': 'Accuracy', 'test_precision': 'Precision',
                     'test_recall': 'Recall', 'test_f1': 'F1 Score'}

    models = results_df['model_type'].tolist()
    x = np.arange(len(models))
    width = 0.8 / len(metrics_cols)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(models)), 8))
    for i, col in enumerate(metrics_cols):
        bars = ax.bar(x + (i - (len(metrics_cols) - 1) / 2) * width, results_df[col].values, width,
                      label=labels_pretty.get(col, col), color=colors[i % len(colors)],
                      edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f'{h:.3f}',
                        ha='center', fontsize=15, fontweight='bold', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=21, rotation=20, ha='right')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.set_title('Cross-Model Test Performance Comparison', fontsize=27, fontweight='bold')
    ax.legend(fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    _save_figure(fig, output_dir, 'cross_model_comparison.png')


# =====================================================================================
#  B-SCAN (IMAGE) LEVEL AGGREGATION
# =====================================================================================

def aggregate_to_bscan(image_ids, y_true, y_pred, y_scores, binary=False, method='mean_prob'):
    """Aggregate A-line predictions into one prediction per B-scan (source image).

    Parameters
    ----------
    image_ids : array-like
        B-scan ID of every A-line.
    y_true, y_pred : array-like
        True and predicted class of every A-line.
    y_scores : array-like
        A-line probabilities: (N,) for binary, (N, n_classes) for multiclass.
    method : {'mean_prob', 'majority_vote'}
        'mean_prob'    - average the A-line probabilities, then argmax / threshold at 0.5.
        'majority_vote'- most common hard A-line prediction in the B-scan.

    Returns
    -------
    tuple
        (unique_images, bscan_true, bscan_pred, bscan_scores,
         bscan_vote_fraction, bscan_n_alines)
        where vote_fraction is the share of a B-scan's A-lines that agree with
        its final B-scan prediction.
    """
    image_ids = np.array(image_ids)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    unique_images = np.unique(image_ids)
    bscan_true, bscan_pred, bscan_scores, bscan_vote_fraction, bscan_n_alines = [], [], [], [], []

    for img in unique_images:
        mask = image_ids == img
        true_label = int(np.bincount(y_true[mask].astype(int)).argmax())

        if method == 'majority_vote':
            pred_label = int(np.bincount(y_pred[mask].astype(int)).argmax())
            agg_score = float(y_scores[mask].mean()) if binary else y_scores[mask].mean(axis=0)
        else:  # 'mean_prob'
            if binary:
                agg_score = float(y_scores[mask].mean())
                pred_label = int(agg_score > 0.5)
            else:
                agg_score = y_scores[mask].mean(axis=0)
                pred_label = int(np.argmax(agg_score))

        bscan_true.append(true_label)
        bscan_pred.append(pred_label)
        bscan_scores.append(agg_score)
        bscan_vote_fraction.append(float(np.mean(y_pred[mask] == pred_label)))
        bscan_n_alines.append(int(mask.sum()))

    return (unique_images, np.array(bscan_true), np.array(bscan_pred), np.array(bscan_scores),
            np.array(bscan_vote_fraction), np.array(bscan_n_alines))


def plot_bscan_vote_confidence(vote_fraction, output_dir):
    """Histogram of the per-B-scan fraction of A-lines agreeing with the B-scan prediction
    (a quick read on how internally consistent each B-scan is)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.hist(vote_fraction, bins=20, color='#9467bd', edgecolor='black', alpha=0.85)
    ax.axvline(np.mean(vote_fraction), color='crimson', linestyle='--', linewidth=3,
               label=f'Mean agreement = {np.mean(vote_fraction):.2%}')
    ax.set_xlabel('Fraction of A-lines agreeing with the B-scan prediction', fontsize=23, fontweight='bold')
    ax.set_ylabel('Number of B-scans', fontsize=24, fontweight='bold')
    ax.set_title('A-line \u2192 B-scan Voting Agreement', fontsize=27, fontweight='bold')
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    _save_figure(fig, output_dir, 'bscan_vote_agreement.png')


def plot_ascan_vs_bscan_comparison(ascan_metrics, bscan_metrics, output_dir):
    """Bar chart comparing A-line level vs B-scan level Accuracy/Precision/Recall/F1."""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    ascan_vals = [ascan_metrics.get(k, np.nan)
                  for k in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']]
    bscan_vals = [bscan_metrics.get(k, np.nan)
                  for k in ['bscan_accuracy', 'bscan_precision', 'bscan_recall', 'bscan_f1']]

    x = np.arange(len(metrics_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, ascan_vals, width, label='A-line level', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width / 2, bscan_vals, width, label='B-scan level', color='#2ca02c', edgecolor='black')

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f'{h:.3f}',
                        ha='center', fontsize=16, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=21)
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_title('A-line-level vs B-scan-level Performance', fontsize=27, fontweight='bold')
    ax.legend(fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    _save_figure(fig, output_dir, 'ascan_vs_bscan_comparison.png')


# =====================================================================================
#  CSV EXPORT HELPERS
# =====================================================================================

def save_history_csv(history, output_dir):
    """Save the per-epoch training history to training_history.csv."""
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index_label='epoch')


def save_classification_report_csv(y_true, y_pred, class_names, output_dir,
                                   filename='classification_report.csv'):
    """Save sklearn's per-class precision/recall/F1/support report as CSV."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, filename))


def save_predictions_csv(image_ids, y_true, y_pred, y_scores, class_names, output_dir,
                         filename='ascan_predictions.csv'):
    """Save every A-line's source image, true/predicted class and probabilities."""
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)
    df = pd.DataFrame({
        'image_name': np.array(image_ids),
        'true_label': y_true, 'true_class': [class_names[int(t)] for t in y_true],
        'pred_label': y_pred, 'pred_class': [class_names[int(p)] for p in y_pred],
    })
    if y_scores.ndim == 1:
        df['pred_score'] = y_scores
    else:
        for i, cname in enumerate(class_names):
            df[f'prob_{cname}'] = y_scores[:, i]
    df.to_csv(os.path.join(output_dir, filename), index=False)


def compute_and_save_bscan_metrics(image_ids, y_true, y_pred, y_scores, binary, class_names,
                                   output_dir, method='mean_prob'):
    """Aggregate to B-scan level, compute metrics, and save JSON/CSV files and plots.

    Returns
    -------
    tuple
        (metrics_dict, image_names, bscan_true, bscan_pred, bscan_scores, vote_fraction)
    """
    imgs, b_true, b_pred, b_scores, vote_frac, n_alines = aggregate_to_bscan(
        image_ids, y_true, y_pred, y_scores, binary=binary, method=method)

    avg = 'binary' if binary else 'weighted'
    metrics = {
        'bscan_accuracy': accuracy_score(b_true, b_pred),
        'bscan_precision': precision_score(b_true, b_pred, average=avg, zero_division=0),
        'bscan_recall': recall_score(b_true, b_pred, average=avg, zero_division=0),
        'bscan_f1': f1_score(b_true, b_pred, average=avg, zero_division=0),
        'num_bscans': int(len(imgs)),
        'aggregation_method': method,
    }

    with open(os.path.join(output_dir, 'bscan_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'bscan_metrics.csv'), index=False)

    pred_df = pd.DataFrame({
        'image_name': imgs,
        'true_label': b_true, 'true_class': [class_names[t] for t in b_true],
        'pred_label': b_pred, 'pred_class': [class_names[p] for p in b_pred],
        'vote_fraction': vote_frac, 'num_alines': n_alines,
    })
    if binary:
        pred_df['mean_prob'] = b_scores
    else:
        b_scores_arr = np.array(list(b_scores))
        for i, cname in enumerate(class_names):
            pred_df[f'mean_prob_{cname}'] = b_scores_arr[:, i]
    pred_df.to_csv(os.path.join(output_dir, 'bscan_predictions.csv'), index=False)

    save_classification_report_csv(b_true, b_pred, class_names, output_dir,
                                   filename='bscan_classification_report.csv')
    plot_confusion_matrix(b_true, b_pred, output_dir, class_names=class_names,
                          filename_prefix='bscan_', title_suffix=' (B-scan level)')
    plot_confusion_matrix_counts(b_true, b_pred, output_dir, class_names=class_names,
                                 filename_prefix='bscan_', title_suffix=' (B-scan level)')
    plot_bscan_vote_confidence(vote_frac, output_dir)

    return metrics, imgs, b_true, b_pred, b_scores, vote_frac


# =====================================================================================
#  UMAP FEATURE VISUALIZATION
# =====================================================================================

def extract_features_for_umap(model, dataloader, device):
    """Collect penultimate-layer feature vectors for every sample in `dataloader`.

    Architecture-agnostic: a forward hook on the model's LAST ``nn.Linear`` layer
    (the final class-score layer in every architecture here) captures that layer's
    input, i.e. the learned feature embedding.

    Returns
    -------
    tuple of np.ndarray
        (features of shape (N, D), labels of shape (N,))
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No Linear layer found in model; cannot extract features for UMAP.")
    final_linear = linear_layers[-1]

    captured = {}

    def hook(module, inputs, output):
        captured['feat'] = inputs[0].detach().cpu().numpy()

    handle = final_linear.register_forward_hook(hook)
    model.eval()
    all_feats, all_labels = [], []
    try:
        with torch.no_grad():
            for inputs, targets in dataloader:
                _ = model(inputs.to(device))
                all_feats.append(captured['feat'])
                all_labels.append(targets.numpy())
    finally:
        handle.remove()

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


def plot_umap_features(features, labels, output_dir, class_names, preds=None,
                       n_clusters=None, random_state=42):
    """Project learned features to 2-D with UMAP and save visualizations and tables.

    Outputs
    -------
    - umap_features_true_label.png   : colored by true class
    - umap_features_correctness.png  : correct vs misclassified (if `preds` is given)
    - umap_cluster_map.png           : colored by unsupervised K-means cluster
    - umap_embedding.npy / .csv      : the 2-D embedding (+ labels, clusters)
    - umap_cluster_vs_true_class.csv : contingency table of true class vs cluster
    - umap_cluster_metrics.json      : sample/feature counts and silhouette score

    K-means is run on the ORIGINAL feature vectors (not the 2-D embedding) and never
    sees the true labels, so it shows whether the feature space separates naturally.
    Requires the optional `umap-learn` package.
    """
    try:
        import umap
    except ImportError:
        print("  ✗ umap-learn not installed. Install it with: pip install umap-learn")
        return

    features = np.asarray(features)
    labels = np.asarray(labels).astype(int)

    if features.ndim != 2:
        raise ValueError(f"UMAP expects a 2D feature matrix, got shape {features.shape}")
    if len(features) < 3:
        raise ValueError("At least 3 samples are required for UMAP visualization.")

    # UMAP cannot use more neighbors than there are samples.
    n_neighbors = min(15, max(2, len(features) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
                        random_state=random_state)
    embedding = reducer.fit_transform(features)
    np.save(os.path.join(output_dir, 'umap_embedding.npy'), embedding)

    df_umap = pd.DataFrame({
        'umap_1': embedding[:, 0], 'umap_2': embedding[:, 1],
        'true_label': labels, 'true_class': [class_names[int(l)] for l in labels],
    })
    if preds is not None:
        preds = np.asarray(preds).astype(int)
        if len(preds) != len(labels):
            raise ValueError("preds and labels must have the same length.")
        df_umap['pred_label'] = preds
        df_umap['pred_class'] = [class_names[int(p)] for p in preds]

    # --- Unsupervised K-means clustering of the learned features --------------------
    if n_clusters is None:
        n_clusters = len(class_names)  # default: one cluster per class
    n_clusters = int(n_clusters)
    if n_clusters < 2 or n_clusters >= len(features):
        raise ValueError(f"n_clusters must be between 2 and n_samples-1; "
                         f"got n_clusters={n_clusters}, n_samples={len(features)}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    cluster_labels = kmeans.fit_predict(features)
    sil = float(silhouette_score(features, cluster_labels))  # cluster separation (-1 to 1)

    df_umap['cluster'] = cluster_labels
    df_umap.to_csv(os.path.join(output_dir, 'umap_embedding.csv'), index=False)

    cluster_summary = (
        pd.crosstab(pd.Series(labels, name='true_label'), pd.Series(cluster_labels, name='cluster'))
        .reindex(index=range(len(class_names)), fill_value=0)
        .reindex(columns=range(n_clusters), fill_value=0)
    )
    cluster_summary.to_csv(os.path.join(output_dir, 'umap_cluster_vs_true_class.csv'))

    with open(os.path.join(output_dir, 'umap_cluster_metrics.json'), 'w') as f:
        json.dump({'n_samples': int(len(features)), 'n_features': int(features.shape[1]),
                   'n_clusters': n_clusters, 'silhouette_score': sil}, f, indent=4)

    # --- Plot 1: colored by true class ---------------------------------------------
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, cname in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=24, alpha=0.70,
                   color=colors[i % len(colors)], label=cname, edgecolors='none')
    ax.set_xlabel('UMAP-1', fontsize=24, fontweight='bold')
    ax.set_ylabel('UMAP-2', fontsize=24, fontweight='bold')
    ax.set_title('UMAP Projection of Learned Features (True Class)', fontsize=27, fontweight='bold')
    ax.legend(fontsize=20, markerscale=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    _save_figure(fig, output_dir, 'umap_features_true_label.png')

    # --- Plot 2: correct vs misclassified ------------------------------------------
    if preds is not None:
        correct = labels == preds
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(embedding[correct, 0], embedding[correct, 1], s=24, alpha=0.55,
                   color='#2ca02c', label='Correct', edgecolors='none')
        ax.scatter(embedding[~correct, 0], embedding[~correct, 1], s=42, alpha=0.90,
                   color='#d62728', label='Misclassified', marker='x')
        ax.set_xlabel('UMAP-1', fontsize=24, fontweight='bold')
        ax.set_ylabel('UMAP-2', fontsize=24, fontweight='bold')
        ax.set_title('UMAP Projection \u2014 Correct vs Misclassified', fontsize=27, fontweight='bold')
        ax.legend(fontsize=20, markerscale=1.5)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        _save_figure(fig, output_dir, 'umap_features_correctness.png')

    # --- Plot 3: unsupervised cluster map ------------------------------------------
    cmap = plt.get_cmap('tab10', n_clusters)
    fig, ax = plt.subplots(figsize=(12, 10))
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=28, alpha=0.72, color=cmap(c),
                   label=f'Cluster {c + 1} (n={mask.sum()})', edgecolors='none')
        # Mark each cluster's center in UMAP space
        cx, cy = float(np.mean(embedding[mask, 0])), float(np.mean(embedding[mask, 1]))
        ax.scatter(cx, cy, s=220, color=cmap(c), marker='X', edgecolors='black',
                   linewidths=1.8, zorder=5)
        ax.text(cx, cy, f'{c + 1}', fontsize=18, fontweight='bold', ha='center', va='center', zorder=6)

    ax.set_xlabel('UMAP-1', fontsize=24, fontweight='bold')
    ax.set_ylabel('UMAP-2', fontsize=24, fontweight='bold')
    ax.set_title(f'UMAP Cluster Map (K-means, k={n_clusters}, Silhouette={sil:.3f})',
                 fontsize=27, fontweight='bold')
    ax.legend(fontsize=18, markerscale=1.4, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4)
    _save_figure(fig, output_dir, 'umap_cluster_map.png')

    print(f"    UMAP clusters: k={n_clusters}, silhouette={sil:.4f}")
    print("    Saved: umap_cluster_map.png, umap_cluster_vs_true_class.csv, umap_cluster_metrics.json")


# =====================================================================================
#  FULL EVALUATION REPORT
# =====================================================================================

def generate_full_report(model, test_loader, test_dataset, test_preds, test_targets, test_outputs,
                         test_image_ids, binary, class_names, output_dir, test_metrics,
                         history=None, train_dataset=None, val_dataset=None):
    """Generate every plot, CSV and metric export for the test set.

    Shared by training mode (after training) and evaluate-only mode. Each section is
    wrapped in try/except so one failing plot never blocks the others. Covers:
    confusion matrices, ROC, PR, prediction distribution, per-class metrics, class
    distribution, CSV exports, B-scan aggregation, UMAP, calibration and qualitative
    A-line examples.

    Returns
    -------
    tuple
        (test_metrics, bscan_metrics) - bscan_metrics is None if aggregation was skipped.
    """
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_outputs = np.array(test_outputs)

    print("Generating plots and reports...")

    try:
        plot_confusion_matrix(test_targets, test_preds, output_dir, class_names=class_names)
        plot_confusion_matrix_counts(test_targets, test_preds, output_dir, class_names=class_names)
        print("  ✓ Confusion matrices (A-line level)")
    except Exception as e:
        print(f"  ✗ Error generating confusion matrices: {e}")

    try:
        plot_roc_curve(test_targets, test_outputs, output_dir, binary=binary, class_names=class_names)
        print("  ✓ ROC curve")
    except Exception as e:
        print(f"  ✗ Error generating ROC curve: {e}")

    try:
        plot_pr_curve(test_targets, test_outputs, output_dir, binary=binary, class_names=class_names)
        print("  ✓ Precision-Recall curve")
    except Exception as e:
        print(f"  ✗ Error generating Precision-Recall curve: {e}")

    try:
        plot_prediction_distribution(test_outputs, test_targets, output_dir, binary=binary)
        print("  ✓ Prediction distribution")
    except Exception as e:
        print(f"  ✗ Error generating prediction distribution: {e}")

    try:
        plot_class_metrics(test_targets, test_preds, output_dir, class_names=class_names)
        print("  ✓ Per-class metrics")
    except Exception as e:
        print(f"  ✗ Error generating class metrics: {e}")

    if train_dataset is not None and val_dataset is not None:
        try:
            plot_class_distribution(train_dataset.labels.numpy(), val_dataset.labels.numpy(),
                                    test_targets, output_dir, class_names, binary=binary)
            print("  ✓ Class distribution")
        except Exception as e:
            print(f"  ✗ Error generating class distribution: {e}")

    try:
        pd.DataFrame([test_metrics]).to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)
        if history is not None:
            save_history_csv(history, output_dir)
        save_classification_report_csv(test_targets, test_preds, class_names, output_dir)
        save_predictions_csv(
            test_image_ids if test_image_ids is not None else np.arange(len(test_targets)),
            test_targets, test_preds, test_outputs, class_names, output_dir)
        print("  ✓ CSV exports (metrics, classification report, predictions)")
    except Exception as e:
        print(f"  ✗ Error saving CSV exports: {e}")

    bscan_metrics = None
    if ENABLE_BSCAN_AGGREGATION:
        if test_image_ids is not None:
            try:
                bscan_metrics, *_ = compute_and_save_bscan_metrics(
                    test_image_ids, test_targets, test_preds, test_outputs, binary, class_names,
                    output_dir, method=BSCAN_AGG_METHOD)
                plot_ascan_vs_bscan_comparison(test_metrics, bscan_metrics, output_dir)
                print(f"  ✓ B-scan-level aggregation: Acc={bscan_metrics['bscan_accuracy']:.4f}, "
                      f"F1={bscan_metrics['bscan_f1']:.4f} over {bscan_metrics['num_bscans']} B-scans")
            except Exception as e:
                print(f"  ✗ Error during B-scan-level aggregation: {e}")
        else:
            print("  ✗ Skipping B-scan aggregation: no B-scan (image) IDs available for the test set.")

    if ENABLE_UMAP:
        try:
            device = next(model.parameters()).device
            feats, feat_labels = extract_features_for_umap(model, test_loader, device)
            feat_preds = test_preds
            if len(feats) > UMAP_MAX_SAMPLES:  # subsample for speed
                rng = np.random.RandomState(SEED)
                sub_idx = rng.choice(len(feats), UMAP_MAX_SAMPLES, replace=False)
                feats, feat_labels, feat_preds = feats[sub_idx], feat_labels[sub_idx], feat_preds[sub_idx]
            plot_umap_features(feats, feat_labels, output_dir, class_names, preds=feat_preds,
                               n_clusters=UMAP_N_CLUSTERS, random_state=SEED)
            print("  ✓ UMAP feature projection")
        except Exception as e:
            print(f"  ✗ Error generating UMAP projection: {e}")

    if ENABLE_CALIBRATION_PLOT:
        try:
            ece = plot_calibration_curve(test_targets, test_outputs, output_dir, binary=binary)
            print(f"  ✓ Calibration curve (Expected Calibration Error = {ece:.4f})")
        except Exception as e:
            print(f"  ✗ Error generating calibration curve: {e}")

    if ENABLE_SAMPLE_ALINE_PLOTS:
        try:
            test_alines_np = test_dataset.alines.numpy().squeeze()
            plot_sample_alines_grid(test_alines_np, test_targets, test_preds, output_dir,
                                    class_names, n_per_class=SAMPLE_ALINES_PER_CLASS)
            print("  ✓ Qualitative A-line examples (correct vs misclassified)")
        except Exception as e:
            print(f"  ✗ Error generating qualitative A-line plots: {e}")

    print(f"✓ All plots, CSVs, and metrics saved to {output_dir}")
    return test_metrics, bscan_metrics


# =====================================================================================
#  TRAINING & VALIDATION LOOPS
# =====================================================================================

def _compute_metrics(targets, preds, binary):
    """Accuracy, precision, recall and F1 ('binary' averaging, or 'weighted' for multiclass)."""
    avg = 'binary' if binary else 'weighted'
    return (accuracy_score(targets, preds),
            precision_score(targets, preds, average=avg, zero_division=0),
            recall_score(targets, preds, average=avg, zero_division=0),
            f1_score(targets, preds, average=avg, zero_division=0))


def train_epoch(model, dataloader, optimizer, criterion, device, binary=False, scheduler=None):
    """Run one training epoch (the LR scheduler, if given, steps once per batch).

    Returns
    -------
    tuple
        (loss, accuracy, precision, recall, f1, probabilities, targets)
    """
    model.train()
    epoch_loss = 0
    all_preds, all_targets, all_outputs_list = [], [], []
    batch_progress = tqdm(dataloader, desc="Training Batches", leave=False)

    for inputs, targets in batch_progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if binary:
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets.float())
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            all_outputs_list.extend(probs.detach().cpu().numpy())
        else:
            loss = criterion(outputs, targets)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_outputs_list.append(probs.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        batch_progress.set_postfix({'batch_loss': f"{loss.item():.4f}"})

    epoch_loss /= len(dataloader.dataset)
    all_outputs = np.vstack(all_outputs_list) if not binary and all_outputs_list else np.array(all_outputs_list)
    acc, prec, rec, f1 = _compute_metrics(all_targets, all_preds, binary)
    return epoch_loss, acc, prec, rec, f1, all_outputs, all_targets


def validate(model, dataloader, criterion, device, binary=False):
    """Evaluate the model on a dataloader without gradient updates.

    Returns
    -------
    tuple
        (loss, accuracy, precision, recall, f1, predictions, targets, probabilities)
    """
    model.eval()
    val_loss = 0
    all_preds, all_targets, all_outputs_list = [], [], []
    val_progress = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for inputs, targets in val_progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if binary:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets.float())
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                all_outputs_list.extend(probs.cpu().numpy())
            else:
                loss = criterion(outputs, targets)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_outputs_list.append(probs.cpu().numpy())

            val_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss /= len(dataloader.dataset)
    all_outputs = np.vstack(all_outputs_list) if not binary and all_outputs_list else np.array(all_outputs_list)
    acc, prec, rec, f1 = _compute_metrics(all_targets, all_preds, binary)
    return val_loss, acc, prec, rec, f1, all_preds, all_targets, all_outputs


# =====================================================================================
#  TRAINING ORCHESTRATION
# =====================================================================================

def train_model(model, train_dataset, val_dataset, test_dataset, train_args, output_dir, class_names=None):
    """Train a model, keep the best checkpoint, then evaluate it on the test set.

    Training details
    ----------------
    - Loss: class-weighted CrossEntropy (with label smoothing) or BCEWithLogits (binary).
      Class weights come from the TRAIN labels only.
    - Optimizer: AdamW. Schedule: linear warmup then cosine annealing, stepped per batch.
    - The checkpoint with the lowest validation loss is saved as best_model.pt;
      training stops early after `early_stop_patience` epochs without improvement.

    Parameters
    ----------
    train_args : dict
        Keys: batch_size, epochs, learning_rate, weight_decay, early_stop_patience,
        num_workers, device, binary, use_class_weights, label_smoothing.

    Returns
    -------
    tuple
        (best_model_path, best_val_loss, test_metrics); on evaluation failure
        (None, inf, None).
    """
    loader_kwargs = dict(batch_size=train_args['batch_size'],
                         num_workers=train_args['num_workers'], pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = train_args['device']
    binary = train_args['binary']
    model.to(device)

    # --- Loss function --------------------------------------------------------------
    train_labels_arr = train_dataset.labels.numpy()
    num_classes_for_weights = 2 if binary else int(train_labels_arr.max()) + 1
    use_weights = train_args.get('use_class_weights', False)
    class_weights = (compute_class_weights(train_labels_arr, num_classes_for_weights, binary=binary).to(device)
                     if use_weights else None)

    if binary:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights if use_weights else None)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=train_args.get('label_smoothing', 0.0))

    # --- Optimizer and LR schedule (linear warmup -> cosine annealing) ---------------
    optimizer = optim.AdamW(model.parameters(), lr=train_args['learning_rate'],
                            weight_decay=train_args['weight_decay'])
    warmup_iters = WARMUP_STEPS
    total_steps = len(train_loader) * train_args['epochs']
    cosine_steps = total_steps - warmup_iters
    print(f"Scheduler: Warmup for {warmup_iters} steps, then Cosine Anneal for {cosine_steps} steps.")
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                                   total_iters=warmup_iters)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                                                milestones=[warmup_iters])

    history = {k: [] for k in [
        'train_loss', 'train_acc', 'train_precision', 'train_recall', 'train_f1',
        'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1']}
    best_val_loss = float('inf')
    early_stop_count = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # --- Training loop --------------------------------------------------------------
    print("\nStarting training...")
    for epoch in tqdm(range(train_args['epochs']), desc="Total Epochs"):
        start_time = time.time()

        train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = train_epoch(
            model, train_loader, optimizer, criterion, device, binary=binary, scheduler=scheduler)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = validate(
            model, val_loader, criterion, device, binary=binary)

        time_taken = time.time() - start_time

        for key, value in zip(history.keys(), [train_loss, train_acc, train_prec, train_rec, train_f1,
                                               val_loss, val_acc, val_prec, val_rec, val_f1]):
            history[key].append(value)

        print(f"\nEpoch {epoch + 1}/{train_args['epochs']} | "
              f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {time_taken:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, best_model_path)
            print(f"✓ Saved new best model (Epoch {epoch + 1}) with val_loss: {val_loss:.4f}")
        else:
            early_stop_count += 1

        if early_stop_count >= train_args['early_stop_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- Final evaluation of the best checkpoint on the test set --------------------
    print("\nTraining finished. Evaluating best model on test set...")
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_targets, test_outputs = validate(
            model, test_loader, criterion, device, binary=binary)

        print(f"Test Results - F1: {test_f1:.4f}, Acc: {test_acc:.4f}, Loss: {test_loss:.4f}, "
              f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        test_metrics = {
            'test_loss': test_loss, 'test_accuracy': test_acc, 'test_precision': test_prec,
            'test_recall': test_rec, 'test_f1': test_f1, 'best_epoch': checkpoint['epoch'] + 1,
        }
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump({k: [float(i) for i in v] for k, v in history.items()}, f, indent=4)

        class_names_final = class_names if class_names is not None else (
            ['Normal', 'Cancer'] if binary else ['NORMAL', 'CIS', 'WD-OSCC', 'PD-OSCC'])

        print("Generating plots...")
        try:
            plot_history(history, output_dir)
            print("  ✓ Training history")
        except Exception as e:
            print(f"  ✗ Error generating training history: {e}")

        generate_full_report(
            model, test_loader, test_dataset, test_preds, test_targets, test_outputs,
            test_dataset.image_ids, binary, class_names_final, output_dir, test_metrics,
            history=history, train_dataset=train_dataset, val_dataset=val_dataset)

    except Exception as e:
        print(f"An error occurred during final evaluation: {e}")
        return None, float('inf'), None

    return best_model_path, best_val_loss, test_metrics


def build_model(model_type, num_classes, binary, model_kwargs):
    """Model factory: create an architecture by name (shared by single- and multi-model runs)."""
    model_map = {
        'cnn_lstm': CNN_LSTM,
        'cnn_1d': CNN_1D,
        'lstm_only': LSTM_Only,
        'cnn_gru': CNN_GRU,
        'inception_1d': InceptionNet_1D,
        'transformer_1d': Transformer_1D,
    }
    if model_type not in model_map:
        raise ValueError(f"FATAL: Unknown model type '{model_type}'. "
                         f"Please choose from {list(model_map.keys())}")
    return model_map[model_type](num_classes=num_classes, binary=binary, **model_kwargs)


def run_one_model(model_type, output_root, datasets, class_names, device, run_timestamp):
    """Build, train (or evaluate) and fully report on one architecture.

    Creates its own output folder ``<model>_<task>_<timestamp>``, saves a config
    snapshot (args.json), then either evaluates the checkpoint in EVALUATE_MODEL
    or trains from scratch.

    Returns
    -------
    dict
        Flat summary (model name, status, parameter count, test metrics...) used for
        the cross-model comparison table.
    """
    train_dataset, val_dataset, test_dataset = datasets

    mode = "binary" if BINARY else "multiclass"
    output_dir = os.path.join(output_root, f"{model_type}_{mode}_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 90}\n  MODEL: {model_type.upper()}  ->  {output_dir}\n{'=' * 90}")

    model_kwargs = {
        'input_size': INPUT_SIZE, 'conv1_filters': CONV1_FILTERS, 'conv2_filters': CONV2_FILTERS,
        'conv3_filters': CONV3_FILTERS, 'lstm_hidden_size': LSTM_HIDDEN_SIZE,
        'lstm_num_layers': LSTM_NUM_LAYERS, 'fc_size': FC_SIZE, 'dropout_rate': DROPOUT_RATE,
        'num_inception_blocks': NUM_INCEPTION_BLOCKS, 'transformer_dim': TRANSFORMER_DIM,
        'transformer_nhead': TRANSFORMER_NHEAD, 'transformer_layers': TRANSFORMER_LAYERS,
    }
    num_classes = 2 if BINARY else 4

    print(f"\nInstantiating model: {model_type.upper()}")
    model = build_model(model_type, num_classes, BINARY, model_kwargs)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Save a snapshot of this run's configuration
    config_snapshot = {
        'MODEL_TYPE': model_type, 'BINARY': BINARY, 'DATA_PATH': DATA_PATH, 'SEED': SEED,
        'EPOCHS': EPOCHS, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY, 'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
        'NUM_WORKERS': NUM_WORKERS, 'INPUT_SIZE': INPUT_SIZE, 'num_trainable_params': int(num_params),
        'GROUP_SPLIT_BY_BSCAN': GROUP_SPLIT_BY_BSCAN, 'VAL_SIZE': VAL_SIZE, 'TEST_SIZE': TEST_SIZE,
        'ENABLE_BSCAN_AGGREGATION': ENABLE_BSCAN_AGGREGATION, 'BSCAN_AGG_METHOD': BSCAN_AGG_METHOD,
        'ENABLE_UMAP': ENABLE_UMAP, 'ENABLE_CALIBRATION_PLOT': ENABLE_CALIBRATION_PLOT,
        'ENABLE_SAMPLE_ALINE_PLOTS': ENABLE_SAMPLE_ALINE_PLOTS, 'FIG_DPI': FIG_DPI,
    }
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=4)

    summary = {'model_type': model_type, 'output_dir': output_dir,
               'num_trainable_params': int(num_params), 'status': 'not_started'}

    if EVALUATE_MODEL:
        # --- Evaluation-only mode ---------------------------------------------------
        print("\n--- Running in Evaluation-Only Mode ---")
        checkpoint = torch.load(EVALUATE_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("WARNING: Ensure the CONFIG architecture values match the saved model's architecture!")

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        criterion = nn.BCEWithLogitsLoss() if BINARY else nn.CrossEntropyLoss()

        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_targets, test_outputs = validate(
            model, test_loader, criterion, device, binary=BINARY)

        test_metrics = {'test_loss': test_loss, 'test_accuracy': test_acc, 'test_precision': test_prec,
                        'test_recall': test_rec, 'test_f1': test_f1}
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

        generate_full_report(
            model, test_loader, test_dataset, test_preds, test_targets, test_outputs,
            test_dataset.image_ids, BINARY, class_names, output_dir, test_metrics,
            history=None, train_dataset=train_dataset, val_dataset=val_dataset)
        print(f"✓ Evaluation outputs saved to {output_dir}")
        summary.update(test_metrics)
        summary['status'] = 'evaluated'
    else:
        # --- Training mode ------------------------------------------------------------
        print("\n--- Running in Training Mode ---")
        train_args = {
            'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY, 'early_stop_patience': EARLY_STOP_PATIENCE,
            'num_workers': NUM_WORKERS, 'device': device, 'binary': BINARY,
            'use_class_weights': USE_CLASS_WEIGHTS, 'label_smoothing': LABEL_SMOOTHING,
        }
        best_model_path, best_val_loss, test_metrics = train_model(
            model, train_dataset, val_dataset, test_dataset, train_args, output_dir,
            class_names=class_names)
        summary['best_val_loss'] = best_val_loss
        summary['best_model_path'] = best_model_path
        if test_metrics:
            summary.update(test_metrics)
            summary['status'] = 'trained'
        else:
            summary['status'] = 'failed_evaluation'

    return summary


def run_all_models(model_types, output_root, datasets, class_names, device, run_timestamp):
    """Run every architecture in `model_types` back-to-back in one execution.

    A failure in one model is logged and the run continues (unless STOP_ON_ERROR).
    When all models finish, a combined summary (CSV + JSON) and a cross-model
    comparison plot are saved into ``ALL_MODELS_<timestamp>``.

    Returns
    -------
    pd.DataFrame
        One row per model with its status and test metrics.
    """
    all_summaries = []
    combined_dir = os.path.join(output_root, f"ALL_MODELS_{run_timestamp}")
    os.makedirs(combined_dir, exist_ok=True)

    for i, model_type in enumerate(model_types):
        print(f"\n\n########## Running model {i + 1}/{len(model_types)}: {model_type} ##########")
        try:
            summary = run_one_model(model_type, output_root, datasets, class_names, device, run_timestamp)
        except Exception as e:
            print(f"✗✗✗ Model '{model_type}' failed with an error: {e}")
            traceback.print_exc()
            summary = {'model_type': model_type, 'status': 'error', 'error': str(e)}
            if STOP_ON_ERROR:
                all_summaries.append(summary)
                break
        all_summaries.append(summary)

    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv(os.path.join(combined_dir, 'all_models_summary.csv'), index=False)
    with open(os.path.join(combined_dir, 'all_models_summary.json'), 'w') as f:
        json.dump(all_summaries, f, indent=4, default=str)

    print(f"\n\n{'=' * 90}\n  ALL MODELS FINISHED \u2014 summary saved to {combined_dir}\n{'=' * 90}")
    if 'test_f1' in results_df.columns:
        cols = ['model_type', 'status'] + [c for c in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
                                           if c in results_df.columns]
        print(results_df[cols].to_string(index=False))

    try:
        plot_cross_model_comparison(results_df, combined_dir)
        print(f"  ✓ Cross-model comparison plot saved to {combined_dir}/cross_model_comparison.png")
    except Exception as e:
        print(f"  ✗ Error generating cross-model comparison plot: {e}")

    return results_df


# =====================================================================================
#  MAIN
# =====================================================================================

def main():
    """Entry point: load and split the data once, then run one model or all models."""
    set_seed(SEED)
    setup_plot_style()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load once: every model in this run shares the identical train/val/test split
    try:
        (train_alines, val_alines, test_alines,
         train_labels, val_labels, test_labels,
         train_ids, val_ids, test_ids) = load_data(
            DATA_PATH, binary=BINARY, group_split=GROUP_SPLIT_BY_BSCAN,
            val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SEED)

        train_dataset = OCTAlineDataset(
            train_alines, train_labels, image_ids=train_ids, binary=BINARY,
            augment=USE_ALINE_AUGMENTATION, aug_noise_std=AUG_NOISE_STD,
            aug_max_shift=AUG_MAX_SHIFT, aug_scale_range=AUG_SCALE_RANGE)
        # Validation and test sets are never augmented: evaluation must be deterministic
        val_dataset = OCTAlineDataset(val_alines, val_labels, image_ids=val_ids, binary=BINARY)
        test_dataset = OCTAlineDataset(test_alines, test_labels, image_ids=test_ids, binary=BINARY)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(e)
        return 1

    class_names = ['Normal', 'Cancer'] if BINARY else ['NORMAL', 'CIS', 'WD-OSCC', 'PD-OSCC']
    datasets = (train_dataset, val_dataset, test_dataset)

    if RUN_ALL_MODELS:
        print(f"\n>>> RUN_ALL_MODELS is enabled \u2014 running {len(MODEL_TYPES)} models one by one: {MODEL_TYPES}")
        run_all_models(MODEL_TYPES, OUTPUT_DIR, datasets, class_names, device, run_timestamp)
    else:
        run_one_model(MODEL_TYPE, OUTPUT_DIR, datasets, class_names, device, run_timestamp)

    print("\nScript finished.")
    return 0


if __name__ == "__main__":
    main()
