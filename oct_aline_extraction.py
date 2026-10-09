"""
OCT A-line Extraction
=====================

Extracts fixed-depth A-lines (single image columns) from OCT B-scan images
and saves them, with labels and metadata, into one compressed .npz dataset.

Pipeline
--------
1. Scan the class folders of the selected classification task.
2. For every image, detect the tissue surface (topmost boundary) per column.
3. Every COLUMN_STEP columns, cut an A-line of EXTRACTION_DEPTH_PIXELS
   starting at the detected surface.
4. Pad all A-lines to a common length and (optionally) normalize each to [0, 1].
5. Save everything to ``combined_OCT_dataset.npz``.

Expected folder layout
----------------------
    Binary_Classification_Data/
        Non_Cancer/*.jpg
        OSCC/*.jpg

    Multiclass_Classification_Data/
        Normal/*.jpg
        CIS/*.jpg
        WD_OSCC/*.jpg
        PD_OSCC/*.jpg

Output (.npz) contents
----------------------
    alines   : float32 array, shape (N, padded_length)
    labels   : int32 array,   shape (N,)
    metadata : object array of dicts, one per A-line, with keys
               image_name, class_label, class_name, column_index, padded_length

Note
----
Keep the ``metadata`` array in the output. Downstream scripts use it for
B-scan (image) level aggregation and leak-free, B-scan-aware train/val/test
splitting.
"""

import glob
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - edit these values directly
# =============================================================================

# Which task to run: 'binary' or 'multiclass'
CLASSIFICATION_TASK = 'multiclass'

# Dataset root folder and {label: class folder name} for each task
CLASS_FOLDERS = {
    'binary': {
        'root': 'Binary_Classification_Data',
        'classes': {0: 'Non_Cancer', 1: 'OSCC'},
    },
    'multiclass': {
        'root': 'Multiclass_Classification_Data',
        'classes': {0: 'Normal', 1: 'CIS', 2: 'WD_OSCC', 3: 'PD_OSCC'},
    },
}

IMAGE_GLOB = '*.jpg'                                        # image file pattern
OUTPUT_DIR = f"./5px_extracted_alines_{CLASSIFICATION_TASK}"  # output folder

# Optional crop applied identically to every image,
# e.g. {'x': 0, 'y': 0, 'w': 512, 'h': 512}. Use None for the full image.
CROP = None

COLUMN_STEP = 5                 # take one A-line every N columns
EXTRACTION_DEPTH_PIXELS = 500   # A-line depth below the detected surface (pixels)
NORMALIZE = True                # True: scale each A-line to [0, 1]; False: keep raw intensity

SAVE_VISUALIZATIONS = True      # save sample surface/A-line plots
SAMPLE_VIS_PER_CLASS = 3        # number of visualized images per class (0 disables)

# -----------------------------------------------------------------------------
# Surface-detection parameters
# -----------------------------------------------------------------------------
SURFACE_START_ROW = 5           # ignore the first rows (image-top artifacts)
GRADIENT_THRESHOLD = 5          # minimum upward intensity gradient to count as surface
INTENSITY_THRESHOLD = 40        # minimum pixel intensity to count as surface
MIN_ALINE_LENGTH = 10           # discard A-lines shorter than this (pixels)
SURFACE_SMOOTH_WINDOW = 121     # Savitzky-Golay window for smoothing the surface curve

# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCT_Extractor")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def normalize_aline(aline):
    """
    Min-max scale an A-line to the range [0, 1].

    A constant A-line cannot be scaled: it becomes all ones if its value is
    positive, otherwise all zeros.

    Parameters
    ----------
    aline : np.ndarray
        1-D array of intensities.

    Returns
    -------
    np.ndarray
        float32 array of the same length.
    """
    aline = aline.astype(np.float32)
    lo, hi = aline.min(), aline.max()
    if hi > lo:
        return (aline - lo) / (hi - lo)
    return np.ones_like(aline) if hi > 0 else np.zeros_like(aline)


def crop_image(image, crop):
    """
    Crop an image to the region defined in ``crop``.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    crop : dict or None
        Dict with keys 'x', 'y', 'w', 'h'. If None, the image is returned unchanged.
        The region is clipped to the image bounds.
    """
    if crop is None:
        return image

    img_h, img_w = image.shape
    x, y = max(0, crop['x']), max(0, crop['y'])
    w, h = min(crop['w'], img_w - x), min(crop['h'], img_h - y)
    return image[y:y + h, x:x + w]


# -----------------------------------------------------------------------------
# Surface detection
# -----------------------------------------------------------------------------
def detect_surface(image):
    """
    Detect the tissue surface (topmost boundary) in every column of an image.

    Method
    ------
    1. Enhance contrast (CLAHE) and denoise (Gaussian + median blur).
    2. Per column, take the first row where the smoothed intensity gradient is
       steeply positive and the pixel is bright enough.
    3. If no such row exists, fall back to the first Canny edge pixel.
    4. Fill columns with no detection by interpolation (extrapolate at the ends).
    5. Smooth the whole surface curve with a Savitzky-Golay filter.

    Parameters
    ----------
    image : np.ndarray
        2-D uint8 grayscale image.

    Returns
    -------
    list
        One surface row index per column (ints), or a list of None
        if no surface could be found anywhere.
    """
    height, width = image.shape

    # --- Preprocessing -------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed = clahe.apply(image)
    processed = cv2.GaussianBlur(processed, (7, 7), 0)
    processed = cv2.medianBlur(processed, 9)
    edges = cv2.Canny(processed, 40, 90)

    # --- Per-column detection ------------------------------------------------
    start = min(SURFACE_START_ROW, height - 1)
    surface = []
    for col in range(width):
        column = processed[:, col].astype(np.float32)
        gradient = np.gradient(gaussian_filter1d(column, sigma=2))

        # Primary: first steep upward gradient above the intensity threshold
        candidates = np.where(
            (gradient[start:] > GRADIENT_THRESHOLD) & (column[start:] > INTENSITY_THRESHOLD)
        )[0]
        if candidates.size > 0:
            surface.append(int(candidates[0] + start))
            continue

        # Fallback: first Canny edge pixel in this column
        edge_pixels = np.where(edges[start:, col] > 0)[0]
        surface.append(int(edge_pixels[0] + start) if edge_pixels.size > 0 else None)

    # --- Fill gaps -----------------------------------------------------------
    arr = np.array([v if v is not None else np.nan for v in surface], dtype=float)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0:
        return [None] * width

    if len(valid) > 1:
        # Interpolate missing columns between the first and last detection
        arr[valid[0]:valid[-1] + 1] = np.interp(
            np.arange(valid[0], valid[-1] + 1), valid, arr[valid])
    # Extrapolate (constant) before the first and after the last detection
    arr[:valid[0]] = arr[valid[0]]
    arr[valid[-1] + 1:] = arr[valid[-1]]

    # --- Smooth the surface curve -------------------------------------------
    window = min(SURFACE_SMOOTH_WINDOW, len(arr))
    if window % 2 == 0:          # Savitzky-Golay requires an odd window
        window -= 1
    if window > 2:
        arr = savgol_filter(arr, window, 2)

    arr = np.clip(np.round(arr), 0, height - 1)
    return arr.astype(int).tolist()


# -----------------------------------------------------------------------------
# A-line extraction
# -----------------------------------------------------------------------------
def extract_alines_from_image(image, depth_pixels, column_step):
    """
    Extract fixed-depth A-lines from one image, starting at the detected surface.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    depth_pixels : int
        Number of pixels to extract below the surface (truncated at the image bottom).
    column_step : int
        Take an A-line every ``column_step`` columns.

    Returns
    -------
    alines : list of np.ndarray
        Extracted A-lines (variable length if truncated at the image bottom).
    columns : list of int
        Source column index of each A-line.
    surface : list
        Detected surface row for every column of the image.
    """
    surface = detect_surface(image)
    height, width = image.shape

    alines, columns = [], []
    for col in range(0, width, column_step):
        surface_row = surface[col]
        if surface_row is None:
            continue

        end_row = min(surface_row + depth_pixels, height)
        if end_row <= surface_row:
            continue

        aline = image[surface_row:end_row, col]
        if len(aline) < MIN_ALINE_LENGTH:
            continue

        alines.append(aline)
        columns.append(col)

    return alines, columns, surface


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_sample(image, surface, sample_col, depth_pixels, out_path):
    """
    Save a two-panel PNG for quality control.

    Left panel : image with the detected surface, the extraction-depth boundary
                 and one highlighted sample A-line.
    Right panel: the normalized intensity profile of that sample A-line.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    surface : list
        Detected surface row per column (from ``detect_surface``).
    sample_col : int
        Column whose A-line is highlighted and plotted.
    depth_pixels : int
        Extraction depth below the surface, in pixels.
    out_path : str
        Output PNG file path.
    """
    height, _ = image.shape
    fig, (ax_img, ax_line) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: image + surface overlays
    ax_img.imshow(image, cmap='gray')
    valid = [(i, y) for i, y in enumerate(surface) if y is not None]
    if valid:
        xs, ys = zip(*valid)
        ax_img.plot(xs, ys, 'r-', linewidth=1.5, label='Detected surface')
        bottom = [min(y + depth_pixels, height - 1) for y in ys]
        ax_img.plot(xs, bottom, 'g-', linewidth=1.5, alpha=0.7, label=f'{depth_pixels}px region')

    sample_surface = surface[sample_col]
    if sample_surface is not None:
        sample_end = min(sample_surface + depth_pixels, height - 1)
        ax_img.plot([sample_col, sample_col], [sample_surface, sample_end],
                    'c-', linewidth=1.5, label='Sample A-line')
    ax_img.legend(loc='lower right', fontsize=8)
    ax_img.set_title('Surface detection')

    # Right: the sample A-line profile
    if sample_surface is not None:
        end = min(sample_surface + depth_pixels, height)
        aline = normalize_aline(image[sample_surface:end, sample_col])
        ax_line.plot(aline, 'k-')
        ax_line.set_ylim(0, 1.05)
    ax_line.set_xlabel('Depth from surface (pixels)')
    ax_line.set_ylabel('Normalized intensity')
    ax_line.set_title(f'Sample A-line (col {sample_col})')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Dataset building
# -----------------------------------------------------------------------------
def build_dataset():
    """
    Run the full extraction over all class folders and save the combined dataset.

    Steps: read each image -> optional crop -> extract A-lines -> collect labels
    and metadata -> pad to a common length -> normalize (optional) -> save .npz.
    """
    task_cfg = CLASS_FOLDERS[CLASSIFICATION_TASK]
    root, classes = task_cfg['root'], task_cfg['classes']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_alines, all_labels, all_metadata = [], [], []

    for label, folder_name in classes.items():
        folder_path = os.path.join(root, folder_name)
        image_paths = sorted(glob.glob(os.path.join(folder_path, IMAGE_GLOB)))
        if not image_paths:
            logger.warning(f"No images found in {folder_path}")
            continue

        vis_dir = os.path.join(OUTPUT_DIR, f"class_{label}_{folder_name}_visuals")
        if SAVE_VISUALIZATIONS:
            os.makedirs(vis_dir, exist_ok=True)

        for i, image_path in enumerate(tqdm(image_paths, desc=f"Class {label} ({folder_name})")):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read {image_path}. Skipping.")
                continue
            image = crop_image(image, CROP)

            alines, columns, surface = extract_alines_from_image(
                image, EXTRACTION_DEPTH_PIXELS, COLUMN_STEP)
            if not alines:
                logger.warning(f"No A-lines extracted from {image_path}")
                continue

            # Save QC plots for the first few images of each class
            if SAVE_VISUALIZATIONS and i < SAMPLE_VIS_PER_CLASS:
                sample_col = columns[len(columns) // 2]
                out_name = os.path.splitext(os.path.basename(image_path))[0] + '_sample.png'
                visualize_sample(image, surface, sample_col, EXTRACTION_DEPTH_PIXELS,
                                 os.path.join(vis_dir, out_name))

            # Store each A-line with its label and source information
            for aline, col in zip(alines, columns):
                all_alines.append(aline)
                all_labels.append(label)
                all_metadata.append({
                    'image_name': os.path.basename(image_path),
                    'class_label': label,
                    'class_name': folder_name,
                    'column_index': col,
                })

    if not all_alines:
        logger.error("No A-lines were extracted. Nothing to save.")
        return

    # Pad every A-line (zeros at the end) to the longest length, then normalize
    max_len = max(len(a) for a in all_alines)
    processed = []
    for aline in all_alines:
        padded = np.pad(aline, (0, max_len - len(aline)), mode='constant')
        processed.append(normalize_aline(padded) if NORMALIZE else padded.astype(np.float32))

    for meta in all_metadata:
        meta['padded_length'] = max_len

    alines_array = np.array(processed, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)
    metadata_array = np.array(all_metadata, dtype=object)

    out_file = os.path.join(OUTPUT_DIR, "combined_OCT_dataset.npz")
    np.savez_compressed(out_file, alines=alines_array, labels=labels_array,
                        metadata=metadata_array)

    # Summary
    logger.info(f"Saved {alines_array.shape[0]} A-lines to {out_file}")
    unique, counts = np.unique(labels_array, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  Class {label} ({classes[label]}): {count} A-lines")


if __name__ == "__main__":
    build_dataset()
