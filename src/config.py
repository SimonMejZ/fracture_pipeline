"""
Set directories for the entire pipeline. Adjust if structure is changed from original commit. All paths are relative
"""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"    # gitignored: checkpoints, split manifest, scratch
RESULTS_DIR = ROOT / "results"   # tracked

TIBIA_FIBULA_DIR = DATA_DIR / "tibia_fibula"
FRACATLAS_DIR = DATA_DIR / "fracatlas"  # raw download. only needed to re-run filter_leg_images.py
FRACATLAS_LEG_NOHW_DIR = DATA_DIR / "fracatlas_leg_nohw"  # leg only, hardware-positive images excluded (filter_leg_images.py)

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
DEVICE = "mps" if torch.backends.mps.is_available() else ( # global setting in case you're lucky enough to have CUDA
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSIFIER_CKPT = OUTPUT_DIR / "classifier.pt"
SEGMENTATION_CKPT = OUTPUT_DIR / "segmentation.pt"

OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
