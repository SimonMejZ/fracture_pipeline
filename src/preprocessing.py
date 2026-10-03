"""Shared image preprocessing for both the classifier and segmentation model."""
import cv2
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

from config import IMG_SIZE

cv2.setNumThreads(0)

# some of FracAtlas's leg/non-fractured JPEGs are missing their last  bytes
# (a packaging artifact in the original download) —
# load what's there instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def clahe_normalize(gray: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization, used to improve fracture-line
    visibility more than plain min-max normalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def load_raw(path: str) -> np.ndarray:
    """Read an X-ray exactly as it exists, no CLAHE for showing the
    raw-input -> preprocessed-input step. 
    Same Pillow-based read as load_and_normalize, so it fails/succeeds
    on the same files."""
    try:
        pil_img = Image.open(path).convert("L")
    except Exception as e:
        raise FileNotFoundError(f"{path} ({e})") from e
    gray = np.array(pil_img)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def load_and_normalize(path: str) -> np.ndarray:
    """Read an X-ray as grayscale, CLAHE-normalize, return as 3-channel uint8
    (so pretrained ImageNet backbones can be reused as-is).
    """
    try:
        pil_img = Image.open(path).convert("L")
    except Exception as e:
        raise FileNotFoundError(f"{path} ({e})") from e
    gray = np.array(pil_img)
    gray = clahe_normalize(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Augmentation of traininfg datasets due to lack of publicly available annotated datasets
classifier_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Deliberately no horizontal flip: it swaps medial/lateral on a limb X-ray, so the
    # fibula would appear on the wrong side relative to anything the model saw in training.
    transforms.RandomRotation(7),
    transforms.RandomAdjustSharpness(1.5, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

classifier_eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
