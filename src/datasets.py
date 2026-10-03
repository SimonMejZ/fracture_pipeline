"""Dataset loaders.

TibiaFibulaDataset  -> data/tibia_fibula/{fractured,not_fractured}/*.png
FracAtlasSegDataset  -> data/fracatlas via the COCO JSON annotation file
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as coco_mask # lazy import
from torch.utils.data import Dataset

from preprocessing import load_and_normalize


def _md5(path: Path) -> str: # Hash images to remove duplicates from unclean datasets
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_holdout(path: Path, pct: int = 15) -> bool:
    """Deterministic content-hash hold-out shared by the classifier and segmenter. the full pipeline
    can then be demoed on images neither model trained on (due to limited amount of data and PoC nature)."""
    return int(_md5(Path(path)), 16) % 100 < pct


class TibiaFibulaDataset(Dataset):
    """Binary fracture / no-fracture classification.
    label 0 = not fractured, label 1 = fractured.
    """

    NOT_FRACTURED_DIRNAMES = ["not_fractured", "normal", "Non_fractured", "Not fractured"] # Fom datasets
    FRACTURED_DIRNAMES = ["fractured", "fracture", "Fractured"]

    def __init__(self, root: Path, transform=None, dedupe: bool = True, source: str = "tibia_fibula",
                 include_not_fractured: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, label, source)
        all_paths = []

        # include_not_fractured=False: the Mendeley set has only 34 unique
        # non-fractured images, several of which aren't even leg X-rays (crops
        # with just a laterality marker) — the classifier learned "looks like a
        # Mendeley normal image" as a shortcut for "not fractured" and threw
        # false alarms on  normal-looking scans. Dropped; FracAtlas's
        # 2000 non-fractured leg images (via CombinedClassifierDataset) cover
        # the negative class instead. (Lack of publicly available data :\)

        label_sources = [self.FRACTURED_DIRNAMES] if not include_not_fractured else \
            [self.NOT_FRACTURED_DIRNAMES, self.FRACTURED_DIRNAMES]
        label_offset = 1 if not include_not_fractured else 0
        for i, candidates in enumerate(label_sources):
            label = i + label_offset
            cls_dir = next((self.root / name for name in candidates if (self.root / name).is_dir()), None)
            if cls_dir is None:
                continue
            for p in cls_dir.glob("*"):
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    all_paths.append((p, label))

        if dedupe:
            # This dataset ships with exact-duplicate files within each class
            # (verified by md5) without this, the same image can land in both
            # the train and val split and inflate val accuracy.
            seen_hashes = set()
            n_dropped = 0 # reporting
            for p, label in all_paths:
                h = _md5(p)
                if h in seen_hashes:
                    n_dropped += 1
                    continue
                seen_hashes.add(h)
                self.samples.append((p, label, source))
            if n_dropped:
                print(f"dropped {n_dropped} exact-duplicate images "
                      f"(kept {len(self.samples)}/{len(all_paths)})")
        else:
            self.samples = [(p, label, source) for p, label in all_paths]

# basic handling funcs.

    def class_counts(self):
        counts = [0, 0]
        for _, label, _ in self.samples:
            counts[label] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        img = load_and_normalize(str(path))
        if self.transform:
            img = self.transform(img)
        return img, label, source


class CombinedClassifierDataset(TibiaFibulaDataset):
    """
    TibiaFibulaDataset (tibia/fibula-specific) plus FracAtlas's leg images as
    extra examples of both classes.
    """

    def __init__(self, tibia_fibula_root: Path, fracatlas_leg_root: Path, transform=None):
        super().__init__(tibia_fibula_root, transform=transform, dedupe=True, source="tibia_fibula",
                          include_not_fractured=False)

        seen_hashes = {_md5(p) for p, _, _ in self.samples}
        for subfolder, label in [("Non_fractured", 0), ("Fractured", 1)]:
            extra_dir = Path(fracatlas_leg_root) / "images" / subfolder
            extra_paths = [p for p in extra_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            n_added, n_dupe = 0, 0
            for p in extra_paths:
                h = _md5(p)
                if h in seen_hashes:
                    n_dupe += 1
                    continue
                seen_hashes.add(h)
                self.samples.append((p, label, "fracatlas"))
                n_added += 1
            print(f"CombinedClassifierDataset: added {n_added} FracAtlas leg {subfolder} images "
                  f"as label={label} ({n_dupe} duplicates skipped)")


class FracAtlasSegDataset(Dataset):
    """Fracture segmentation masks from a FracAtlas-style COCO annotation file.

    Only yields images that actually have >=1 fracture instance (the mask task
    is meaningless on non-fractured scans). Non-fractured images aren't used here.
    
      - the leg-filtered output of filter_leg_images.py: root/images/Fractured/*.jpg
        + root/annotations_leg.json, or
      - the raw FracAtlas download: root/images/... + root/Annotations/COCO JSON/*.json
    """

    def __init__(self, root: Path, img_size=224, augment=False):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.img_size = img_size
        self.augment = augment

        leg_json = self.root / "annotations_leg.json"
        if leg_json.exists():
            json_path = leg_json
        else:
            candidates = list((self.root / "Annotations" / "COCO JSON").glob("*.json"))
            json_path = candidates[0]
        with open(json_path) as f:
            coco = json.load(f)

        self.images_by_id = {im["id"]: im for im in coco["images"]}
        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # body_part_filter is a best-effort filename/category heuristic — FracAtlas
        # doesn't cleanly expose body part in the COCO json, so this keeps every
        # annotated image by default; do the leg-only curation manually (see data/README.md)
        self.image_ids = [iid for iid in self.anns_by_image if iid in self.images_by_id]

    def __len__(self):
        return len(self.image_ids)

    def _resolve_image_path(self, file_name: str) -> Path:
        flat = self.images_dir / file_name
        if flat.exists():
            return flat
        nested = self.images_dir / "Fractured" / file_name  # filter_leg_images.py layout
        if nested.exists():
            return nested
        raise FileNotFoundError(f"Could not find {file_name} under {self.images_dir}")

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.images_by_id[image_id]
        img_path = self._resolve_image_path(info["file_name"])
        img = load_and_normalize(str(img_path))
        h, w = info["height"], info["width"]

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.anns_by_image[image_id]:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                rles = coco_mask.frPyObjects(seg, h, w)
                rle = coco_mask.merge(rles)
            else:
                rle = seg
            m = coco_mask.decode(rle)
            combined_mask = np.maximum(combined_mask, m.astype(np.uint8))

        import cv2
        img_r = cv2.resize(img, (self.img_size, self.img_size))
        mask_r = cv2.resize(combined_mask, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img_r, mask_r = self._augment(img_r, mask_r)

        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_r).long()
        return img_t, mask_t

    def _augment(self, img, mask):
        """Rotation + flip applied identically to image and mask — 263 training
        images is small for a segmenter, so this matters more here than for the
        classifier. Safe to flip here (unlike the classifier): this only affects
        mask-prediction training."""
        import cv2
        h, w = mask.shape

        if np.random.rand() < 0.5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)

        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img, mask
