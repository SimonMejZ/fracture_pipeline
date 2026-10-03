"""Filter FracAtlas down to leg-only images, using the body-part flags in
dataset.csv (columns: hand, leg, hip, shoulder, mixed, ... fractured, ...).

Creates:
  data/fracatlas_leg/images/{Fractured,Non_fractured}/*.jpg   (copied, not symlinked,
      so the folder is self-contained and easy to hand off to a teammate)
  data/fracatlas_leg/annotations_leg.json                     (COCO json filtered to
      only the leg+fractured images, for segmentation training)
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import config # Pipeline config


def main(src_dir: Path, dst_dir: Path, exclude_hardware: bool = False):
    csv_path = src_dir / "dataset.csv"
    df = pd.read_csv(csv_path)
    leg_df = df[df["leg"] == 1]
    print(f"Total images: {len(df)}  |  Leg images: {len(leg_df)} "
          f"({leg_df['fractured'].sum()} fractured, {(leg_df['fractured'] == 0).sum()} not)")

    if exclude_hardware:
        # some fractured leg images show surgical hardware,
        # Grad-CAM showed the classifier keying on the
        # hardware itself rather than the fracture line on such images. Dropping
        # them forces training on the fracture line instead of this shortcut.
        n_before = len(leg_df)
        leg_df = leg_df[leg_df["hardware"] == 0]
        print(f"Excluding hardware-positive images: kept {len(leg_df)}/{n_before}")

    dst_images = dst_dir / "images"
    (dst_images / "Fractured").mkdir(parents=True, exist_ok=True)
    (dst_images / "Non_fractured").mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    for _, row in leg_df.iterrows():
        fname = row["image_id"]
        subfolder = "Fractured" if row["fractured"] == 1 else "Non_fractured"
        src_path = src_dir / "images" / subfolder / fname
        if not src_path.exists():
            missing.append(str(src_path))
            continue
        shutil.copy2(src_path, dst_images / subfolder / fname)
        copied += 1
    print(f"Copied {copied} images to {dst_images}")

    leg_fractured_ids = set(leg_df.loc[leg_df["fractured"] == 1, "image_id"])
    coco_candidates = list((src_dir / "Annotations" / "COCO JSON").glob("*.json"))
    with open(coco_candidates[0]) as f:
        coco = json.load(f)

    kept_images = [im for im in coco["images"] if im["file_name"] in leg_fractured_ids]
    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]

    filtered_coco = {
        "info": coco.get("info", {}),
        "images": kept_images,
        "annotations": kept_anns,
        "categories": coco["categories"],
    }
    out_json = dst_dir / "annotations_leg.json"
    with open(out_json, "w") as f:
        json.dump(filtered_coco, f)
    print(f"Filtered COCO annotations: {len(kept_images)} images, {len(kept_anns)} instances "
          f"-> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(config.FRACATLAS_DIR))
    parser.add_argument("--dst", default=str(config.DATA_DIR / "fracatlas_leg"))
    parser.add_argument("--exclude-hardware", action="store_true")
    args = parser.parse_args()
    main(Path(args.src), Path(args.dst), exclude_hardware=args.exclude_hardware)
