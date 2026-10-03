"""Proof-of-concept report: run the full pipeline on a small set of X-rays and
draw one panel per case — input, Grad-CAM, predicted mask vs ground-truth mask,
and the derived metrics — with every case flagged held-out or not.

    python poc_report.py                     # auto: seeded random held-out cases per group
    python poc_report.py --cases cases.json  # hand-picked: ["path", ...]

All cases are reported, including misses — the point is an honest illustration of
each pipeline stage, not a performance claim. Sample sizes here are tiny.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Pipeline imports
import config
from datasets import FracAtlasSegDataset, is_holdout
from metrics import fracture_length_px, severity_score
from models import build_classifier, build_segmenter
from preprocessing import classifier_eval_transform, load_and_normalize, load_raw

FRACTURED_DIRS = {"Fractured", "fracture", "fractured"}


def true_label(path: Path) -> int:
    return int(path.parent.name in FRACTURED_DIRS)


def auto_cases(seed: int, per_group: int):
    rng = random.Random(seed)
    groups = {
        "FracAtlas fractured": config.FRACATLAS_LEG_NOHW_DIR / "images" / "Fractured",
        "FracAtlas non-fractured": config.FRACATLAS_LEG_NOHW_DIR / "images" / "Non_fractured",
        "Mendeley tibia/fibula fractured": config.TIBIA_FIBULA_DIR / "fracture",
        "Mendeley tibia/fibula normal": config.TIBIA_FIBULA_DIR / "normal",
    }
    cases = []
    for name, d in groups.items():
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        held = [p for p in files if is_holdout(p)]
        rng.shuffle(held)
        cases += held[:per_group]
    return cases


def main(cases, out_dir):
    device = torch.device(config.DEVICE)
    clf = build_classifier(num_classes=2).to(device)
    clf.load_state_dict(torch.load(config.CLASSIFIER_CKPT, map_location=device))
    clf.eval()
    seg = build_segmenter(num_classes=2).to(device)
    seg.load_state_dict(torch.load(config.SEGMENTATION_CKPT, map_location=device))
    seg.eval()
    cam = GradCAM(model=clf, target_layers=[clf.layer4[-1]])

    gt_ds = FracAtlasSegDataset(config.FRACATLAS_LEG_NOHW_DIR, img_size=config.IMG_SIZE, augment=False)
    gt_index = {gt_ds.images_by_id[iid]["file_name"]: i for i, iid in enumerate(gt_ds.image_ids)}

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for p in cases:
        p = Path(p)
        if not p.is_absolute():
            p = config.ROOT / p  # case lists are repo-root-relative, so any cwd works
        raw_unprocessed = cv2.resize(load_raw(str(p)), (config.IMG_SIZE, config.IMG_SIZE))
        raw = load_and_normalize(str(p))
        rgb = cv2.resize(raw, (config.IMG_SIZE, config.IMG_SIZE))
        x = classifier_eval_transform(raw).unsqueeze(0).to(device)
        with torch.no_grad():
            conf = F.softmax(clf(x), dim=1)[0, 1].item()
        pred_fx = conf >= 0.5
        gt_fx = true_label(p)
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
        cam_img = show_cam_on_image(rgb.astype(np.float32) / 255.0, heat, use_rgb=True)

        st = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pmask = seg(st)["out"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        gt_mask = None
        if p.name in gt_index:
            gt_mask = gt_ds[gt_index[p.name]][1].numpy().astype(np.uint8)
        iou = None
        if gt_mask is not None:
            union = ((pmask > 0) | (gt_mask > 0)).sum()
            iou = float(((pmask > 0) & (gt_mask > 0)).sum() / union) if union else 0.0

        contours, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        length_px = fracture_length_px(pmask)
        sev = severity_score(conf, length_px, float(np.hypot(*pmask.shape)), len(contours))["severity_score"] \
            if pred_fx else None

        overlay = rgb.copy()
        overlay[pmask > 0] = (0.5 * overlay[pmask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        if gt_mask is not None:
            gcs, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, gcs, -1, (0, 255, 0), 1)

        heldout = is_holdout(p)
        # Compare model output to image annotation to gather veredict: correct, false negative or false positive
        verdict = "correct" if pred_fx == bool(gt_fx) else ("MISS (false negative)" if gt_fx else "FALSE ALARM") 
        rows.append({"file": p.name, "source": p.parent.parent.name, "truth": "fractured" if gt_fx else "normal",
                     "p_fractured": round(conf, 3), "verdict": verdict, "heldout": heldout,
                     "pred_mask_px": int(pmask.sum()), "gt_mask_px": None if gt_mask is None else int(gt_mask.sum()),
                     "mask_iou": None if iou is None else round(iou, 3),
                     "length_ratio": round(length_px / float(np.hypot(*pmask.shape)), 3), "severity": sev})

        fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.9))
        for a, im, t in zip(ax, [raw_unprocessed, rgb, cam_img, overlay],
                            ["Raw X-ray (as downloaded)", "After CLAHE (model input)",
                             "Grad-CAM (region, not line)",
                             "Pred mask (red)" + (" / GT (green)" if gt_mask is not None else " (no GT mask)")]):
            a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
        fig.suptitle(f"{p.name} | truth: {rows[-1]['truth']} | P(fx)={conf:.2f} -> {verdict} | "
                     f"held-out: {'yes' if heldout else 'NO'}"
                     + (f" | IoU={iou:.2f}" if iou is not None else "")
                     + (f" | severity={sev}" if sev is not None else ""), fontsize=9)
        fig.tight_layout()
        fp = out_dir / f"case_{len(rows):02d}_{p.stem}.png"
        fig.savefig(fp, dpi=110); plt.close(fig)
        panels.append(fp)

    with open(out_dir / "poc_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ok = sum(r["verdict"] == "correct" for r in rows)
    print(f"{len(rows)} cases, {ok} correct  (tiny sample — illustrative only)")
    for r in rows:
        print(f"  {r['file']:<18} truth={r['truth']:<9} P(fx)={r['p_fractured']:.2f}  {r['verdict']:<22}"
              f" heldout={r['heldout']}  IoU={r['mask_iou']}")
    print(f"Panels + poc_results.csv in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="JSON list of image paths (hand-picked)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "demo"))
    a = ap.parse_args()
    cs = json.load(open(a.cases)) if a.cases else auto_cases(a.seed, a.per_group)
    main(cs, Path(a.out))
