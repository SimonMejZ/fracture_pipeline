"""End-to-end demo: classify -> Grad-CAM overlay -> segment -> derived metrics.

Usage:
    python infer_demo.py --image path/to/xray.jpg
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from metrics import fracture_length_px, severity_score
from models import build_classifier, build_segmenter
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, classifier_eval_transform, load_and_normalize


def run(image_path: str):
    device = torch.device(config.DEVICE)

    classifier = build_classifier(num_classes=2).to(device)
    classifier.load_state_dict(torch.load(config.CLASSIFIER_CKPT, map_location=device))
    classifier.eval()

    segmenter = build_segmenter(num_classes=2).to(device)
    segmenter.load_state_dict(torch.load(config.SEGMENTATION_CKPT, map_location=device))
    segmenter.eval()

    # see preprocessing file for definitions
    raw_img = load_and_normalize(image_path)  # HxWx3 uint8, CLAHE-normalized 
    input_tensor = classifier_eval_transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
    fractured = bool(probs.argmax().item())
    confidence = float(probs[1])
    print(f"Classifier: fractured={fractured}  confidence={confidence:.3f}")

    if not fractured:
        print("No fracture detected — stopping before segmentation/severity (nothing to measure).")
        return

    cam = GradCAM(model=classifier, target_layers=[classifier.layer4[-1]])
    targets = [ClassifierOutputTarget(1)]  # explain the "fractured" class
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    rgb_float = cv2.resize(raw_img, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    cv2.imwrite(str(config.OUTPUT_DIR / "gradcam_overlay.png"), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM overlay to {config.OUTPUT_DIR / 'gradcam_overlay.png'}")

    seg_input = cv2.resize(raw_img, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32) / 255.0
    seg_tensor = torch.from_numpy(seg_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        seg_out = segmenter(seg_tensor)["out"]
    mask = seg_out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    cv2.imwrite(str(config.OUTPUT_DIR / "fracture_mask.png"), mask * 255)

    if mask.sum() == 0:
        fracture_prob = F.softmax(seg_out, dim=1)[0, 1]
        print(f"WARNING: segmenter found no pixels above its 0.5 threshold "
              f"(peak fracture-class confidence was only {fracture_prob.max().item():.2f}) — "
              f"the classifier is confident a fracture is present, but the segmentation model "
              f"couldn't localize it on this image.")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length_px = fracture_length_px(mask)
    diag_px = float(np.hypot(*mask.shape))
    severity = severity_score( # See metrics.py for calculation steps
        classifier_confidence=confidence,
        fracture_length_px=length_px,
        image_diagonal_px=diag_px,
        num_fracture_instances=len(contours),
    )
    print("Severity (indicative, not a validated clinical scale):", severity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    run(args.image)
