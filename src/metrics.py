"""Derived metrics computed from the classifier + segmentation outputs.

These are deliberately simple, explainable operations (not more deep learning)
so they're easy to justify to a clinical board and to debug.
"""
import cv2
import numpy as np


def fracture_length_px(mask: np.ndarray) -> float:
    """Longest dimension of the fracture-line mask, in pixels, via minAreaRect
    on the largest contour. Convert to mm by multiplying by a pixel spacing
    calibration factor if one is known for the source image — plain de-identified
    JPEGs (e.g. FracAtlas)  do NOT carry DICOM pixel spacing, so treat
    this as a relative/comparative measure, not an absolute one. In a real clinical
    this could be easily transformed to real-world units."""
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(largest)
    return float(max(w, h))


def severity_score(classifier_confidence: float, fracture_length_px: float,
                    image_diagonal_px: float, num_fracture_instances: int) -> dict:
    """Combine three simple, explainable signals into a 0-100 indicative score.
    This is NOT a validated clinical severity scale (e.g. AO/OTA), but
    a triage/screening aid that should be read alongside, not instead of, a
    radiologist's AO/OTA classification of the same films.
    """
    length_ratio = min(fracture_length_px / max(image_diagonal_px, 1e-6), 1.0)
    fragment_penalty = min(num_fracture_instances / 5.0, 1.0)  # FracAtlas caps at 5/scan

    score = 100 * (0.5 * classifier_confidence + 0.35 * length_ratio + 0.15 * fragment_penalty) # final formula for score
    return {
        "severity_score": round(score, 1),
        "components": {
            "classifier_confidence": round(classifier_confidence, 3),
            "length_ratio": round(length_ratio, 3),
            "fragment_penalty": round(fragment_penalty, 3),
        },
    }
