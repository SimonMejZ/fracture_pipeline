# Tibial Fracture Assessment Pipeline (proof of concept)

Group 4, EMS620U Coursework 1: *assessing and monitoring fractures in a sports health centre*.
This repo is the engineering demo behind the pitch. It is a proof of concept on public data, **not** a clinical tool.
The written analysis is in the accompanying report; this README is the map to the code.

## Pipeline

```
X-ray -> CLAHE normalise -> Classifier (ResNet18, fracture / no fracture)
                                   |-> Grad-CAM overlay (what the CNN looked at)
                                   '-> if fractured: Segmenter (DeepLabV3) -> fracture-line mask
                                                         |-> relative fracture length (px)
                                                         '-> indicative severity score
```

Not built, deliberately: bone density from plain-film pixels (not a validated measure; DXA is the standard),
absolute fracture length in mm (the JPEGs carry no pixel spacing), and healing-time prediction (no public
longitudinal data). Severity is a hand-weighted triage score, not an AO/OTA classification.

Tried and removed: a bilateral symmetry check (flip the opposite limb, ORB + affine registration, SSIM). It only
scored sensibly on a self-mirrored synthetic pair; on the one real same-patient frame in the data its registration
failed, and no public dataset has real left/right pairs to validate or fix it. It is not part of the pipeline.

## Layout

```
src/        all code (flat; modules import each other by name, so run scripts from anywhere via their path)
  config.py preprocessing.py datasets.py models.py metrics.py        library modules
  train_classifier.py train_segmentation.py                          training entry points
  filter_leg_images.py scan_dataset.py                               data preparation / integrity
  poc_report.py infer_demo.py gradcam_audit.py                       evaluation and illustration
data/       README.md only is tracked; datasets are git-ignored (see data/README.md)
results/    tracked evidence: demo/ (hand-picked panels), heldout_sample/ (seeded held-out sample),
            logs/, curated_cases.json
outputs/    git-ignored: classifier.pt, segmentation.pt, holdout_files.json (written by training)
```

## Setup

Use a Python with PyTorch built for your accelerator (the project was run on Apple-silicon MPS with the system
anaconda Python; an older CPU-only venv was far slower).

```bash
pip install -r requirements.txt
```

Fetch and prepare the data as described in [`data/README.md`](data/README.md).

## Run

```bash
python src/filter_leg_images.py --dst data/fracatlas_leg_nohw --exclude-hardware   # once
python src/train_classifier.py   --epochs 20     # writes outputs/classifier.pt, outputs/holdout_files.json
python src/train_segmentation.py --epochs 40     # writes outputs/segmentation.pt

python src/poc_report.py --cases results/curated_cases.json --out results/demo       # raw / CLAHE / Grad-CAM / mask panels
python src/poc_report.py --out results/heldout_sample                                # seeded random held-out cases
python src/infer_demo.py --image path/to/xray.jpg
```

Train/validation membership is a deterministic content hash (`datasets.is_holdout`), shared by both models, so an
image held out from one is held out from the other. Panel titles state whether each case was held out.

## What the results do and don't show

Final classifier: balanced accuracy about 0.96 on the held-out split, but that is carried by the easy groups
(FracAtlas non-fractured 292/300, Mendeley fractured 163/163). **FracAtlas fracture recall is 11/24 (46%)**, which is the
honest detection figure. Segmentation best validation IoU is about 0.29; masks are sometimes empty even when the
classifier is right. These figures come from the final run; its log was not retained, so retrain to reproduce them
(`results/logs/` holds the segmentation log only).

Shortcuts found with Grad-CAM, each partly or fully mitigated:

- **Surgical hardware**: 69/263 fractured leg images vs 2/2,010 non-fractured showed implants, and the model read the
  implant. Those images are excluded.
- **Dataset identity**: the Mendeley set's 34 unique normals made "looks like a Mendeley image" the negative-class
  cue. They were dropped, but held-out Mendeley normals still false-alarm (one on a printed clinic stamp), so
  cross-dataset generalisation is **unresolved**.
- **Not tibia-specific**: FracAtlas's `leg` tag includes ankle, foot, knee and pelvis views.



Data: Mendeley tibia/fibula set and FracAtlas, both CC BY 4.0, attribution in [`data/README.md`](data/README.md).
