# Data

Datasets are **not in the repo** (large, third-party, CC BY 4.0) and are git-ignored. Download them
manually and place them as below; nothing here is fetched automatically.

## 1. `tibia_fibula/` — Mendeley tibia/fibula radiographs

<https://data.mendeley.com/datasets/2h62x9xzyd/1> · CC BY 4.0 · PNG · sourced from the University of
Gondar referral hospital and the MURA repository (per the dataset page).

```
data/tibia_fibula/
  fracture/*.png      # used: classifier positives (1,123 unique after de-duplication)
  normal/*.png        # NOT used for training (only 34 unique images, see report); kept for qualitative checks
```

About 46% of the files are exact duplicates; `TibiaFibulaDataset` removes them by MD5 hash.

## 2. `fracatlas/` — FracAtlas (raw download, only needed once)

<https://figshare.com/articles/dataset/The_dataset/22363012> · CC BY 4.0 · 4,083 images, hand / leg / hip / shoulder,
with classification, bounding-box and segmentation annotations
(paper: *FracAtlas*, Scientific Data, <https://www.nature.com/articles/s41597-023-02432-4>).

```
data/fracatlas/
  dataset.csv                              # per-image body-part, hardware and fracture flags
  images/{Fractured,Non_fractured}/*.jpg
  Annotations/COCO JSON/COCO_fracture_masks.json
```

Then build the leg-only, hardware-free subset used for training (this is the only thing the code reads):

```bash
python src/filter_leg_images.py --dst data/fracatlas_leg_nohw --exclude-hardware
```

```
data/fracatlas_leg_nohw/
  images/{Fractured,Non_fractured}/*.jpg   # 194 fractured + 2,008 non-fractured
  annotations_leg.json                     # COCO masks for the fractured subset
```

The raw `data/fracatlas/` folder can be deleted afterwards. Known caveats, all verified by hand:

- FracAtlas's `leg` flag is a **body-region tag, not tibia-specific**; ankle, foot, knee and pelvis images are mixed in.
- 69/263 fractured leg images show surgical hardware vs 2/2,010 non-fractured, which the classifier exploited, so they are excluded (`--exclude-hardware`). The flag itself misses some implants.
- 28 of the non-fractured JPEGs are truncated at source; `preprocessing.py` loads them anyway.

## Not used

The Kaggle *Bone Fracture Multi-Region X-ray Data* set was considered and rejected (binary labels only, no body-region tag, no masks).
No public dataset found has same-patient left/right pairs or longitudinal follow-up films, which is why the symmetry check
is only demonstrated (synthetic pair plus one real bilateral frame) and healing-time prediction is not built.

## Notes

- Figures under `results/` are derived from these images; keep the CC BY 4.0 attribution above if you reuse them.
- Verify every count and licence claim against the dataset pages before citing it: the coursework AI policy excludes
  statistics sourced from AI.
