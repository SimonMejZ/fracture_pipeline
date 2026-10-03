"""Scan a folder tree for images Pillow can't open, so bad files are caught
before a training run dies partway through. Doesn't modify anything.
Extra caution step after encountering issues with cv's imread"""
import argparse
from pathlib import Path

from PIL import Image


def scan(root: Path):
    bad = []
    paths = [p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    for p in paths:
        try:
            with Image.open(p) as im:
                im.convert("L").load()
        except Exception as e:
            bad.append((p, str(e)))
    print(f"Scanned {len(paths)} images under {root}")
    if bad:
        print(f"{len(bad)} unreadable:")
        for p, e in bad:
            print(f"  {p}: {e}")
    else:
        print("All images OK.")
    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    scan(Path(args.root))
