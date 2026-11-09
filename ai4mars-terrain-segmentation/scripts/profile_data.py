"""Profile the extracted MSL NAVCAM part of AI4Mars -> data/PROFILE.json (counts only, no images)."""

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path(os.getenv("AI4MARS_RAW_DIR", Path(__file__).resolve().parents[1] / "data" / "raw"))
NCAM = RAW / "ai4mars-dataset-merged-0.6" / "msl" / "ncam"
SCLK = re.compile(r"^N[LR][A-Z]_(\d{9})")


def stem(name: str) -> str:
    return name.split("_merged")[0].replace(".png", "").replace(".JPG", "")


def value_counts(paths, limit=None) -> dict:
    c = Counter()
    for p in paths[:limit]:
        v, n = np.unique(np.asarray(Image.open(p)), return_counts=True)
        c.update(dict(zip(v.tolist(), n.tolist())))
    return {str(k): int(v) for k, v in sorted(c.items())}


def main() -> None:
    imgs = sorted((NCAM / "images" / "edr").glob("*.JPG"))
    train = sorted((NCAM / "labels" / "train").glob("*.png"))
    test = {lvl: sorted((NCAM / "labels" / "test" / f"masked-gold-{lvl}-100agree").glob("*.png"))
            for lvl in ("min1", "min2", "min3")}
    img_stems = {stem(p.name) for p in imgs}
    train_stems = {stem(p.name) for p in train}
    test_stems = {stem(p.name) for p in test["min1"]}
    sclk = lambda s: int(SCLK.match(s).group(1)) if SCLK.match(s) else None
    tr_t = sorted(t for t in map(sclk, train_stems) if t)
    te_t = sorted(t for t in map(sclk, test_stems) if t)
    sample_img = Image.open(imgs[0])
    out = {
        "n_images": len(imgs), "n_train_labels": len(train), "n_test_labels": {k: len(v) for k, v in test.items()},
        "train_labels_with_image": len(train_stems & img_stems), "test_labels_with_image": len(test_stems & img_stems),
        "test_also_in_train_labels": len(test_stems & train_stems),
        "image_size_mode": [sample_img.size, sample_img.mode],
        "label_size": list(Image.open(train[0]).size) if train else None,
        "train_label_pixel_values_first2000": value_counts(train, 2000),
        "test_label_pixel_values": {k: value_counts(v) for k, v in test.items()},
        "mxy_values_first200": value_counts(sorted((NCAM / "images" / "mxy").glob("*.png")), 200),
        "rng30_values_first200": value_counts(sorted((NCAM / "images" / "rng-30m").glob("*.png")), 200),
        "sclk_train": [tr_t[0], tr_t[len(tr_t) // 2], tr_t[-1]] if tr_t else None,
        "sclk_test": [te_t[0], te_t[len(te_t) // 2], te_t[-1]] if te_t else None,
        "test_sclk_within_train_range": sum(tr_t[0] <= t <= tr_t[-1] for t in te_t) if tr_t else None,
        "camera_prefix": dict(Counter(s[:3] for s in train_stems)),
        "notes": [p.read_text()[:1500] for p in (NCAM / "labels" / "test" / "NOTE.txt",) if p.exists()],
    }
    Path(__file__).resolve().parents[1].joinpath("data", "PROFILE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "notes"}, indent=1))


if __name__ == "__main__":
    main()
