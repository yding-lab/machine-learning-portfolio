"""Summarise the downloaded data set (counts only, no images) into data/PROFILE.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

RAW = Path("data") / "raw"


def find(name: str) -> Path:
    hits = list(RAW.rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under {RAW}: {sorted(str(p) for p in RAW.rglob('*'))[:30]}")
    return hits[0]


def main() -> int:
    names = {}
    for line in find("msl_synset_words-indexed.txt").read_text().splitlines():
        if line.strip():
            idx, *rest = line.split()
            names[int(idx)] = " ".join(rest)
    root = find("train-calibrated-shuffled.txt").parent
    prof = {"classes": names, "splits": {}}
    for split in ("train", "val", "test"):
        rows = [l.split() for l in (root / f"{split}-calibrated-shuffled.txt").read_text().splitlines() if l.strip()]
        labels = Counter(int(r[1]) for r in rows)
        sizes, modes, instr, missing = Counter(), Counter(), Counter(), 0
        sols = []
        for rel, _ in rows:
            p = root / rel
            if not p.exists():
                missing += 1
                continue
            with Image.open(p) as im:
                sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                modes[im.mode] += 1
            stem = Path(rel).name
            instr[stem[4:6] if len(stem) > 6 else "?"] += 1
            try:
                sols.append(int(stem[:4]))
            except ValueError:
                pass
        prof["splits"][split] = {
            "n": len(rows), "missing_files": missing,
            "per_class": {names.get(k, str(k)): v for k, v in sorted(labels.items())},
            "image_sizes_top": dict(sizes.most_common(5)), "modes": dict(modes),
            "instrument_codes": dict(instr.most_common(6)),
            "sol_range": [min(sols), max(sols)] if sols else None,
            "example_paths": [r[0] for r in rows[:3]],
        }
    Path("data/PROFILE.json").write_text(json.dumps(prof, indent=2) + "\n")
    print(json.dumps({s: {k: v for k, v in d.items() if k in ("n", "missing_files", "sol_range", "modes")}
                      for s, d in prof["splits"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
