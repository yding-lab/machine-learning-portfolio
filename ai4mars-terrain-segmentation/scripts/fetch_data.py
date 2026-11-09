"""Download AI4Mars v0.6 (NASA JPL, CC BY 4.0), verify its MD5 and extract only the
Curiosity (MSL) NAVCAM images, masks and labels. The 16.2 GB zip is deleted afterwards.

    python scripts/fetch_data.py          # -> $AI4MARS_RAW_DIR (default data/raw/), not committed
"""

import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip?download=1"
MD5 = "daf80a86021253292e6c425f97baa5c6"
PREFIX = "ai4mars-dataset-merged-0.6/msl/ncam/"
RAW = Path(os.getenv("AI4MARS_RAW_DIR", Path(__file__).resolve().parents[1] / "data" / "raw"))


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "ai4mars.zip"
    md5 = hashlib.md5()
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                md5.update(chunk)
    if md5.hexdigest() != MD5:
        sys.exit(f"MD5 mismatch: got {md5.hexdigest()}, expected {MD5}")
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.startswith(PREFIX) and not m.endswith("/")]
        z.extractall(RAW, members)
    size = zpath.stat().st_size
    zpath.unlink()
    print(f"ok: {size / 1e9:.1f} GB zip, md5 verified; extracted {len(members)} files under {RAW / PREFIX}")


if __name__ == "__main__":
    main()
