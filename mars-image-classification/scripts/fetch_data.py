"""Download and verify the MSL (Curiosity rover) labeled image data set.

    cd mars-image-classification && python scripts/fetch_data.py     # -> data/raw/ (not committed)

Source: Stanboli & Wagstaff, "Mars surface image (Curiosity rover) labeled data set", NASA/JPL,
Zenodo 2017, doi:10.5281/zenodo.1049137 (listed on data.gov:
https://catalog.data.gov/dataset/mars-surface-image-curiosity-rover-labeled-data-set-version-1).
The Zenodo record is licensed CC BY-SA 4.0, so the images are downloaded at run time and never
committed to this repository.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/records/1049137/files/msl-images.zip?download=1"
MD5 = "b078d8d04d71e04ab7dcfad29ae0c07e"
RAW = Path("data") / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "msl-images.zip"
    if not zpath.exists() or hashlib.md5(zpath.read_bytes()).hexdigest() != MD5:
        r = requests.get(URL, timeout=600, headers={"User-Agent": "ml-portfolio"})
        r.raise_for_status()
        zpath.write_bytes(r.content)
    digest = hashlib.md5(zpath.read_bytes()).hexdigest()
    if digest != MD5:
        raise SystemExit(f"md5 {digest} != published {MD5}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
    print(f"ok: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, md5 verified), extracted to {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
