"""Read the index of the AI4Mars zip on Zenodo (HTTP range requests, no full download)
and write data/ZIP_PROFILE.json: file counts and bytes per folder, plus example names.

    pip install remotezip && python scripts/list_zip.py
"""

import json
from collections import defaultdict
from pathlib import Path

from remotezip import RemoteZip

URL = "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip?download=1"


def main() -> None:
    with RemoteZip(URL) as z:
        infos = z.infolist()
    folders = defaultdict(lambda: {"files": 0, "bytes": 0, "examples": []})
    for i in infos:
        if i.is_dir():
            continue
        parent = str(Path(i.filename).parent)
        f = folders[parent]
        f["files"] += 1
        f["bytes"] += i.file_size
        if len(f["examples"]) < 3:
            f["examples"].append(Path(i.filename).name)
    out = {"source": URL, "n_entries": len(infos),
           "total_uncompressed_gb": round(sum(i.file_size for i in infos) / 1e9, 2),
           "folders": dict(sorted(folders.items()))}
    Path("data").mkdir(exist_ok=True)
    Path("data/ZIP_PROFILE.json").write_text(json.dumps(out, indent=1))
    for k, v in out["folders"].items():
        print(f"{v['files']:>7} {v['bytes'] / 1e6:>9.1f} MB  {k}  e.g. {v['examples'][0]}")


if __name__ == "__main__":
    main()
