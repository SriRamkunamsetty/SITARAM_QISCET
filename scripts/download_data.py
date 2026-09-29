"""Download open datasets used for real-data validation into data/raw/ (never committed).

  uv run python scripts/download_data.py shanghai     # ShanghaiT2DM + T1DM, CC BY 4.0, ~3.4 MB

CGMacros (PhysioNet, CC BY-NC-SA 4.0) and AI-READI (controlled access) must be obtained by each
user under their own licence acceptance; see DATA.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/20444397"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pratibimb-data-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def shanghai() -> None:
    meta = json.loads(_get(FIGSHARE_ARTICLE))
    print(f"ShanghaiT1DM/T2DM — licence: {meta['license']['name']}")
    out = RAW / "shanghai"
    out.mkdir(parents=True, exist_ok=True)
    for f in meta["files"]:
        blob = _get(f["download_url"])
        md5 = hashlib.md5(blob).hexdigest()
        if f.get("computed_md5") and f["computed_md5"] != md5:
            raise SystemExit(f"checksum mismatch for {f['name']}")
        path = out / f["name"]
        path.write_bytes(blob)
        print(f"  saved {path} ({len(blob) / 1e6:.1f} MB, md5 ok)")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
            print(f"  extracted to {out}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["shanghai"]
    for t in targets:
        {"shanghai": shanghai}[t]()
