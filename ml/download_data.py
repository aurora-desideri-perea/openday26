"""
Download QuickDraw numpy bitmap files for the three inhibitor classes.
  - rectangle  → selective inhibitor (class 0)
  - triangle   → off-target toxicity (class 1)
  - others     → inactive            (class 2)
"""

import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

CATEGORIES = [
    "rectangle",
    "triangle",
    # inactive mix
    "circle",
    "star",
    "zigzag",
    "line",
    "cloud",
]


def download(name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.npy"
    if dest.exists():
        print(f"  {name}.npy already exists, skipping")
        return

    url = f"{BASE_URL}/{name}.npy"
    print(f"  Downloading {name}...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  Saved → {dest}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    print(f"Saving to: {out_dir}\n")
    for cat in CATEGORIES:
        download(cat, out_dir)
    print("\nDone.")
