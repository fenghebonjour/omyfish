"""
Download and organize fish datasets for training.

Supported sources:
  - Kaggle 'A Large Scale Fish Dataset':  crowww/a-large-scale-fish-dataset
  - Kaggle fish species dataset:          smit15/fish-species
  - iNaturalist open API (research-grade observations, no auth required)
  - Any dataset with <class>/<image> layout

Prerequisites for Kaggle:
  pip install kaggle
  Place ~/.kaggle/kaggle.json (from https://www.kaggle.com/settings → API)
"""

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


def download_kaggle(dataset: str, output_dir: str = "data/raw"):
    try:
        import kaggle
    except ImportError:
        print("Install the Kaggle client:  pip install kaggle")
        print("API key setup:  https://www.kaggle.com/docs/api")
        return

    tmp = Path("data/kaggle_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} ...")
    kaggle.api.dataset_download_files(dataset, path=str(tmp), unzip=True)
    print(f"Downloaded to {tmp}")
    print(f"Run 'organize {tmp}' to move images into {output_dir}/<class>/ structure.")


def organize(source_dir: str, output_dir: str = "data/raw"):
    """
    Flatten any nested folder structure into:
        output_dir/<class_name>/<image>
    Class names come from the immediate parent directory of each image.
    Folder names with spaces are preserved; the predictor normalizes them.
    """
    src, out = Path(source_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    for img in src.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        dest_dir = out / img.parent.name
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img, dest_dir / img.name)
        count += 1

    classes = sorted(d.name for d in out.iterdir() if d.is_dir())
    print(f"Organized {count} images into {out}")
    print(f"Classes ({len(classes)}): {classes}")


def stats(data_dir: str = "data/raw"):
    root = Path(data_dir)
    if not root.exists():
        print(f"{data_dir} does not exist.")
        return

    rows = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            rows[cls_dir.name] = n

    total = sum(rows.values())
    print(f"\n{data_dir}  —  {total} images  |  {len(rows)} classes\n")
    for cls, n in sorted(rows.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, max(1, n * 40 // max(total, 1)))
        print(f"  {cls:<35s} {n:5d}  {bar}")


# ---------------------------------------------------------------------------
# North American freshwater species targeted for the NA dataset expansion.
# taxon: scientific name used to query iNaturalist; label: data/raw folder name.
# ---------------------------------------------------------------------------
NA_FRESHWATER_SPECIES = [
    {"taxon": "Micropterus salmoides",   "label": "largemouth_bass"},
    {"taxon": "Micropterus dolomieu",    "label": "smallmouth_bass"},
    {"taxon": "Pomoxis nigromaculatus",  "label": "crappie"},
    {"taxon": "Sander vitreus",          "label": "walleye"},
    {"taxon": "Esox lucius",             "label": "northern_pike"},
    {"taxon": "Perca flavescens",        "label": "yellow_perch"},
    {"taxon": "Salvelinus fontinalis",   "label": "brook_trout"},
    {"taxon": "Salvelinus namaycush",    "label": "lake_trout"},
    {"taxon": "Esox masquinongy",        "label": "muskellunge"},
    {"taxon": "Oncorhynchus mykiss",     "label": "rainbow_trout"},
    {"taxon": "Salmo salar",             "label": "atlantic_salmon"},
    {"taxon": "Alosa sapidissima",       "label": "american_shad"},
    {"taxon": "Acipenser fulvescens",    "label": "lake_sturgeon"},
]


def download_inaturalist(taxon: str, label: str, count: int = 400, output_dir: str = "data/raw"):
    """
    Download research-grade observation photos from iNaturalist (no auth required).
    Photos are saved as data/raw/<label>/inat_<photo_id>.jpg.
    Rate-limited to ~1 req/sec to respect API guidelines.
    """
    dest = Path(output_dir) / label
    dest.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in dest.glob("inat_*.jpg")}
    needed = count - len(existing)
    if needed <= 0:
        print(f"{label}: {len(existing)} images already present, skipping.")
        return

    print(f"Fetching up to {needed} images for '{label}' (taxon: {taxon}) ...")
    collected = 0
    page = 1

    while collected < needed:
        params = urllib.parse.urlencode({
            "taxon_name": taxon,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 200,
            "page": page,
        })
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/observations?{params}",
            headers={"Accept": "application/json", "User-Agent": "OMyFish/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            print(f"  API error on page {page}: {exc}")
            break

        results = data.get("results", [])
        if not results:
            print(f"  No more results after page {page - 1}.")
            break

        for obs in results:
            if collected >= needed:
                break
            for photo in obs.get("photos", [])[:1]:
                url = photo.get("url", "")
                if not url:
                    continue
                # iNaturalist square URLs → medium for better resolution
                url = url.replace("/square.", "/medium.")
                photo_id = str(photo.get("id", ""))
                fname = f"inat_{photo_id}.jpg"
                if fname in existing:
                    continue
                try:
                    urllib.request.urlretrieve(url, dest / fname)
                    existing.add(fname)
                    collected += 1
                    if collected % 50 == 0:
                        print(f"  {label}: {collected}/{needed}")
                except Exception as exc:
                    print(f"  Download failed ({url}): {exc}")

        page += 1
        time.sleep(1.0)

    total = sum(1 for _ in dest.glob("inat_*.jpg"))
    print(f"  {label}: {total} iNaturalist images in {dest}")


def download_na_freshwater(count: int = 400, output_dir: str = "data/raw"):
    """Download all 8 target North American freshwater species from iNaturalist."""
    print(f"Downloading {len(NA_FRESHWATER_SPECIES)} NA freshwater species ({count} images each) ...\n")
    for sp in NA_FRESHWATER_SPECIES:
        download_inaturalist(sp["taxon"], sp["label"], count=count, output_dir=output_dir)
        print()
    stats(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset download and organization helpers.")
    sub = parser.add_subparsers(dest="cmd")

    p_dl = sub.add_parser("download", help="Download a Kaggle dataset by slug")
    p_dl.add_argument("dataset", help="e.g. crowww/a-large-scale-fish-dataset")
    p_dl.add_argument("--output", default="data/raw")

    p_org = sub.add_parser("organize", help="Flatten nested folders into class/<image> layout")
    p_org.add_argument("source")
    p_org.add_argument("--output", default="data/raw")

    p_st = sub.add_parser("stats", help="Print per-class image counts")
    p_st.add_argument("--dir", default="data/raw")

    p_inat = sub.add_parser("inaturalist", help="Download research-grade photos from iNaturalist")
    p_inat.add_argument("--taxon",  required=True, help="Scientific name, e.g. 'Micropterus salmoides'")
    p_inat.add_argument("--label",  required=True, help="Output folder name, e.g. largemouth_bass")
    p_inat.add_argument("--count",  type=int, default=400, help="Target image count (default 400)")
    p_inat.add_argument("--output", default="data/raw")

    p_na = sub.add_parser("download-na-freshwater", help="Download all 8 NA freshwater species from iNaturalist")
    p_na.add_argument("--count",  type=int, default=400, help="Images per species (default 400)")
    p_na.add_argument("--output", default="data/raw")

    args = parser.parse_args()
    if args.cmd == "download":
        download_kaggle(args.dataset, args.output)
    elif args.cmd == "organize":
        organize(args.source, args.output)
    elif args.cmd == "stats":
        stats(args.dir)
    elif args.cmd == "inaturalist":
        download_inaturalist(args.taxon, args.label, count=args.count, output_dir=args.output)
    elif args.cmd == "download-na-freshwater":
        download_na_freshwater(count=args.count, output_dir=args.output)
    else:
        parser.print_help()
