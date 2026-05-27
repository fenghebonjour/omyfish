"""
Download and organize fish datasets for training.

Supported sources:
  - Kaggle 'A Large Scale Fish Dataset':  crowww/a-large-scale-fish-dataset
  - Kaggle fish species dataset:          smit15/fish-species
  - Any dataset with <class>/<image> layout

Prerequisites for Kaggle:
  pip install kaggle
  Place ~/.kaggle/kaggle.json (from https://www.kaggle.com/settings → API)
"""

import argparse
import shutil
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

    args = parser.parse_args()
    if args.cmd == "download":
        download_kaggle(args.dataset, args.output)
    elif args.cmd == "organize":
        organize(args.source, args.output)
    elif args.cmd == "stats":
        stats(args.dir)
    else:
        parser.print_help()
