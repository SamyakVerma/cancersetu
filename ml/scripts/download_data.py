"""
download_data.py — Dataset acquisition for JanArogya ML pipeline.

Usage:
    python scripts/download_data.py          # auto-detects kaggle credentials
    python scripts/download_data.py --dry    # show instructions only
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ML_ROOT = Path(__file__).parent.parent          # ml/
DATA_RAW = ML_ROOT / "data" / "raw"
ISIC_DIR = DATA_RAW / "isic"
ORAL_DIR = DATA_RAW / "oral"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


# ── Dataset registry ───────────────────────────────────────────────────────
DATASETS = {
    "isic": {
        "name": "ISIC 2024 Skin Lesion Challenge",
        "source": "Kaggle competition",
        "kaggle_id": "isic-2024-challenge",
        "url": "kaggle.com/competitions/isic-2024-challenge",
        "classes": ["benign", "malignant"],
        "approx_images": 401_059,
        "note": "Large dataset — ~100 GB. Use --subset flag to download a sample.",
    },
    "orchid": {
        "name": "ORCHID Oral Cancer Dataset",
        "source": "GitHub",
        "url": "github.com/NishaChaudhary23/ORCHID",
        "classes": ["normal", "precancerous", "cancerous"],
        "approx_images": 1_073,
        "note": "Clone the repo and copy images to data/raw/oral/",
        "cmd": "git clone https://github.com/NishaChaudhary23/ORCHID data/raw/orchid_raw",
    },
    "roboflow_oral": {
        "name": "Oral Cancer Data (Roboflow — quickstart)",
        "source": "Roboflow Universe",
        "url": "universe.roboflow.com/sagari-vijay/oral-cancer-data",
        "classes": ["normal", "suspicious"],
        "approx_images": 500,
        "note": "Good for quick experiments. Export as 'folder' format.",
    },
}

# Placeholder CSV schema — matches expected format for preprocess.py
CSV_HEADER = ["image_path", "label", "split", "source"]
PLACEHOLDER_ROWS = [
    ["data/raw/isic/sample_001.jpg", "LOW_RISK",    "train", "isic"],
    ["data/raw/isic/sample_002.jpg", "HIGH_RISK",   "train", "isic"],
    ["data/raw/oral/sample_001.jpg", "MEDIUM_RISK", "train", "oral"],
    ["data/raw/oral/sample_002.jpg", "LOW_RISK",    "val",   "oral"],
]


def ensure_dirs() -> None:
    for d in [ISIC_DIR, ORAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ready:\n  {ISIC_DIR}\n  {ORAL_DIR}")


def print_setup_instructions() -> None:
    print("\n" + "═" * 64)
    print("  DATASET SETUP INSTRUCTIONS — JanArogya ML Pipeline")
    print("═" * 64)
    for key, ds in DATASETS.items():
        print(f"\n{'─' * 60}")
        print(f"  [{key.upper()}] {ds['name']}")
        print(f"  Source  : {ds['source']}")
        print(f"  URL     : {ds['url']}")
        print(f"  Classes : {', '.join(ds['classes'])}")
        print(f"  ~Images : {ds['approx_images']:,}")
        print(f"  Note    : {ds['note']}")
        if "cmd" in ds:
            print(f"  Command : {ds['cmd']}")

    print("\n" + "═" * 64)
    print("  KAGGLE SETUP (for ISIC download)")
    print("─" * 64)
    print("  1. Create account at kaggle.com")
    print("  2. Settings → API → Create New Token → downloads kaggle.json")
    print("  3. Move it:  mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
    print("  4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)")
    print("  5. Run: python scripts/download_data.py")
    print("═" * 64 + "\n")


def download_isic_kaggle(subset: bool = False) -> None:
    """Download ISIC 2024 via Kaggle API."""
    try:
        import kaggle  # noqa: F401 — checks install
    except ImportError:
        print("✗ kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    ISIC_DIR.mkdir(parents=True, exist_ok=True)
    cmd_parts = ["kaggle", "competitions", "download",
                 "-c", "isic-2024-challenge", "-p", str(ISIC_DIR)]
    if subset:
        print("⚠  --subset flag set: downloading only metadata files")
        cmd_parts += ["--file", "train-metadata.csv"]

    import subprocess
    print(f"\nRunning: {' '.join(cmd_parts)}")
    result = subprocess.run(cmd_parts, capture_output=False)
    if result.returncode != 0:
        print("✗ Download failed. Check your Kaggle credentials and competition access.")
        sys.exit(1)

    # Unzip if needed
    zips = list(ISIC_DIR.glob("*.zip"))
    if zips:
        import zipfile
        for z in zips:
            print(f"  Extracting {z.name}...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ISIC_DIR)
            z.unlink()
        print(f"✓ Extracted {len(zips)} zip file(s)")


def create_placeholder_csvs() -> None:
    """Write placeholder CSVs so preprocess.py can run with stub data."""
    csv_path = DATA_RAW / "dataset_index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(PLACEHOLDER_ROWS)
    print(f"✓ Placeholder CSV written → {csv_path}")

    # Also write a class distribution summary
    summary = {
        "classes": {"LOW_RISK": 0, "MEDIUM_RISK": 0, "HIGH_RISK": 0},
        "splits": {"train": 0, "val": 0, "test": 0},
        "note": "Placeholder — run after real data is downloaded",
    }
    for row in PLACEHOLDER_ROWS:
        summary["classes"][row[1]] = summary["classes"].get(row[1], 0) + 1
        summary["splits"][row[2]] = summary["splits"].get(row[2], 0) + 1

    summary_path = DATA_RAW / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"✓ Dataset summary written → {summary_path}")


def print_dataset_stats() -> None:
    """Print counts of discovered images per class per source."""
    print("\n── Dataset scan ──────────────────────────────────────────")
    total = 0
    for src_dir in [ISIC_DIR, ORAL_DIR]:
        images = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png"))
        print(f"  {src_dir.name:<10}: {len(images):>6} images")
        total += len(images)
    print(f"  {'TOTAL':<10}: {total:>6} images")

    csv_path = DATA_RAW / "dataset_index.csv"
    if csv_path.exists():
        class_counts: dict[str, int] = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                label = row["label"]
                class_counts[label] = class_counts.get(label, 0) + 1
        print("\n── Class distribution (from CSV index) ───────────────────")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls:<15}: {count:>6}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="JanArogya dataset downloader")
    parser.add_argument("--dry", action="store_true",
                        help="Print instructions only, download nothing")
    parser.add_argument("--subset", action="store_true",
                        help="Download metadata only (faster, for CI)")
    args = parser.parse_args()

    ensure_dirs()

    if args.dry or not KAGGLE_JSON.exists():
        print_setup_instructions()
        if not KAGGLE_JSON.exists():
            print(f"ℹ  kaggle.json not found at {KAGGLE_JSON}")
            print("   Creating placeholder CSVs so the pipeline can run with stub data.\n")
        create_placeholder_csvs()
        print_dataset_stats()
        return

    # Kaggle credentials found — attempt real download
    print(f"✓ Found kaggle credentials at {KAGGLE_JSON}")
    download_isic_kaggle(subset=args.subset)
    create_placeholder_csvs()
    print_dataset_stats()
    print("✓ Done. Next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
