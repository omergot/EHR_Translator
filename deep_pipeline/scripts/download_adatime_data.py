#!/usr/bin/env python3
"""Download AdaTime benchmark datasets.

Downloads preprocessed datasets from NTU Data Repository.
Each dataset is a collection of .pt files with train/test splits per domain.

Usage:
    python scripts/download_adatime_data.py --dataset HAR
    python scripts/download_adatime_data.py --all
    python scripts/download_adatime_data.py --list
"""

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

# Dataset download URLs from AdaTime README
# These point to NTU Singapore's data repository
DATASET_URLS = {
    "HAR": "https://researchdata.ntu.edu.sg/api/access/datafile/68370",
    "HHAR": "https://researchdata.ntu.edu.sg/api/access/datafile/68371",
    "WISDM": "https://researchdata.ntu.edu.sg/api/access/datafile/68369",
    "EEG": "https://researchdata.ntu.edu.sg/api/access/datafile/68372",
    "FD": "https://researchdata.ntu.edu.sg/api/access/datafile/108811",
}

# Alternative: construct from persistent IDs
DATASET_DOIS = {
    "HAR": "doi:10.21979/N9/0SYHTZ",
    "HHAR": "doi:10.21979/N9/OWDFXO",
    "WISDM": "doi:10.21979/N9/KJWE5B",
    "EEG": "doi:10.21979/N9/UD1IM9",
    "FD": "doi:10.21979/N9/PU85XN",
}


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress bar."""
    import urllib.request

    print(f"Downloading {desc or url}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"\n  Manual download instructions:")
        print(f"  1. Visit the AdaTime repository: https://github.com/emadeldeen24/AdaTime")
        print(f"  2. Follow dataset links in README to NTU repository")
        print(f"  3. Download and extract to: {Path(dest).parent}")
        return False


def extract_archive(archive_path: str, dest_dir: str):
    """Extract a zip or tar archive."""
    print(f"Extracting {archive_path}...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(dest_dir)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, 'r') as tf:
            tf.extractall(dest_dir)
    else:
        print(f"  Unknown archive format: {archive_path}")
        return False
    print(f"  Extracted to {dest_dir}")
    return True


def verify_dataset(data_dir: str, dataset_name: str) -> bool:
    """Verify that a dataset has the expected .pt files."""
    from src.benchmarks.adatime.data_loader import DATASET_CONFIGS

    if dataset_name not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        print(f"  Dataset directory not found: {dataset_dir}")
        return False

    # Check for at least some .pt files
    pt_files = list(Path(dataset_dir).glob("*.pt"))
    if not pt_files:
        print(f"  No .pt files found in {dataset_dir}")
        return False

    # Check for first scenario's files
    src_id, trg_id = config.scenarios[0]
    required = [
        f"train_{src_id}.pt", f"test_{src_id}.pt",
        f"train_{trg_id}.pt", f"test_{trg_id}.pt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(dataset_dir, f))]
    if missing:
        print(f"  Missing files: {missing}")
        print(f"  Available: {[f.name for f in pt_files]}")
        return False

    print(f"  {dataset_name}: OK ({len(pt_files)} .pt files)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AdaTime datasets")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_URLS.keys()))
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--verify", action="store_true", help="Verify existing datasets")
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "AdaTime" / "data"),
        help="Directory to store datasets",
    )
    args = parser.parse_args()

    if args.list:
        print("Available AdaTime datasets:")
        for name, url in DATASET_URLS.items():
            from src.benchmarks.adatime.data_loader import DATASET_CONFIGS
            config = DATASET_CONFIGS.get(name)
            if config:
                print(f"  {name}: {config.input_channels}ch x {config.sequence_len}ts, "
                      f"{config.num_classes} classes, {len(config.scenarios)} scenarios")
            print(f"    DOI: {DATASET_DOIS.get(name, 'N/A')}")
        return

    if args.verify:
        print(f"Verifying datasets in {args.data_dir}...")
        for name in DATASET_URLS:
            verify_dataset(args.data_dir, name)
        return

    datasets = list(DATASET_URLS.keys()) if args.all else [args.dataset] if args.dataset else []
    if not datasets:
        parser.print_help()
        return

    os.makedirs(args.data_dir, exist_ok=True)

    for name in datasets:
        url = DATASET_URLS[name]
        dest_dir = os.path.join(args.data_dir, name)

        if os.path.exists(dest_dir) and list(Path(dest_dir).glob("*.pt")):
            print(f"{name}: Already exists at {dest_dir}, skipping.")
            continue

        # Download
        archive_path = os.path.join(args.data_dir, f"{name}.zip")
        success = download_file(url, archive_path, desc=name)

        if success and os.path.exists(archive_path):
            # Try to extract
            try:
                extract_archive(archive_path, args.data_dir)
                os.remove(archive_path)
            except Exception as e:
                print(f"  Extraction failed: {e}")
                print(f"  The file may need manual extraction.")

        # Verify
        verify_dataset(args.data_dir, name)

    print("\nDone. Data directory:", args.data_dir)
    print("\nIf automatic download failed, manually download from:")
    print("  HAR:   https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ")
    print("  HHAR:  https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO")
    print("  WISDM: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B")
    print(f"\nExtract each dataset into: {args.data_dir}/<DATASET_NAME>/")


if __name__ == "__main__":
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
