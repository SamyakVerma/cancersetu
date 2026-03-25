"""
preprocess.py — Image preprocessing and TFRecord generation for JanArogya.

Reads images from data/raw/, applies augmentation, stratified-splits
70/20/10, writes TFRecords to data/processed/, saves class_weights.json.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --raw data/raw --out data/processed
"""
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info spam

import numpy as np
import tensorflow as tf

# ── Constants ──────────────────────────────────────────────────────────────
ML_ROOT = Path(__file__).parent.parent
DATA_RAW = ML_ROOT / "data" / "raw"
DATA_PROCESSED = ML_ROOT / "data" / "processed"

IMAGE_SIZE = (224, 224)          # EfficientNetB3 input
BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10               # remainder
SEED        = 42

LABEL_MAP = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── Augmentation layers (applied only on training set) ────────────────────
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.083),       # ±15° (15/180 ≈ 0.083 normalised)
    tf.keras.layers.RandomBrightness(0.2),        # ±20%
    tf.keras.layers.RandomZoom(0.10),             # ±10%
    tf.keras.layers.RandomContrast(0.2),
], name="augmentation")


# ── TFRecord helpers ───────────────────────────────────────────────────────

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def image_to_example(image_path: Path, label_idx: int) -> tf.train.Example:
    raw = image_path.read_bytes()
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": _bytes_feature(raw),
        "image/label":   _int64_feature(label_idx),
        "image/filename": _bytes_feature(image_path.name.encode()),
    }))


def parse_tfrecord(example_proto: tf.Tensor, training: bool = False):
    feature_desc = {
        "image/encoded":  tf.io.FixedLenFeature([], tf.string),
        "image/label":    tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    img = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = augment(img, training=True)
        img = tf.clip_by_value(img, 0.0, 1.0)

    label = tf.cast(parsed["image/label"], tf.int32)
    return img, label


# ── Image discovery ────────────────────────────────────────────────────────

def discover_images(raw_dir: Path) -> list[tuple[Path, int]]:
    """
    Walk raw_dir and infer labels from sub-directory names.

    Expected structure (either works):
      data/raw/<source>/<CLASS>/image.jpg
      data/raw/<CLASS>/image.jpg

    CLASS names are mapped through LABEL_MAP (case-insensitive).
    Directories not matching any class are skipped with a warning.
    """
    samples: list[tuple[Path, int]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    skipped_dirs: set[str] = set()

    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in exts:
            continue
        # Walk up parents until we find one matching a class
        label_idx = None
        for parent in img_path.parents:
            name_upper = parent.name.upper()
            if name_upper in LABEL_MAP:
                label_idx = LABEL_MAP[name_upper]
                break
            # Also handle source-specific aliases
            if name_upper in {"BENIGN", "NORMAL"}:
                label_idx = LABEL_MAP["LOW_RISK"]
                break
            if name_upper in {"PRECANCEROUS", "SUSPICIOUS"}:
                label_idx = LABEL_MAP["MEDIUM_RISK"]
                break
            if name_upper in {"MALIGNANT", "CANCEROUS", "POSITIVE"}:
                label_idx = LABEL_MAP["HIGH_RISK"]
                break
            if parent == raw_dir:
                break

        if label_idx is None:
            skipped_dirs.add(img_path.parent.name)
            continue
        samples.append((img_path, label_idx))

    if skipped_dirs:
        print(f"⚠  Skipped directories (no class match): {sorted(skipped_dirs)}")
    return samples


# ── Stratified split ───────────────────────────────────────────────────────

def stratified_split(
    samples: list[tuple[Path, int]],
) -> dict[str, list[tuple[Path, int]]]:
    """Return {'train': [...], 'val': [...], 'test': [...]}."""
    by_class: dict[int, list] = {0: [], 1: [], 2: []}
    for item in samples:
        by_class[item[1]].append(item)

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)

    for label_idx, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    # Shuffle within each split
    for split in splits.values():
        rng.shuffle(split)

    return splits


# ── Class weights ──────────────────────────────────────────────────────────

def compute_class_weights(train_samples: list[tuple[Path, int]]) -> dict[int, float]:
    counts = Counter(label for _, label in train_samples)
    total = sum(counts.values())
    n_classes = len(LABEL_MAP)
    weights = {
        idx: total / (n_classes * count)
        for idx, count in counts.items()
    }
    return weights


# ── Write TFRecords ────────────────────────────────────────────────────────

def write_tfrecords(
    split_name: str,
    samples: list[tuple[Path, int]],
    out_dir: Path,
) -> Path:
    out_path = out_dir / f"{split_name}.tfrecord"
    with tf.io.TFRecordWriter(str(out_path)) as writer:
        for img_path, label_idx in samples:
            try:
                example = image_to_example(img_path, label_idx)
                writer.write(example.SerializeToString())
            except Exception as exc:
                print(f"  ⚠ Skipping {img_path.name}: {exc}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="JanArogya preprocessing pipeline")
    parser.add_argument("--raw", type=Path, default=DATA_RAW)
    parser.add_argument("--out", type=Path, default=DATA_PROCESSED)
    args = parser.parse_args()

    raw_dir: Path = args.raw
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── JanArogya Preprocessing Pipeline ──────────────────────")
    print(f"   Raw dir : {raw_dir}")
    print(f"   Out dir : {out_dir}")
    print(f"   Image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    print(f"   Split   : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} "
          f"train/val/test (stratified)\n")

    t0 = time.time()

    # 1. Discover
    samples = discover_images(raw_dir)
    if not samples:
        print("⚠  No labeled images found in raw_dir.")
        print("   Run download_data.py first, or place images under:")
        print("   data/raw/<LOW_RISK|MEDIUM_RISK|HIGH_RISK>/image.jpg")
        print("   Creating empty TFRecord placeholders...\n")
        for split in ["train", "val", "test"]:
            (out_dir / f"{split}.tfrecord").touch()
        _save_empty_metadata(out_dir)
        return

    print(f"✓ Discovered {len(samples)} images")
    counts = Counter(IDX_TO_LABEL[l] for _, l in samples)
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:<15}: {cnt:>5}")

    # 2. Split
    splits = stratified_split(samples)
    print(f"\n── Split counts ────────────────────────────────────────────")
    for split_name, items in splits.items():
        class_counts = Counter(IDX_TO_LABEL[l] for _, l in items)
        print(f"  {split_name:<6} ({len(items):>5} total): "
              + "  ".join(f"{k}={v}" for k, v in sorted(class_counts.items())))

    # 3. Class weights
    weights = compute_class_weights(splits["train"])
    weights_serializable = {IDX_TO_LABEL[k]: round(v, 4) for k, v in weights.items()}
    weights_path = out_dir / "class_weights.json"
    weights_path.write_text(json.dumps(weights_serializable, indent=2))
    print(f"\n✓ Class weights → {weights_path}")
    for cls, w in weights_serializable.items():
        print(f"  {cls:<15}: {w:.4f}")

    # 4. Write TFRecords
    print(f"\n── Writing TFRecords ────────────────────────────────────────")
    for split_name, items in splits.items():
        out_path = write_tfrecords(split_name, items, out_dir)
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"  {split_name:<6}: {len(items):>5} images → {out_path.name} ({size_mb:.1f} MB)")

    # 5. Save metadata
    metadata = {
        "label_map": LABEL_MAP,
        "image_size": list(IMAGE_SIZE),
        "split_sizes": {s: len(items) for s, items in splits.items()},
        "class_weights": weights_serializable,
        "total_images": len(samples),
        "augmentations": [
            "RandomFlip(horizontal_and_vertical)",
            "RandomRotation(±15°)",
            "RandomBrightness(±20%)",
            "RandomZoom(±10%)",
            "RandomContrast(±20%)",
        ],
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n✓ Metadata → {meta_path}")
    print(f"✓ Done in {time.time() - t0:.1f}s")
    print("\nNext: python scripts/train.py")


def _save_empty_metadata(out_dir: Path) -> None:
    meta = {
        "label_map": LABEL_MAP,
        "image_size": list(IMAGE_SIZE),
        "split_sizes": {"train": 0, "val": 0, "test": 0},
        "class_weights": {"LOW_RISK": 1.0, "MEDIUM_RISK": 1.0, "HIGH_RISK": 1.0},
        "total_images": 0,
        "note": "Placeholder — no images found. Run download_data.py first.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "class_weights.json").write_text(
        json.dumps(meta["class_weights"], indent=2)
    )
    print(f"✓ Empty metadata written to {out_dir}")


if __name__ == "__main__":
    main()
