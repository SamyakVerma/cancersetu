"""
export_tflite.py — Convert trained H5 model to TFLite for on-device inference.

Produces two variants:
  • cancersetu_model_float32.tflite  — full precision, larger
  • cancersetu_model_int8.tflite     — INT8 quantized, use this in the app

Usage:
    python scripts/export_tflite.py
    python scripts/export_tflite.py --model models/efficientnet_janarogya.h5
    python scripts/export_tflite.py --test-images 100
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────
ML_ROOT = Path(__file__).parent.parent
MODELS_DIR  = ML_ROOT / "models"
DATA_PROCESSED = ML_ROOT / "data" / "processed"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
LABEL_MAP   = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ── Calibration dataset for INT8 quantization ─────────────────────────────

def make_representative_dataset(n_samples: int = 200):
    """Generator of sample images from the validation TFRecord for calibration."""
    tfrecord = DATA_PROCESSED / "val.tfrecord"
    if not tfrecord.exists() or tfrecord.stat().st_size == 0:
        # Fallback: synthetic random images
        print("  ⚠  No val.tfrecord — using random images for INT8 calibration")
        def gen():
            for _ in range(n_samples):
                yield [np.random.rand(1, *IMAGE_SIZE, 3).astype(np.float32)]
        return gen

    feature_desc = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/label":   tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
    }

    def gen():
        ds = tf.data.TFRecordDataset(str(tfrecord))
        count = 0
        for raw in ds:
            if count >= n_samples:
                break
            parsed = tf.io.parse_single_example(raw, feature_desc)
            img = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
            img = tf.image.resize(img, IMAGE_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, 0)   # add batch dim
            yield [img.numpy()]
            count += 1

    return gen


# ── Accuracy evaluation on TFLite model ───────────────────────────────────

def evaluate_tflite(
    tflite_path: Path,
    n_images: int = 100,
) -> tuple[float, float]:
    """Returns (accuracy, avg_inference_ms) on up to n_images from test set."""
    tfrecord = DATA_PROCESSED / "test.tfrecord"
    if not tfrecord.exists() or tfrecord.stat().st_size == 0:
        print(f"  ⚠  No test.tfrecord — skipping accuracy check for {tflite_path.name}")
        return 0.0, 0.0

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail  = interpreter.get_input_details()[0]
    out_detail  = interpreter.get_output_details()[0]

    feature_desc = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/label":   tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
    }

    correct = 0
    total   = 0
    total_ms = 0.0

    ds = tf.data.TFRecordDataset(str(tfrecord))
    for raw in ds:
        if total >= n_images:
            break
        parsed = tf.io.parse_single_example(raw, feature_desc)
        img = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = (tf.cast(img, tf.float32) / 255.0).numpy()
        img_batch = np.expand_dims(img, 0)

        # Handle INT8 quantized inputs
        if inp_detail["dtype"] == np.int8:
            scale, zero_point = inp_detail["quantization"]
            img_batch = (img_batch / scale + zero_point).astype(np.int8)

        t0 = time.perf_counter()
        interpreter.set_tensor(inp_detail["index"], img_batch)
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])
        total_ms += (time.perf_counter() - t0) * 1000

        # Dequantize INT8 output if needed
        if out_detail["dtype"] == np.int8:
            scale, zero_point = out_detail["quantization"]
            out = (out.astype(np.float32) - zero_point) * scale

        pred_idx  = int(np.argmax(out[0]))
        true_idx  = int(parsed["image/label"].numpy())
        correct  += int(pred_idx == true_idx)
        total    += 1

    accuracy = correct / total if total > 0 else 0.0
    avg_ms   = total_ms / total if total > 0 else 0.0
    return accuracy, avg_ms


# ── Conversion helpers ─────────────────────────────────────────────────────

def convert_float32(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []          # no quantization
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"  ✓ float32 TFLite saved → {out_path}")


def convert_int8(model: tf.keras.Model, out_path: Path, n_calib: int = 200) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(n_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"  ✓ INT8 TFLite saved   → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="JanArogya TFLite exporter")
    parser.add_argument("--model", type=Path,
                        default=MODELS_DIR / "efficientnet_janarogya.h5")
    parser.add_argument("--test-images", type=int, default=100,
                        help="Images to use for accuracy check per variant")
    parser.add_argument("--calib-images", type=int, default=200,
                        help="Images for INT8 calibration")
    args = parser.parse_args()

    h5_path: Path = args.model
    float32_path  = MODELS_DIR / "cancersetu_model_float32.tflite"
    int8_path     = MODELS_DIR / "cancersetu_model_int8.tflite"
    labels_path   = MODELS_DIR / "labels.json"

    print("\n── JanArogya TFLite Export ────────────────────────────────────")
    print(f"   Source model : {h5_path}")

    if not h5_path.exists():
        print(f"\n⚠  Model not found at {h5_path}")
        print("   Run python scripts/train.py first.")
        print("   Creating placeholder .tflite files for CI...\n")
        float32_path.touch()
        int8_path.touch()
        labels_path.write_text(json.dumps(
            {str(v): k for k, v in LABEL_MAP.items()}, indent=2))
        return

    # Load model
    print("   Loading model...")
    model = tf.keras.models.load_model(str(h5_path))
    h5_size_mb = h5_path.stat().st_size / 1_048_576
    print(f"   H5 size      : {h5_size_mb:.1f} MB")

    # Save labels
    labels = {str(idx): label for label, idx in LABEL_MAP.items()}
    labels_path.write_text(json.dumps(labels, indent=2))
    print(f"   Labels saved : {labels_path}")

    # ── Convert ──────────────────────────────────────────────────────────
    print("\n── Converting to float32 TFLite ───────────────────────────────")
    convert_float32(model, float32_path)

    print("\n── Converting to INT8 quantized TFLite ────────────────────────")
    convert_int8(model, int8_path, n_calib=args.calib_images)

    # ── Size comparison ───────────────────────────────────────────────────
    f32_mb  = float32_path.stat().st_size / 1_048_576
    int8_mb = int8_path.stat().st_size    / 1_048_576

    print("\n── Size Comparison ─────────────────────────────────────────────")
    print(f"  {'Format':<30} {'Size (MB)':>10}  {'vs H5':>8}")
    print(f"  {'─'*50}")
    print(f"  {'H5 (Keras)':<30} {h5_size_mb:>10.1f}  {'(baseline)':>8}")
    print(f"  {'TFLite float32':<30} {f32_mb:>10.1f}  {f32_mb/h5_size_mb:>7.0%}")
    print(f"  {'TFLite INT8 (use in app)':<30} {int8_mb:>10.1f}  {int8_mb/h5_size_mb:>7.0%}")

    # ── Accuracy check ────────────────────────────────────────────────────
    print(f"\n── Accuracy Check (up to {args.test_images} test images) ───────────────────")
    acc_f32,  ms_f32  = evaluate_tflite(float32_path, args.test_images)
    acc_int8, ms_int8 = evaluate_tflite(int8_path,    args.test_images)

    accuracy_delta = abs(acc_f32 - acc_int8)
    ok = "✓" if accuracy_delta < 0.02 else "⚠"

    print(f"  {'Variant':<30} {'Accuracy':>10}  {'Avg ms/img':>12}")
    print(f"  {'─'*56}")
    print(f"  {'float32':<30} {acc_f32:>9.1%}  {ms_f32:>11.1f}ms")
    print(f"  {'INT8':<30} {acc_int8:>9.1%}  {ms_int8:>11.1f}ms")
    print(f"\n  {ok} Quantization accuracy delta: {accuracy_delta:.2%}  "
          f"(target <2%)")
    if accuracy_delta >= 0.02:
        print("  ⚠  Delta exceeds 2% — consider more calibration images (--calib-images 500)")

    # ── Final summary ─────────────────────────────────────────────────────
    export_summary = {
        "h5_size_mb": round(h5_size_mb, 2),
        "float32_size_mb": round(f32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "float32_accuracy": round(acc_f32, 4),
        "int8_accuracy": round(acc_int8, 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "float32_avg_ms": round(ms_f32, 2),
        "int8_avg_ms": round(ms_int8, 2),
        "models": {
            "float32": str(float32_path),
            "int8": str(int8_path),
            "labels": str(labels_path),
        },
    }
    summary_path = MODELS_DIR / "export_summary.json"
    summary_path.write_text(json.dumps(export_summary, indent=2))

    print(f"\n✓ Export summary → {summary_path}")
    print(f"✓ Use in app     : {int8_path.name}")
    print("\nNext: python scripts/inference.py --image <path/to/image.jpg>")


if __name__ == "__main__":
    main()
