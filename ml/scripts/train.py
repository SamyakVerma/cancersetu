"""
train.py — Two-phase EfficientNetB3 training for JanArogya.

Phase 1: frozen base, train head only  (Adam lr=1e-4, 20 epochs)
Phase 2: unfreeze top 30 layers, fine-tune  (Adam lr=1e-5, 10 epochs)

Usage:
    python scripts/train.py
    python scripts/train.py --epochs1 20 --epochs2 10 --batch 32
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend for servers
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
ML_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ML_ROOT / "data" / "processed"
MODELS_DIR = ML_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
LABEL_MAP = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ── TFRecord parsing (mirrors preprocess.py) ───────────────────────────────

def _parse_example(example_proto, training: bool = False):
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
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)

    label = tf.cast(parsed["image/label"], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


def load_dataset(split: str, batch_size: int, training: bool = False) -> tf.data.Dataset:
    tfrecord = DATA_PROCESSED / f"{split}.tfrecord"
    if not tfrecord.exists() or tfrecord.stat().st_size == 0:
        print(f"⚠  {tfrecord} missing or empty — returning empty dataset")
        return tf.data.Dataset.from_tensors(
            (tf.zeros([batch_size, *IMAGE_SIZE, 3]), tf.zeros([batch_size, NUM_CLASSES]))
        ).take(0)

    ds = tf.data.TFRecordDataset(str(tfrecord))
    ds = ds.map(lambda x: _parse_example(x, training=training),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=1000, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model architecture ─────────────────────────────────────────────────────

def build_model(trainable_base: bool = False) -> tf.keras.Model:
    """EfficientNetB3 backbone + custom classification head."""
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="drop_256")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_128")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="janarogya_efficientnetb3")


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ── Training helpers ───────────────────────────────────────────────────────

def load_class_weights() -> dict[int, float]:
    weights_file = DATA_PROCESSED / "class_weights.json"
    if not weights_file.exists():
        print("⚠  class_weights.json not found — using uniform weights")
        return {0: 1.0, 1: 1.0, 2: 1.0}
    raw = json.loads(weights_file.read_text())
    return {LABEL_MAP[k]: v for k, v in raw.items()}


def make_callbacks(checkpoint_path: Path, patience: int) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_training_curves(histories: list[dict], save_path: Path) -> None:
    """Concatenate phase histories and plot accuracy + loss."""
    metrics_to_plot = ["accuracy", "loss", "auc"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4))

    for ax, metric in zip(axes, metrics_to_plot):
        train_vals, val_vals = [], []
        for h in histories:
            if metric in h:
                train_vals.extend(h[metric])
            if f"val_{metric}" in h:
                val_vals.extend(h[f"val_{metric}"])

        epochs = range(1, len(train_vals) + 1)
        phase_boundary = len(histories[0].get(metric, []))

        if train_vals:
            ax.plot(epochs, train_vals, label="Train")
        if val_vals:
            ax.plot(range(1, len(val_vals) + 1), val_vals, label="Val")
        if phase_boundary and len(histories) > 1:
            ax.axvline(phase_boundary, color="gray", linestyle="--",
                       alpha=0.7, label="Phase 2 start")

        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("JanArogya — EfficientNetB3 Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training curves saved → {save_path}")


def extract_best_metrics(history: dict) -> dict:
    best_idx = int(np.argmin(history["val_loss"]))
    return {
        "val_accuracy":  round(float(history["val_accuracy"][best_idx]), 4),
        "val_auc":       round(float(history.get("val_auc", [0])[best_idx]), 4),
        "val_precision": round(float(history.get("val_precision", [0])[best_idx]), 4),
        "val_recall":    round(float(history.get("val_recall", [0])[best_idx]), 4),
        "val_loss":      round(float(history["val_loss"][best_idx]), 4),
        "best_epoch":    best_idx + 1,
    }


def evaluate_with_confusion_matrix(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> list[list[int]]:
    """Return a 3×3 confusion matrix as nested lists."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for imgs, labels in ds:
        preds = model.predict(imgs, verbose=0)
        pred_idx  = np.argmax(preds, axis=1)
        label_idx = np.argmax(labels.numpy(), axis=1)
        for t, p in zip(label_idx, pred_idx):
            cm[t][p] += 1
    return cm.tolist()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="JanArogya model training")
    parser.add_argument("--epochs1", type=int, default=20, help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=10, help="Phase 2 epochs")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--unfreeze", type=int, default=30,
                        help="Number of top EfficientNetB3 layers to unfreeze in Phase 2")
    args = parser.parse_args()

    print("\n── JanArogya Training Pipeline ──────────────────────────────")
    print(f"   Phase 1 : {args.epochs1} epochs, lr=1e-4 (frozen base)")
    print(f"   Phase 2 : {args.epochs2} epochs, lr=1e-5 (top {args.unfreeze} layers unfrozen)")
    print(f"   Batch   : {args.batch}")
    print(f"   Classes : {list(LABEL_MAP.keys())}\n")

    t_total_start = time.time()

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = load_dataset("train", args.batch, training=True)
    val_ds   = load_dataset("val",   args.batch, training=False)
    test_ds  = load_dataset("test",  args.batch, training=False)
    class_weights = load_class_weights()
    print(f"✓ Class weights: {class_weights}")

    # ── Phase 1: head only ────────────────────────────────────────────────
    print("\n━━━ Phase 1: Training head (base frozen) ━━━━━━━━━━━━━━━━━━━━")
    model = build_model(trainable_base=False)
    compile_model(model, lr=1e-4)
    model.summary(line_length=80, print_fn=lambda x: print("  " + x))

    ckpt_phase1 = MODELS_DIR / "best_phase1.h5"
    t1_start = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs1,
        class_weight=class_weights,
        callbacks=make_callbacks(ckpt_phase1, patience=5),
        verbose=1,
    )
    t1_mins = (time.time() - t1_start) / 60
    print(f"\n✓ Phase 1 complete in {t1_mins:.1f} min")
    metrics_p1 = extract_best_metrics(h1.history)
    print(f"  Best val_accuracy : {metrics_p1['val_accuracy']}")
    print(f"  Best val_auc      : {metrics_p1['val_auc']}")

    # ── Phase 2: fine-tune ────────────────────────────────────────────────
    print(f"\n━━━ Phase 2: Fine-tuning (top {args.unfreeze} layers) ━━━━━━━━━━━━━━━━━━")
    # Reload best phase-1 weights, then selectively unfreeze
    model.load_weights(str(ckpt_phase1))
    base_model = model.get_layer("efficientnetb3")
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze]:
        layer.trainable = False
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Base layers total    : {len(base_model.layers)}")
    print(f"  Trainable base layers: {trainable_count}")

    compile_model(model, lr=1e-5)   # recompile with new lr

    ckpt_final = MODELS_DIR / "efficientnet_janarogya.h5"
    t2_start = time.time()
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs2,
        class_weight=class_weights,
        callbacks=make_callbacks(ckpt_final, patience=5),
        verbose=1,
    )
    t2_mins = (time.time() - t2_start) / 60
    total_mins = (time.time() - t_total_start) / 60
    print(f"\n✓ Phase 2 complete in {t2_mins:.1f} min")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n━━━ Final Evaluation on Test Set ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    model.load_weights(str(ckpt_final))
    metrics_p2 = extract_best_metrics(h2.history)

    test_results = model.evaluate(test_ds, verbose=1)
    test_metric_names = model.metrics_names
    test_metrics = dict(zip(test_metric_names, test_results))
    cm = evaluate_with_confusion_matrix(model, test_ds)

    # ── Save training report ───────────────────────────────────────────────
    report = {
        "model": "EfficientNetB3",
        "training_time_minutes": round(total_mins, 1),
        "phase1": metrics_p1,
        "phase2": metrics_p2,
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "val_accuracy":  metrics_p2["val_accuracy"],
        "val_auc":       metrics_p2["val_auc"],
        "val_precision": metrics_p2["val_precision"],
        "val_recall":    metrics_p2["val_recall"],
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(IDX_TO_LABEL.values()),
        "hyperparameters": {
            "epochs_phase1": args.epochs1,
            "epochs_phase2": args.epochs2,
            "batch_size": args.batch,
            "lr_phase1": 1e-4,
            "lr_phase2": 1e-5,
            "unfreeze_top_n": args.unfreeze,
            "image_size": list(IMAGE_SIZE),
        },
    }
    report_path = MODELS_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # ── Save training curves ───────────────────────────────────────────────
    combined_histories = [h1.history, h2.history]
    plot_training_curves(combined_histories, MODELS_DIR / "training_curves.png")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════════════")
    print("  TRAINING COMPLETE — JanArogya EfficientNetB3")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  val_accuracy : {metrics_p2['val_accuracy']:.4f}")
    print(f"  val_auc      : {metrics_p2['val_auc']:.4f}")
    print(f"  val_precision: {metrics_p2['val_precision']:.4f}")
    print(f"  val_recall   : {metrics_p2['val_recall']:.4f}")
    print(f"  Total time   : {total_mins:.1f} min")
    print(f"\n  Model saved  : {ckpt_final}")
    print(f"  Report saved : {report_path}")
    print(f"\n  Confusion matrix ({', '.join(IDX_TO_LABEL.values())}):")
    for i, row in enumerate(cm):
        print(f"    {IDX_TO_LABEL[i]:<15}: {row}")
    print("═══════════════════════════════════════════════════════════════")
    print("\nNext: python scripts/export_tflite.py")


if __name__ == "__main__":
    main()
