"""
inference.py — On-device inference for JanArogya.

Importable function:
    from scripts.inference import predict_image
    result = predict_image("path/to/image.jpg")
    result = predict_image(image_bytes)    # bytes also accepted

CLI:
    python scripts/inference.py --image path/to/image.jpg
    python scripts/inference.py --image path/to/image.jpg --model float32

Returns:
    {
        "risk_level": "HIGH_RISK",
        "confidence": 0.89,
        "probabilities": {
            "LOW_RISK": 0.05,
            "MEDIUM_RISK": 0.06,
            "HIGH_RISK": 0.89
        },
        "inference_time_ms": 120,
        "model_variant": "int8"
    }
"""
import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ML_ROOT    = Path(__file__).parent.parent
MODELS_DIR = ML_ROOT / "models"

_INT8_PATH    = MODELS_DIR / "cancersetu_model_int8.tflite"
_FLOAT32_PATH = MODELS_DIR / "cancersetu_model_float32.tflite"
_LABELS_PATH  = MODELS_DIR / "labels.json"

IMAGE_SIZE = (224, 224)
LABEL_MAP  = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}

# Default labels if labels.json not yet written
_DEFAULT_LABELS = {0: "LOW_RISK", 1: "MEDIUM_RISK", 2: "HIGH_RISK"}


# ── Model loader (singleton-style, lazy) ───────────────────────────────────

_cached_interpreter: dict = {}      # variant → tf.lite.Interpreter


def _load_interpreter(variant: str = "int8"):
    """Load and cache a TFLite interpreter. Falls back float32 → error."""
    if variant in _cached_interpreter:
        return _cached_interpreter[variant], variant

    import tensorflow as tf

    candidates = (
        [(_INT8_PATH, "int8"), (_FLOAT32_PATH, "float32")]
        if variant == "int8"
        else [(_FLOAT32_PATH, "float32"), (_INT8_PATH, "int8")]
    )

    for model_path, actual_variant in candidates:
        if model_path.exists() and model_path.stat().st_size > 0:
            interp = tf.lite.Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            _cached_interpreter[actual_variant] = interp
            return interp, actual_variant

    raise FileNotFoundError(
        f"No TFLite model found in {MODELS_DIR}. "
        "Run export_tflite.py first."
    )


def _load_labels() -> dict[int, str]:
    if _LABELS_PATH.exists():
        raw = json.loads(_LABELS_PATH.read_text())
        return {int(k): v for k, v in raw.items()}
    return _DEFAULT_LABELS


# ── Image preprocessing ────────────────────────────────────────────────────

def _preprocess(image_data: Union[str, Path, bytes]) -> np.ndarray:
    """
    Accept a file path (str/Path) or raw bytes.
    Returns float32 numpy array of shape (1, 224, 224, 3) in [0, 1].
    """
    from PIL import Image

    if isinstance(image_data, (str, Path)):
        img = Image.open(image_data).convert("RGB")
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        raise TypeError(f"Expected str/Path/bytes, got {type(image_data)}")

    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)      # → (1, 224, 224, 3)


# ── Public API ─────────────────────────────────────────────────────────────

def predict_image(
    image_data: Union[str, Path, bytes],
    model_variant: str = "int8",
) -> dict:
    """
    Run inference on a single image.

    Args:
        image_data: File path (str/Path) or raw image bytes.
        model_variant: "int8" (default, smaller) or "float32".

    Returns:
        {
            "risk_level": "HIGH_RISK",
            "confidence": 0.89,
            "probabilities": {"LOW_RISK": 0.05, "MEDIUM_RISK": 0.06, "HIGH_RISK": 0.89},
            "inference_time_ms": 120,
            "model_variant": "int8"
        }
    """
    labels = _load_labels()
    img_array = _preprocess(image_data)

    interp, actual_variant = _load_interpreter(model_variant)
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # Quantize input for INT8 models
    input_data = img_array.copy()
    if inp_detail["dtype"] == np.int8:
        scale, zero_point = inp_detail["quantization"]
        input_data = (input_data / scale + zero_point).astype(np.int8)

    t0 = time.perf_counter()
    interp.set_tensor(inp_detail["index"], input_data)
    interp.invoke()
    raw_output = interp.get_tensor(out_detail["index"])
    inference_ms = (time.perf_counter() - t0) * 1000

    # Dequantize output for INT8 models
    if out_detail["dtype"] == np.int8:
        scale, zero_point = out_detail["quantization"]
        raw_output = (raw_output.astype(np.float32) - zero_point) * scale

    probs = raw_output[0].tolist()
    pred_idx = int(np.argmax(probs))

    return {
        "risk_level": labels[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            labels[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "inference_time_ms": round(inference_ms, 1),
        "model_variant": actual_variant,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="JanArogya inference — predict risk level from an image"
    )
    parser.add_argument("--image", required=True,
                        help="Path to image file (jpg/png)")
    parser.add_argument("--model", choices=["int8", "float32"], default="int8",
                        help="TFLite variant to use (default: int8)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        raise SystemExit(1)

    result = predict_image(image_path, model_variant=args.model)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    risk = result["risk_level"]
    emoji_map = {"HIGH_RISK": "🔴", "MEDIUM_RISK": "🟡", "LOW_RISK": "🟢"}
    emoji = emoji_map.get(risk, "⚪")

    print(f"\n── JanArogya Screening Result ──────────────────────────────")
    print(f"  Image          : {image_path.name}")
    print(f"  Risk level     : {emoji} {risk}")
    print(f"  Confidence     : {result['confidence']:.1%}")
    print(f"  Inference time : {result['inference_time_ms']:.0f} ms")
    print(f"  Model variant  : {result['model_variant']}")
    print(f"\n  Class probabilities:")
    for label, prob in sorted(result["probabilities"].items(),
                               key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {label:<15}: {prob:>5.1%}  {bar}")
    print()


if __name__ == "__main__":
    main()
