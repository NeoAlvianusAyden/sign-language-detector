import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import MODELS_DIR, RESULTS_DIR, OUT
from aggregate import AGGREGATORS
from models import assert_no_recurrent


def load_model(name_or_path):
    p = MODELS_DIR / f"{name_or_path}.keras" if not str(name_or_path).endswith(".keras") else name_or_path
    m = keras.models.load_model(p, compile=False)
    assert_no_recurrent(m)
    return m


def predict_frames(model, frames_rgb):
    """frames_rgb: list frame uint8 RGB (letterbox 224). Resize ke input model, lalu softmax."""
    size = model.input_shape[1]
    arr = np.stack(frames_rgb).astype(np.float32)
    if arr.shape[1] != size:
        arr = tf.image.resize(arr, (size, size), antialias=True).numpy()  # sama dgn training
    return model(arr, training=False).numpy()


def load_bundle(model_name=None):
    """Muat model terbaik (hasil evaluate.py) atau model yang dipilih manual."""
    best = json.loads((RESULTS_DIR / "best_model.json").read_text())
    name = model_name or best["model"]
    cfg = best if name == best["model"] else best["per_model"][name]
    return {
        "name": name, "model": load_model(name), "classes": json.loads((OUT / "classes.json").read_text()),
        "aggregator": cfg["aggregator"], "agg_fn": AGGREGATORS[cfg["aggregator"]],
        "threshold": cfg["threshold"], "n_frames": best["n_frames"],
    }
