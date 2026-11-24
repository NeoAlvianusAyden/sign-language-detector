"""Langkah 3: latih model. Contoh: python train.py --model cnn | mobilenetv2"""
import argparse
import json
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import MODELS_DIR, RESULTS_DIR, SEED, CNN_SIZE, MOBILENET_SIZE, EXPECTED_CLASSES
from data import load_classes, load_frames_df, make_ds
from models import build_cnn, build_mobilenetv2, set_trainable_top


def cbs(path, csv, patience):
    return [
        keras.callbacks.ModelCheckpoint(str(path), monitor="val_accuracy", mode="max",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 3), min_lr=1e-7),
        keras.callbacks.CSVLogger(str(csv), append=True),
    ]


def compile_(model, lr):
    model.compile(keras.optimizers.Adam(lr), keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenetv2"], required=True)
    ap.add_argument("--epochs", type=int, default=60, help="epoch CNN / fase head MobileNetV2")
    ap.add_argument("--ft_epochs", type=int, default=12, help="epoch tiap fase fine-tuning")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    classes = load_classes()
    nc = len(classes)
    if nc != EXPECTED_CLASSES:
        print(f"[PERINGATAN] {nc} kelas pada data aktual (diharapkan {EXPECTED_CLASSES}).")
    size = CNN_SIZE if a.model == "cnn" else MOBILENET_SIZE

    tr, va = load_frames_df("train"), load_frames_df("val")
    cnt = tr.label_idx.value_counts()
    cw = {int(k): float(len(tr) / (nc * v)) for k, v in cnt.items()}
    train_ds, val_ds = make_ds(tr, size, a.batch, shuffle=True), make_ds(va, size, a.batch)

    csv = RESULTS_DIR / f"history_{a.model}.csv"
    csv.unlink(missing_ok=True)
    t0 = time.time()
    final = MODELS_DIR / f"{a.model}.keras"

    if a.model == "cnn":
        model = build_cnn(size, nc)
        model.summary()
        compile_(model, 1e-3)
        h = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=cw,
                      callbacks=cbs(final, csv, patience=12))
        best_val = max(h.history["val_accuracy"])
    else:
        model, base = build_mobilenetv2(size, nc)
        model.summary()
        phases = [("head", 1e-3, a.epochs if a.epochs < 30 else 15, 0),
                  ("ft1", 1e-4, a.ft_epochs, 40),
                  ("ft2", 1e-5, a.ft_epochs, 100)]
        best_val, best_path = -1, None
        for name, lr, ep, n_unf in phases:
            if n_unf == 0:
                base.trainable = False
            else:
                set_trainable_top(base, n_unf)
            compile_(model, lr)
            path = MODELS_DIR / f"mobilenetv2_{name}.keras"
            print(f"\n=== Fase {name} | lr={lr} | layer terbuka={n_unf} ===")
            h = model.fit(train_ds, validation_data=val_ds, epochs=ep, class_weight=cw,
                          callbacks=cbs(path, csv, patience=6))
            v = max(h.history["val_accuracy"])
            if v > best_val:
                best_val, best_path = v, path
        shutil.copy(best_path, final)
        print("Fase terbaik:", best_path.name)

    info = {"model": a.model, "best_val_frame_acc": float(best_val),
            "train_seconds": time.time() - t0, "n_classes": nc, "input_size": size}
    (RESULTS_DIR / f"train_info_{a.model}.json").write_text(json.dumps(info, indent=2))
    print(info, "\nModel tersimpan:", final)


if __name__ == "__main__":
    main()
