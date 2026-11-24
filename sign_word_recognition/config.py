"""Konfigurasi global. Hanya CNN dan MobileNetV2 yang dipakai (tanpa LSTM/GRU/RNN/Transformer)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Struktur dataset: dataset/<nama_kelas>/*.MOV  (override: set env SIGN_DATASET)
DATASET_DIR = Path(os.environ.get("SIGN_DATASET", ROOT.parent / "kata-kata_ML"))
OUT = ROOT / "outputs"
FRAMES_DIR = OUT / "frames"
MODELS_DIR = OUT / "models"
RESULTS_DIR = OUT / "results"
for _d in (OUT, FRAMES_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
VIDEO_EXTS = {".mov"}
EXPECTED_CLASSES = 27

STORE_SIZE = 224          # ukuran frame yang disimpan (letterbox, rasio aspek dijaga)
CNN_SIZE = 128            # input CNN baseline
MOBILENET_SIZE = 224      # input standar MobileNetV2
SPLIT = (0.70, 0.15, 0.15)  # train / val / test (dipisah per VIDEO, bukan per frame)
TRAIN_FRAME_MULT = 2      # video train diambil N*2 frame (lebih variasi); val/test tepat N frame


def _n_frames(default=16):
    p = OUT / "analysis.json"
    if p.exists():
        return int(json.loads(p.read_text())["n_frames"])
    return default


N_FRAMES = _n_frames()    # ditentukan oleh analyze_videos.py
