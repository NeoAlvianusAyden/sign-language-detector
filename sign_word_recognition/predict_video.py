"""Prediksi satu video MOV baru: python predict_video.py path/video.MOV [--model cnn|mobilenetv2]"""
import argparse
import numpy as np
from config import STORE_SIZE
from inference import load_bundle, predict_frames
from video_utils import sample_video

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--model", choices=["cnn", "mobilenetv2"], default=None)
a = ap.parse_args()

b = load_bundle(a.model)
frames, idx = sample_video(a.video, b["n_frames"], STORE_SIZE)
P = predict_frames(b["model"], frames)
s = b["agg_fn"](P)
top = np.argsort(s)[::-1][:3]
print(f"Model: {b['name']} | agregator: {b['aggregator']} | {len(frames)} frame (indeks {idx.tolist()})")
for k in top:
    print(f"  {b['classes'][k]:<25} {s[k]:.3f}")
k = int(top[0])
print("HASIL:", b["classes"][k] if s[k] >= b["threshold"] else f"(tidak yakin) {b['classes'][k]}",
      f"| confidence {s[k]:.2f} | threshold {b['threshold']:.2f}")
