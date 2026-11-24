"""Langkah 2: split per-video (dipakai KEDUA model) + ekstraksi frame uniform ke JPEG."""
import argparse
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from config import (DATASET_DIR, OUT, FRAMES_DIR, VIDEO_EXTS, SPLIT, SEED, STORE_SIZE,
                    N_FRAMES, TRAIN_FRAME_MULT, ROOT, EXPECTED_CLASSES)
from video_utils import sample_video


def build_splits():
    classes = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    if len(classes) != EXPECTED_CLASSES:
        print(f"[PERINGATAN] {len(classes)} kelas ditemukan (diharapkan {EXPECTED_CLASSES}). "
              "Jumlah output classifier mengikuti label aktual.")
    rng = np.random.default_rng(SEED)
    rows = []
    for ci, c in enumerate(classes):
        vids = sorted(v for v in (DATASET_DIR / c).rglob("*") if v.suffix.lower() in VIDEO_EXTS)
        order = rng.permutation(len(vids))
        vids = [vids[i] for i in order]
        n = len(vids)
        if n < 3:
            print(f"[PERINGATAN] Kelas '{c}' hanya {n} video -> semua masuk train.")
            n_val = n_test = 0
        else:
            n_test = max(1, round(n * SPLIT[2]))
            n_val = max(1, round(n * SPLIT[1]))
        for k, v in enumerate(vids):
            split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
            rows.append({"video_id": f"{c}__{v.stem}", "video_path": str(v), "label": c,
                         "label_idx": ci, "split": split})
    return classes, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="hapus & ekstrak ulang semua frame")
    a = ap.parse_args()
    if a.force and FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    classes, vdf = build_splits()
    (OUT / "classes.json").write_text(json.dumps(classes, indent=2))
    vdf.to_csv(OUT / "splits.csv", index=False)
    print(vdf.groupby("split").size().to_string())

    rows = []
    for i, r in vdf.iterrows():
        n = N_FRAMES * (TRAIN_FRAME_MULT if r.split == "train" else 1)
        d = FRAMES_DIR / r.split / r.label
        d.mkdir(parents=True, exist_ok=True)
        try:
            frames, idx = sample_video(r.video_path, n, STORE_SIZE)
        except Exception as e:  # noqa
            print("LEWATI:", r.video_path, e)
            continue
        for k, (fr, src) in enumerate(zip(frames, idx)):
            p = d / f"{r.video_id}_f{k:03d}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"frame_path": p.relative_to(ROOT).as_posix(), "video_id": r.video_id,
                         "label": r.label, "label_idx": r.label_idx, "split": r.split,
                         "frame_idx": k, "src_frame": int(src)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(vdf)} video")
    pd.DataFrame(rows).to_csv(OUT / "frames.csv", index=False)
    print("Selesai. N_FRAMES =", N_FRAMES, "| total frame:", len(rows))


if __name__ == "__main__":
    main()
