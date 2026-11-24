"""Langkah 1: analisis video untuk menentukan jumlah frame per video (N_FRAMES)."""
import argparse
import json
import numpy as np
import pandas as pd
from config import DATASET_DIR, OUT, VIDEO_EXTS
from video_utils import video_info

CANDIDATES = [8, 12, 16, 20, 24, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_frames", type=int, default=24,
                    help="batas atas N (menjaga latensi webcam)")
    ap.add_argument("--n_frames", type=int, default=None, help="paksa N tertentu")
    a = ap.parse_args()

    rows = []
    for cdir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        for v in sorted(cdir.rglob("*")):
            if v.suffix.lower() in VIDEO_EXTS:
                try:
                    info = video_info(v)
                except Exception as e:  # noqa
                    info = {"frames": 0, "fps": 0, "width": 0, "height": 0, "duration": 0}
                    print("GAGAL:", v, e)
                rows.append({"label": cdir.name, "path": str(v), **info})
    if not rows:
        raise SystemExit(f"Tidak ada video {VIDEO_EXTS} di {DATASET_DIR}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "video_stats.csv", index=False)
    ok = df[df.frames > 0]

    f = ok.frames.values
    p10 = float(np.percentile(f, 10))
    if a.n_frames:
        n = a.n_frames
    else:
        limit = min(p10, a.max_frames)
        n = max([c for c in CANDIDATES if c <= limit] or [CANDIDATES[0]])

    per_class = ok.groupby("label").agg(videos=("path", "count"), frames_med=("frames", "median"),
                                        dur_med=("duration", "median"))
    summary = {
        "n_videos": int(len(df)), "n_unreadable": int((df.frames == 0).sum()),
        "n_classes": int(df.label.nunique()),
        "frames_min": int(f.min()), "frames_p10": p10, "frames_median": float(np.median(f)),
        "frames_max": int(f.max()), "duration_median_s": float(ok.duration.median()),
        "fps_values": sorted({round(x, 1) for x in ok.fps}),
        "resolutions": sorted({f"{w}x{h}" for w, h in zip(ok.width, ok.height)}),
        "videos_per_class_min": int(per_class.videos.min()),
        "videos_per_class_max": int(per_class.videos.max()),
        "n_frames": int(n),
        "rule": "N = kandidat terbesar <= min(persentil-10 jumlah frame, max_frames) "
                "agar hampir semua video tidak punya frame duplikat",
        "videos_shorter_than_n": int((ok.frames < n).sum()),
    }
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(per_class.to_string())
    print(json.dumps(summary, indent=2))
    if summary["n_classes"] != 27:
        print(f"\n[PERINGATAN] Jumlah kelas terdeteksi {summary['n_classes']}, bukan 27.")
    if summary["videos_per_class_min"] < 5:
        print("[PERINGATAN] Ada kelas dengan <5 video; split dan hasil evaluasi akan kurang andal.")


if __name__ == "__main__":
    main()
