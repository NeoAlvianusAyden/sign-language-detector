"""Langkah 4: evaluasi adil CNN vs MobileNetV2 (split, frame sampling, & metrik identik).
- Agregator dipilih per model pada VALIDASI (akurasi rata-rata - std pada bootstrap subset frame).
- Threshold confidence dipilih pada VALIDASI. Test hanya untuk pelaporan.
- Model terbaik dipilih berdasarkan macro-F1 video-level VALIDASI (tie -> inferensi lebih cepat)."""
import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from config import (MODELS_DIR, RESULTS_DIR, N_FRAMES, STORE_SIZE, VIDEO_EXTS, SEED)
from aggregate import AGGREGATORS
from data import load_classes, load_frames_df, make_ds
from inference import load_model, predict_frames
from video_utils import sample_video

MODEL_NAMES = ("cnn", "mobilenetv2")


def frame_probs(model, df, size):
    return model.predict(make_ds(df, size, 64), verbose=0)


def group_videos(df, P):
    out = []
    for vid, g in df.groupby("video_id", sort=False):
        idx = g.sort_values("frame_idx").index.values
        out.append((vid, int(g.label_idx.iloc[0]), P[idx]))
    return out


def video_preds(videos, agg):
    S = [agg(P) for _, _, P in videos]
    y = np.array([v[1] for v in videos])
    pred = np.array([int(np.argmax(s)) for s in S])
    conf = np.array([float(s.max()) for s in S])
    return y, pred, conf


def metrics(y, pred, nc):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(nc), average="macro",
                                                 zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f)}


def stability(videos, agg, R=30, drop=0.3):
    """Robustness agregasi: acak buang ~30% frame tiap video, ulangi R kali."""
    rng = np.random.default_rng(SEED)
    accs = []
    for _ in range(R):
        c = 0
        for _, y, P in videos:
            keep = rng.random(len(P)) >= drop
            if keep.sum() < 3:
                keep[rng.choice(len(P), min(3, len(P)), replace=False)] = True
            c += int(np.argmax(agg(P[keep])) == y)
        accs.append(c / len(videos))
    return float(np.mean(accs)), float(np.std(accs))


def pick_threshold(y, pred, conf, target=0.90):
    best = 0.5
    for t in np.arange(0.30, 0.96, 0.05):
        m = conf >= t
        if m.sum() >= 0.5 * len(y) and (pred[m] == y[m]).mean() >= target:
            return float(round(t, 2))
    return best


def time_model(model, n_frames, reps=30):
    size = model.input_shape[1]
    x1 = np.random.randint(0, 255, (1, size, size, 3)).astype(np.float32)
    xn = np.random.randint(0, 255, (n_frames, size, size, 3)).astype(np.float32)
    for _ in range(5):
        model(x1, training=False)
        model(xn, training=False)
    t = time.perf_counter()
    for _ in range(reps):
        model(x1, training=False).numpy()
    per_frame = (time.perf_counter() - t) / reps * 1000
    t = time.perf_counter()
    for _ in range(reps):
        model(xn, training=False).numpy()
    per_video = (time.perf_counter() - t) / reps * 1000
    return {"ms_per_frame_batch1": per_frame, "ms_per_video_batch": per_video}


def save_cm(cm, classes, path, title):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title(title)
    if len(classes) <= 30:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def eval_new_videos(model, agg, classes, new_dir, thr):
    rows = []
    for cdir in sorted(p for p in Path(new_dir).iterdir() if p.is_dir()):
        for v in sorted(cdir.rglob("*")):
            if v.suffix.lower() not in VIDEO_EXTS:
                continue
            frames, _ = sample_video(v, N_FRAMES, STORE_SIZE)  # end-to-end dari video mentah
            s = agg(predict_frames(model, frames))
            k = int(np.argmax(s))
            rows.append({"video": str(v), "true": cdir.name, "pred": classes[k],
                         "confidence": float(s[k]), "accepted": bool(s[k] >= thr)})
    df = pd.DataFrame(rows)
    res = {"n_videos": len(df)}
    d = df[df.true.isin(classes)]
    if len(d):
        yt = d.true.map(classes.index).values
        yp = d.pred.map(classes.index).values
        res.update(metrics(yt, yp, len(classes)))
    return res, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_dir", default=None,
                    help="folder video baru: new_dir/<kelas>/*.MOV (tidak dipakai saat training)")
    a = ap.parse_args()

    classes = load_classes()
    nc = len(classes)
    val_df, test_df = load_frames_df("val"), load_frames_df("test")
    summary, per_model = [], {}

    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.keras"
        if not path.exists():
            print(f"[LEWATI] {path.name} belum ada")
            continue
        print(f"\n##### {name} #####")
        model = load_model(name)
        size = model.input_shape[1]
        Pv, Pt = frame_probs(model, val_df, size), frame_probs(model, test_df, size)
        vv, tv = group_videos(val_df, Pv), group_videos(test_df, Pt)

        # --- bandingkan strategi agregasi ---
        rows = []
        for an, fn in AGGREGATORS.items():
            yv, pv, _ = video_preds(vv, fn)
            yt, pt, _ = video_preds(tv, fn)
            mu, sd = stability(vv, fn)
            rows.append({"aggregator": an, "val_acc": accuracy_score(yv, pv),
                         "val_boot_mean": mu, "val_boot_std": sd, "score": mu - sd,
                         "test_acc": accuracy_score(yt, pt)})
        agg_df = pd.DataFrame(rows).sort_values("score", ascending=False)
        agg_df.to_csv(RESULTS_DIR / f"aggregation_{name}.csv", index=False)
        print(agg_df.round(4).to_string(index=False))
        chosen = agg_df.iloc[0].aggregator
        fn = AGGREGATORS[chosen]

        yv, pv, cv = video_preds(vv, fn)
        thr = pick_threshold(yv, pv, cv)
        val_m = metrics(yv, pv, nc)
        yt, pt, ct = video_preds(tv, fn)
        test_m = metrics(yt, pt, nc)
        cm = confusion_matrix(yt, pt, labels=range(nc))
        np.savetxt(RESULTS_DIR / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",")
        save_cm(cm, classes, RESULTS_DIR / f"confusion_{name}.png",
                f"{name} | test video-level | agg={chosen}")
        p, r, f, s = precision_recall_fscore_support(yt, pt, labels=range(nc), zero_division=0)
        pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s}) \
            .to_csv(RESULTS_DIR / f"per_class_{name}.csv", index=False)

        frame_acc = float((Pt.argmax(1) == test_df.label_idx.values).mean())
        timing = time_model(model, N_FRAMES)
        res = {"model": name, "aggregator": chosen, "threshold": thr,
               "val_video": val_m, "test_video": test_m, "test_frame_accuracy": frame_acc,
               "n_test_videos": len(tv), "n_params": int(model.count_params()),
               "model_size_mb": path.stat().st_size / 1e6, **timing}
        if a.new_dir:
            r_new, df_new = eval_new_videos(model, fn, classes, a.new_dir, thr)
            res["new_videos"] = r_new
            df_new.to_csv(RESULTS_DIR / f"new_videos_{name}.csv", index=False)
        (RESULTS_DIR / f"result_{name}.json").write_text(json.dumps(res, indent=2))
        per_model[name] = res
        summary.append({
            "model": name, "aggregator": chosen, "threshold": thr,
            "test_accuracy": test_m["accuracy"], "test_precision_macro": test_m["precision_macro"],
            "test_recall_macro": test_m["recall_macro"], "test_f1_macro": test_m["f1_macro"],
            "test_frame_accuracy": frame_acc, "ms_per_frame": timing["ms_per_frame_batch1"],
            "ms_per_video": timing["ms_per_video_batch"], "model_size_mb": res["model_size_mb"],
            "n_params": res["n_params"], "val_f1_macro": val_m["f1_macro"],
            **({"new_video_accuracy": res["new_videos"].get("accuracy")} if a.new_dir else {})})

    if not summary:
        raise SystemExit("Belum ada model terlatih.")
    sdf = pd.DataFrame(summary)
    sdf.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print("\n=== PERBANDINGAN (test, video-level) ===")
    print(sdf.round(4).T.to_string())

    best_row = sdf.sort_values(["val_f1_macro", "ms_per_video"], ascending=[False, True]).iloc[0]
    best = {"model": best_row.model, "aggregator": best_row.aggregator,
            "threshold": float(best_row.threshold), "n_frames": N_FRAMES,
            "selection_rule": "macro-F1 video-level pada VALIDASI; tie -> inferensi lebih cepat",
            "per_model": {k: {"aggregator": v["aggregator"], "threshold": v["threshold"]}
                          for k, v in per_model.items()}}
    (RESULTS_DIR / "best_model.json").write_text(json.dumps(best, indent=2))
    print("\nModel terbaik:", best["model"], "| agregator:", best["aggregator"],
          "| threshold:", best["threshold"])


if __name__ == "__main__":
    main()
