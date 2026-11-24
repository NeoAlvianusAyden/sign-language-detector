"""Aplikasi webcam. Alur: Webcam -> Frame Buffer -> Uniform Sampling -> CNN/MobileNetV2
-> Agregasi -> Confidence Threshold -> Kata.

Kontrol:  SPASI = mulai/akhiri gerakan (Mode A)   m = ganti mode A/B   q = keluar
Mode A (default): pengenalan satu gerakan.  Mode B: deteksi berkelanjutan (smoothing + cooldown)."""
import argparse
import time
from collections import deque
import cv2
import numpy as np
from config import STORE_SIZE
from inference import load_bundle, predict_frames
from video_utils import letterbox, sample_list


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def decide(b, frames, thr):
    P = predict_frames(b["model"], sample_list(frames, b["n_frames"]))
    s = b["agg_fn"](P)
    k = int(np.argmax(s))
    return k, float(s[k]), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenetv2"], default=None,
                    help="default: model terbaik dari evaluate.py")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--max_seconds", type=float, default=6.0, help="batas rekam Mode A")
    ap.add_argument("--min_frames", type=int, default=6)
    ap.add_argument("--flip_input", action="store_true",
                    help="balik horizontal frame sebelum model (jika video training = mirror)")
    ap.add_argument("--no_mirror_display", action="store_true")
    # Mode B
    ap.add_argument("--window", type=float, default=2.0, help="detik buffer Mode B")
    ap.add_argument("--stride", type=float, default=0.4, help="jeda antar-evaluasi Mode B")
    ap.add_argument("--cooldown", type=float, default=2.0, help="batas waktu antar-prediksi")
    ap.add_argument("--confirm", type=int, default=2, help="evaluasi berturut-turut yg sama")
    ap.add_argument("--ema", type=float, default=0.6)
    ap.add_argument("--motion_thr", type=float, default=2.5)
    a = ap.parse_args()

    b = load_bundle(a.model)
    thr = a.threshold if a.threshold is not None else b["threshold"]
    classes = b["classes"]
    print(f"Model={b['name']} agregator={b['aggregator']} N={b['n_frames']} threshold={thr}")

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka.")

    mode = "A"
    recording, rec, rec_t0 = False, [], 0.0
    msg, msg_until = "", 0.0
    # Mode B state
    buf = deque()          # (t, frame224, motion)
    last_gray, last_eval, cooldown_until = None, 0.0, 0.0
    ema, last_k, streak, history = None, -1, 0, deque(maxlen=5)

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        now = time.time()
        model_frame = cv2.flip(raw, 1) if a.flip_input else raw
        f224 = letterbox(cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB), STORE_SIZE)
        gray = cv2.cvtColor(cv2.resize(model_frame, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0 if last_gray is None else float(np.abs(gray - last_gray).mean())
        last_gray = gray

        if mode == "A":
            if recording:
                rec.append(f224)
                if now - rec_t0 >= a.max_seconds:
                    key_stop = True
                else:
                    key_stop = False
            else:
                key_stop = False
        else:
            buf.append((now, f224, motion))
            while buf and now - buf[0][0] > a.window:
                buf.popleft()
            if now >= cooldown_until and now - last_eval >= a.stride and len(buf) >= a.min_frames:
                last_eval = now
                active = np.mean([m > a.motion_thr for _, _, m in buf])
                if active >= 0.3:
                    k, c, s = decide(b, [f for _, f, _ in buf], thr)
                    ema = s if ema is None else a.ema * ema + (1 - a.ema) * s
                    k2 = int(np.argmax(ema))
                    streak = streak + 1 if k2 == last_k else 1
                    last_k = k2
                    if ema[k2] >= thr and streak >= a.confirm:
                        msg, msg_until = f"{classes[k2]} ({ema[k2]:.0%})", now + 2.5
                        history.append(classes[k2])
                        buf.clear()
                        ema, last_k, streak = None, -1, 0
                        cooldown_until = now + a.cooldown
                else:
                    ema, last_k, streak = None, -1, 0

        disp = cv2.flip(raw, 1) if not a.no_mirror_display else raw.copy()
        put(disp, f"Model: {b['name']} | Mode {'A: satu gerakan' if mode == 'A' else 'B: berkelanjutan'}",
            28, (255, 255, 0), 0.6)
        if mode == "A":
            if recording:
                put(disp, f"MEREKAM... {len(rec)} frame (SPASI untuk selesai)", 58, (0, 0, 255))
            else:
                put(disp, "SPASI = mulai gerakan", 58)
        else:
            cd = max(0.0, cooldown_until - now)
            put(disp, f"Mendeteksi... cooldown {cd:.1f}s" if cd else "Mendeteksi...", 58)
            if history:
                put(disp, "Riwayat: " + " > ".join(history), disp.shape[0] - 14, (200, 255, 200), 0.55)
        if now < msg_until:
            put(disp, msg, 100, (0, 255, 0), 0.9)
        cv2.imshow("Pengenalan Kata Bahasa Isyarat", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("m"):
            mode = "B" if mode == "A" else "A"
            recording, rec = False, []
            buf.clear()
            ema, last_k, streak = None, -1, 0
        if mode == "A" and (k == 32 or (recording and key_stop)):
            if not recording:
                recording, rec, rec_t0 = True, [], now
            else:
                recording = False
                if len(rec) < a.min_frames:
                    msg, msg_until = "Gerakan terlalu singkat", now + 2.0
                else:
                    kk, c, _ = decide(b, rec, thr)
                    msg = (f"{classes[kk]} ({c:.0%})" if c >= thr
                           else f"Tidak yakin ({classes[kk]} {c:.0%})")
                    msg_until = time.time() + 4.0
                    print("Hasil:", msg)
                rec = []
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
