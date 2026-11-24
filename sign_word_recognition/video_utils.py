"""Ekstraksi frame & uniform frame sampling."""
import cv2
import numpy as np


def letterbox(img_rgb, size):
    """Resize dengan menjaga rasio aspek, sisanya padding hitam -> (size,size,3) uint8."""
    h, w = img_rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def count_frames(path):
    """Hitung frame sebenarnya (CAP_PROP_FRAME_COUNT sering tidak akurat pada MOV)."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    n = count_frames(path)
    return {"frames": n, "fps": fps, "width": w, "height": h,
            "duration": (n / fps) if fps else 0.0}


def uniform_indices(total, n):
    """n indeks yang tersebar merata dari frame pertama sampai terakhir."""
    return np.linspace(0, total - 1, n).round().astype(int)


def read_frames_at(path, indices):
    wanted = {}
    for pos, i in enumerate(indices):
        wanted.setdefault(int(i), []).append(pos)
    out = [None] * len(indices)
    cap = cv2.VideoCapture(str(path))
    idx, last = 0, max(wanted)
    while idx <= last and cap.grab():
        if idx in wanted:
            ok, f = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                for pos in wanted[idx]:
                    out[pos] = rgb
        idx += 1
    cap.release()
    valid = [f for f in out if f is not None]
    if not valid:
        raise RuntimeError(f"Tidak ada frame terbaca: {path}")
    prev = valid[0]
    for k in range(len(out)):
        if out[k] is None:
            out[k] = prev
        prev = out[k]
    return out


def sample_video(path, n_frames, size):
    """Video -> n_frames frame (uniform sampling), sudah di-letterbox ke size x size (RGB uint8)."""
    total = count_frames(path)
    if total == 0:
        raise RuntimeError(f"Video kosong/tidak terbaca: {path}")
    idx = uniform_indices(total, n_frames)
    frames = read_frames_at(path, idx)
    return [letterbox(f, size) for f in frames], idx


def sample_list(frames, n):
    """Uniform sampling dari list frame (dipakai webcam buffer)."""
    idx = uniform_indices(len(frames), n)
    return [frames[i] for i in idx]
