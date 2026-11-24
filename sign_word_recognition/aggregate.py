"""Agregasi prediksi per-frame -> skor video. Input P: (n_frame, n_kelas) probabilitas softmax.
Output: vektor skor (jumlah=1); prediksi = argmax, confidence = max."""
import numpy as np


def _norm(s):
    return s / max(s.sum(), 1e-12)


def average_probability(P):
    return _norm(P.mean(axis=0))


def majority_voting(P):
    votes = np.bincount(P.argmax(1), minlength=P.shape[1]).astype(float) / len(P)
    return _norm(votes + 1e-3 * P.mean(axis=0))  # tie-break: rata-rata probabilitas


def confidence_weighted(P):
    w = P.max(axis=1)  # frame yang yakin diberi bobot lebih
    return _norm((P * w[:, None]).sum(axis=0))


def position_weighted(P, sigma=0.35):
    # bobot Gaussian: bagian tengah gerakan lebih berbobot dibanding awal/akhir (frame diam).
    # Hanya kandidat; dipakai bila terbukti lebih stabil di validasi.
    pos = np.linspace(0, 1, len(P))
    w = np.exp(-0.5 * ((pos - 0.5) / sigma) ** 2)
    return _norm((P * w[:, None]).sum(axis=0))


AGGREGATORS = {
    "average_probability": average_probability,
    "majority_voting": majority_voting,
    "confidence_weighted": confidence_weighted,
    "position_weighted": position_weighted,
}
