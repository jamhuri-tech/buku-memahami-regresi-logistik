"""Bab 17: populasi dengan kejadian langka dan parameter diketahui.

populasi(n): lima peubah normal baku, w = (1, -0.8, 0.6, 0.5, 0),
b = -4.6; kira-kira dua persen kelas 1.
"""
import numpy as np
from scipy.special import expit

from bab04_data import BENIH

W_LANGKA = np.array([1.0, -0.8, 0.6, 0.5, 0.0])
B_LANGKA = -4.6


def populasi(n=200_000, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(n, 5))
    p = expit(X @ W_LANGKA + B_LANGKA)
    return X, (rng.random(n) < p).astype(int), p


if __name__ == "__main__":
    X, y, p = populasi()
    print(f"populasi n = {len(y)}, kelas 1 = {y.sum()} "
          f"({y.mean():.4f})")
