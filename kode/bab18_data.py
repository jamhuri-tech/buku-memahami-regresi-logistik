"""Bab 18: tiga kelas dari model softmax dengan parameter diketahui.

Skor kelas k adalah b_k + w_k . x, dengan kelas 0 sebagai rujukan
(b_0 = 0, w_0 = 0), x normal baku di bidang.
"""
import numpy as np
from scipy.special import softmax

from bab04_data import BENIH

B_BENAR = np.array([0.0, 0.5, -0.5])
W_BENAR = np.array([[0.0, 0.0],
                    [2.0, 0.5],
                    [-0.5, 2.0]])


def tiga_kelas(n=600, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(n, 2))
    P = softmax(X @ W_BENAR.T + B_BENAR, axis=1)
    u = rng.random(n)[:, None]
    y = (u > np.cumsum(P, axis=1)).sum(axis=1)
    return X, y, P


if __name__ == "__main__":
    X, y, _ = tiga_kelas()
    print(f"n = {len(y)}, banyak per kelas: {np.bincount(y)}")
