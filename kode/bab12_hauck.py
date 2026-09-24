"""Bab 12: efek Hauck-Donner.

Untuk rancangan tetap x = 40 titik di [-2, 2] dan data "harapan"
y_i = sigma(w x_i), MLE-nya tepat w (persamaan skor terpenuhi).
Statistik Wald z = w / SE(w) dan statistik rasio kemungkinan G untuk
uji w = 0 dihitung untuk w yang makin besar.
"""
import numpy as np
from scipy.special import expit

from bab06_turunan import rancang

X = rancang(np.linspace(-2, 2, 40))


def statistik(w):
    p = expit(w * X[:, 1])
    H = (X.T * (p * (1 - p))) @ X
    se = np.sqrt(np.linalg.inv(H)[1, 1])
    l1 = np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
    l0 = len(p) * np.log(0.5)
    return w / se, 2 * (l1 - l0)


if __name__ == "__main__":
    print("   w    Wald z^2   G (rasio kemungkinan)")
    for w in (0.5, 1, 2, 4, 6, 8, 10):
        z, G = statistik(w)
        print(f"{w:5.1f}  {z * z:9.2f}   {G:9.2f}")
    ws = np.linspace(0.1, 12, 1000)
    z2 = np.array([statistik(w)[0] ** 2 for w in ws])
    print(f"Wald z^2 terbesar {z2.max():.2f} di w = {ws[np.argmax(z2)]:.2f}")
