"""Bab 4: gradient descent dan SGD pada data mini dua fitur.

Fungsi gd dan sgd di sini dipakai lagi di bab-bab berikutnya.
"""
import numpy as np
from scipy.special import expit

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien


def gd(X, y, eta, langkah, w0=None):
    """Gradient descent dengan laju belajar tetap; mengembalikan jejak."""
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    jejak = [w.copy()]
    for _ in range(langkah):
        w = w - eta * gradien(w, X, y)
        jejak.append(w.copy())
    return np.array(jejak)


def sgd(X, y, eta, epoch, benih=BENIH, c=10):
    """SGD satu sampel per langkah; laju eta / (1 + k / (c m))."""
    rng = np.random.default_rng(benih)
    m = len(y)
    w = np.zeros(X.shape[1])
    jejak, k = [w.copy()], 0
    for _ in range(epoch):
        for i in rng.permutation(m):
            p = expit(X[i] @ w)
            w = w - eta / (1 + k / (c * m)) * (p - y[i]) * X[i]
            k += 1
        jejak.append(w.copy())
    return np.array(jejak)


def langkah_sampai(jejak, X, y, tol):
    for k, w in enumerate(jejak):
        if np.abs(gradien(w, X, y)).max() < tol:
            return k
    return None


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    L = np.linalg.eigvalsh(X.T @ X / (4 * len(y))).max()
    print(f"(1) L = lambda_maks(X^T X)/(4m) = {L:.4f},"
          f" 1/L = {1 / L:.4f}")
    j = gd(X, y, 0.25, 3)
    print("    GD, eta = 0.25, tiga langkah pertama:")
    print("     k      w0        w1        w2        L")
    for k, w in enumerate(j):
        w = np.round(w, 4) + 0.0
        print(f"    {k:2d}  {w[0]:8.4f}  {w[1]:8.4f}  {w[2]:8.4f}"
              f"   {log_loss(j[k], X, y):.4f}")

    Xc = rancang(Xf - Xf.mean(0))
    Xs = rancang((Xf - Xf.mean(0)) / Xf.std(0))
    print("(2) langkah sampai max|g| < 1e-6, eta = 1/L:")
    for nama, A in [("x mentah   ", X), ("x dipusat  ", Xc),
                    ("x dibakukan", Xs)]:
        La = np.linalg.eigvalsh(A.T @ A / 24).max()
        ev = np.linalg.eigvalsh(A.T @ A / 24)
        jj = gd(A, y, 1 / La, 60000)
        print(f"    {nama}: L = {La:.4f}, kondisi = {ev[-1] / ev[0]:7.1f},"
              f" langkah {langkah_sampai(jj, A, y, 1e-6)}")

    print("(3) SGD (urutan 1, 2, 3, eta = 0.5 tetap):")
    w = np.zeros(3)
    for i in range(3):
        p = expit(X[i] @ w)
        w = w - 0.5 * (p - y[i]) * X[i]
        print(f"    i = {i + 1}: p = {p:.4f}, w = ({w[0]:.4f},"
              f" {w[1]:.4f}, {w[2]:.4f})")
    Ls = log_loss(mle_mini(), X, y)
    j = sgd(Xs, y, 1.0, 4000)
    print("    SGD acak, x dibakukan, eta_k = 1/(1 + k/(10m)):")
    for e in [10, 100, 1000, 4000]:
        print(f"      epoch {e:4d}: L - L* = "
              f"{log_loss(j[e], Xs, y) - Ls:.1e}")
