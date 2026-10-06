"""Bab 3: gradien dan Hessian log-loss, diperiksa dengan beda hingga
dan dengan statsmodels.

Fungsi gradien dan hessian di sini dipakai lagi di bab-bab berikutnya.
Keduanya untuk log-loss RATA-RATA L = -ell / m.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss


def gradien(w, X, y):
    """g = X^T (p - y) / m."""
    p = expit(X @ w)
    return X.T @ (p - y) / len(y)


def hessian(w, X, y):
    """H = X^T D X / m, D = diag(p (1 - p))."""
    p = expit(X @ w)
    d = p * (1 - p)
    return X.T @ (X * d[:, None]) / len(y)


def beda_hingga(f, w, h=1e-6):
    e = np.eye(len(w))
    return np.array([(f(w + h * e[j]) - f(w - h * e[j])) / (2 * h)
                     for j in range(len(w))])


def cetak_matriks(H, awal="      H = "):
    for k, baris in enumerate(H):
        kepala = awal if k == 0 else " " * len(awal)
        print(kepala + " ".join(f"{v:8.4f}" for v in baris))


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    print("(1) gradien dan Hessian (rata-rata):")
    for nama, w in [("w = (0, 0, 0)", np.zeros(3)),
                    ("w = (-2, 1, -1)", np.array([-2.0, 1.0, -1.0])),
                    ("w = MLE", mle_mini())]:
        g = np.round(gradien(w, X, y), 4) + 0.0
        H = hessian(w, X, y)
        print(f"    {nama}")
        print("      g = " + " ".join(f"{v:8.4f}" for v in g))
        cetak_matriks(H)
        ev = np.linalg.eigvalsh(H)
        print("      nilai eigen: " + ", ".join(f"{v:.4f}" for v in ev))

    w = np.array([-2.0, 1.0, -1.0])
    gb = beda_hingga(lambda t: log_loss(t, X, y), w)
    Hb = np.array([beda_hingga(lambda t: gradien(t, X, y)[j], w)
                   for j in range(3)])
    print("(2) beda hingga di (-2, 1, -1):")
    print(f"    selisih gradien < 1e-8: "
          f"{np.abs(gb - gradien(w, X, y)).max() < 1e-8}")
    print(f"    selisih Hessian < 1e-8: "
          f"{np.abs(Hb - hessian(w, X, y)).max() < 1e-8}")
    m = sm.Logit(y, X)
    print("    statsmodels: score = -m g dan hessian = -m H:",
          np.allclose(m.score(w), -6 * gradien(w, X, y)),
          np.allclose(m.hessian(w), -6 * hessian(w, X, y)))

    p = expit(X @ mle_mini())
    print("(3) persamaan skor di MLE:")
    print("    X^T p =", np.round(X.T @ p, 6) + 0.0)
    print("    X^T y =", (X.T @ y).astype(float))
