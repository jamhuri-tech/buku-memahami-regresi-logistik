"""Bab 2: kemungkinan, log-kemungkinan, dan log-loss.

Fungsi loss_titik dan log_loss di sini dipakai lagi di bab-bab
berikutnya. Bobot w = (w0, w1, ..., wn); X memuat kolom x0 = 1.
"""
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss as sk_log_loss

from bab01_data import data_mini, mle_mini, rancang


def loss_titik(z, y):
    """log(1 + e^z) - y z, dihitung tanpa luapan."""
    return np.logaddexp(0, z) - y * z


def log_loss(w, X, y):
    """Log-loss rata-rata L(w) = -ell(w) / m."""
    return np.mean(loss_titik(X @ w, y))


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    print("(1) loss per sampel dan log-loss rata-rata L:")
    for nama, w in [("w = (0, 0, 0)", np.zeros(3)),
                    ("w = (-2, 1, -1)", np.array([-2.0, 1.0, -1.0])),
                    ("w = MLE", mle_mini())]:
        lt = loss_titik(X @ w, y)
        print(f"    {nama:<17} L = {lt.mean():.4f}")
        print("      " + " ".join(f"{v:.4f}" for v in lt))
    w = np.array([-2.0, 1.0, -1.0])
    print(f"    sklearn log_loss di (-2, 1, -1): "
          f"{sk_log_loss(y, expit(X @ w)):.4f}")
    print(f"    log-kemungkinan di (-2, 1, -1):  "
          f"{-6 * log_loss(w, X, y):.4f}")
    print(f"    log-kemungkinan di MLE:          "
          f"{-6 * log_loss(mle_mini(), X, y):.4f}")
    print(f"    3 ln 3 - 10 ln 2 =               "
          f"{3 * np.log(3) - 10 * np.log(2):.4f}")

    print("(2) menghitung untuk skor besar (y = 0):")
    with np.errstate(over="ignore", divide="ignore"):
        for z in [37.0, 800.0]:
            naif = -np.log(1 - 1 / (1 + np.exp(-z)))
            print(f"    z = {z:5.0f}: rumus naif {naif:8.4f}"
                  f"   logaddexp {loss_titik(z, 0):8.4f}")
