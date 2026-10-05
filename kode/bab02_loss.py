"""Bab 2: kemungkinan, log-kemungkinan, dan log-loss.

Fungsi log_loss di sini dipakai lagi di bab-bab berikutnya.
"""
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss as sk_log_loss

from bab01_data import data_mini, mle_mini, rancang


def loss_titik(z, y):
    """log(1 + e^z) - y z, dihitung tanpa luapan."""
    return np.logaddexp(0, z) - y * z


def log_loss(theta, X, y):
    """Log-loss rata-rata L(theta) = -ell(theta) / n."""
    return np.mean(loss_titik(X @ theta, y))


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    print("(1) loss per titik dan log-loss rata-rata L:")
    for nama, th in [("theta = (0, 0)", np.zeros(2)),
                     ("theta = (-2.8, 0.8)", np.array([-2.8, 0.8])),
                     ("theta = MLE", mle_mini())]:
        lt = loss_titik(X @ th, y)
        print(f"    {nama:<20} L = {lt.mean():.4f}")
        print("      " + " ".join(f"{v:.4f}" for v in lt))
    p = expit(X @ np.array([-2.8, 0.8]))
    print(f"    sklearn log_loss di (-2.8, 0.8): {sk_log_loss(y, p):.4f}")
    print(f"    log-kemungkinan di (-2.8, 0.8):  "
          f"{-6 * log_loss(np.array([-2.8, 0.8]), X, y):.4f}")

    print("(2) menghitung untuk skor besar (y = 0):")
    with np.errstate(over="ignore", divide="ignore"):
        for z in [37.0, 800.0]:
            naif = -np.log(1 - 1 / (1 + np.exp(-z)))
            print(f"    z = {z:5.0f}: rumus naif {naif:8.4f}"
                  f"   logaddexp {loss_titik(z, 0):8.4f}")
