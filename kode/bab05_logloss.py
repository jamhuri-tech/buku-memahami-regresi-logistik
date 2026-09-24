"""Bab 5: log-loss dari skor, dicocokkan dengan scikit-learn.

log_loss_kita (Listing 5.1) menghitung log-loss rata-rata langsung
dari skor z memakai log(1 + e^z) - y z, tanpa pernah membentuk
peluang. Bagian utama mencocokkannya dengan sklearn.metrics.log_loss
dan statsmodels, lalu menguji skor ekstrem.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import log_loss

from bab04_data import jam_belajar


def log_loss_kita(y, z):
    return np.mean(np.logaddexp(0, z) - y * z)


def log_loss_polos(y, z):
    p = 1 / (1 + np.exp(-z))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    x, y = jam_belajar()
    hasil = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    b, w = hasil.params
    z = b + w * x
    n = len(y)
    print(f"log_loss_kita         = {log_loss_kita(y, z):.6f}")
    print(f"sklearn log_loss      = {log_loss(y, expit(z)):.6f}")
    print(f"-llf / n (statsmodels)= {-hasil.llf / n:.6f}")
    ybar = y.mean()
    ent = -(ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))
    print(f"baseline (p = rata-rata y = {ybar:.4f}): "
          f"log-loss = {ent:.6f}")
    print(f"log 2 = {np.log(2):.6f}, -log(eps) = "
          f"{-np.log(np.finfo(float).eps):.4f}")
    print("skor ekstrem, label 0 dan skor z:")
    with np.errstate(all="ignore"):
        for zz in (10.0, 40.0, 800.0):
            y0, z0 = np.array([0]), np.array([zz])
            sk = log_loss(y0, expit(z0), labels=[0, 1])
            print(f"  z = {zz:4.0f}: kita {log_loss_kita(y0, z0):8.4f}"
                  f"  polos {log_loss_polos(y0, z0):8.4f}"
                  f"  sklearn {sk:7.4f}")
