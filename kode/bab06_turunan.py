"""Bab 6: log-loss, gradien, dan Hessian dalam bentuk matriks.

Fungsi log_loss, gradien, dan hessian dicetak di naskah (Listing
6.1). Parameter theta = (b, w_1, ..., w_d), dan matriks rancangan X
memuat kolom satu untuk intersep. Bagian utama menghitung gradien
dan Hessian di theta = 0 (yang juga dapat dihitung dengan tangan),
mencocokkannya dengan beda hingga, lalu memeriksa keduanya di MLE.
"""
import numpy as np
from scipy.special import expit

from bab04_data import jam_belajar
from bab04_model import taksir


def rancang(x):
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    return np.column_stack([np.ones(len(x)), x])


def log_loss(theta, X, y):
    z = X @ theta
    return np.mean(np.logaddexp(0, z) - y * z)


def gradien(theta, X, y):
    p = expit(X @ theta)
    return X.T @ (p - y) / len(y)


def hessian(theta, X, y):
    p = expit(X @ theta)
    d = p * (1 - p)
    return (X.T * d) @ X / len(y)


def gradien_beda(theta, X, y, h=1e-6):
    E = np.eye(len(theta))
    return np.array([(log_loss(theta + h * e, X, y)
                      - log_loss(theta - h * e, X, y)) / (2 * h)
                     for e in E])


def hessian_beda(theta, X, y, h=1e-5):
    E = np.eye(len(theta))
    return np.array([(gradien(theta + h * e, X, y)
                      - gradien(theta - h * e, X, y)) / (2 * h)
                     for e in E])


if __name__ == "__main__":
    x, y = jam_belajar()
    X = rancang(x)
    nol = np.zeros(2)
    print(f"sum x = {x.sum():.0f}, sum x^2 = {(x ** 2).sum():.0f}, "
          f"sum x (y = 1) = {x[y == 1].sum():.0f}")
    g = gradien(nol, X, y)
    print(f"di theta = 0: gradien = ({g[0]:.6f}, {g[1]:.6f})")
    H = hessian(nol, X, y)
    print(f"              Hessian = [[{H[0, 0]:.6f}, {H[0, 1]:.6f}],")
    print(f"                         [{H[1, 0]:.6f}, {H[1, 1]:.6f}]]")
    gb, Hb = gradien_beda(nol, X, y), hessian_beda(nol, X, y)
    print(f"beda hingga: maks galat gradien "
          f"{np.max(np.abs(g - gb)):.1e}, Hessian "
          f"{np.max(np.abs(H - Hb)):.1e}")
    b, w = taksir()
    t = np.array([b, w])
    g = gradien(t, X, y)
    H = hessian(t, X, y)
    kecil = np.linalg.norm(g) < 1e-12
    print(f"di MLE: norma gradien < 1e-12: {kecil}")
    print(f"        Hessian = [[{H[0, 0]:.6f}, {H[0, 1]:.6f}],")
    print(f"                   [{H[1, 0]:.6f}, {H[1, 1]:.6f}]]")
    print(f"        nilai eigen = {np.linalg.eigvalsh(H).round(6)}")
