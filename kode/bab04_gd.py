"""Bab 4: gradient descent dan SGD pada data mini.

Fungsi gd dan sgd di sini dipakai lagi di bab-bab berikutnya.
"""
import numpy as np
from scipy.special import expit

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien


def gd(X, y, eta, langkah, theta0=None):
    """Gradient descent dengan laju belajar tetap; mengembalikan jejak."""
    theta = np.zeros(X.shape[1]) if theta0 is None else theta0.copy()
    jejak = [theta.copy()]
    for _ in range(langkah):
        theta = theta - eta * gradien(theta, X, y)
        jejak.append(theta.copy())
    return np.array(jejak)


def sgd(X, y, eta, epoch, benih=BENIH, acak=True):
    """SGD satu titik per langkah; laju eta / (1 + k / n)."""
    rng = np.random.default_rng(benih)
    n = len(y)
    theta = np.zeros(X.shape[1])
    jejak, k = [theta.copy()], 0
    for _ in range(epoch):
        urutan = rng.permutation(n) if acak else np.arange(n)
        for i in urutan:
            p = expit(X[i] @ theta)
            theta = theta - eta / (1 + k / n) * (p - y[i]) * X[i]
            k += 1
        jejak.append(theta.copy())
    return np.array(jejak)


def langkah_sampai(jejak, X, y, tol):
    """Langkah pertama dengan |gradien| di bawah tol."""
    for k, th in enumerate(jejak):
        if np.abs(gradien(th, X, y)).max() < tol:
            return k
    return None


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    Lmax = np.linalg.eigvalsh(X.T @ X / (4 * len(y))).max()
    print(f"(1) L = lambda_maks(X^T X)/(4n) = {Lmax:.4f},"
          f" 1/L = {1 / Lmax:.4f}")
    j = gd(X, y, 0.25, 3)
    print("    GD, eta = 0.25, tiga langkah pertama:")
    print("     k        b         w        L")
    for k, th in enumerate(j):
        print(f"    {k:2d}  {th[0]:8.4f}  {th[1]:8.4f}"
              f"   {log_loss(th, X, y):.4f}")

    m = mle_mini()
    Lstar = log_loss(m, X, y)
    xc = x - 3.5
    Xc = rancang(xc)
    Lc = np.linalg.eigvalsh(Xc.T @ Xc / (4 * len(y))).max()
    print("(2) langkah sampai max|g| < 1e-6:")
    for nama, A, eta in [("x mentah,   eta = 1/L", X, 1 / Lmax),
                         ("x dipusat,  eta = 1/L", Xc, 1 / Lc)]:
        jj = gd(A, y, eta, 20000)
        print(f"    {nama}: {langkah_sampai(jj, A, y, 1e-6)}")
    jj = gd(Xc, y, 1 / Lc, 20000)
    th = np.round(jj[langkah_sampai(jj, Xc, y, 1e-6)], 6) + 0.0
    print(f"    dipusat: b_c = {th[0]:.4f}, w = {th[1]:.4f},"
          f" b = {th[0] - 3.5 * th[1]:.4f}")
    print(f"    L dipusat = {Lc:.4f}; MLE b = {m[0]:.4f}, w = {m[1]:.4f}")

    print("(3) SGD (urutan 1..6, eta = 0.5 tetap), tiga pembaruan:")
    th = np.zeros(2)
    for i in range(3):
        p = expit(X[i] @ th)
        th = th - 0.5 * (p - y[i]) * X[i]
        print(f"    i = {i + 1}: p = {p:.4f}, theta = ({th[0]:.4f},"
              f" {th[1]:.4f})")
    j = sgd(Xc, y, 0.5, 2000)
    print("    SGD acak pada x dipusat, eta_k = 0.5/(1 + k/n):")
    for e in [10, 100, 1000, 2000]:
        print(f"      epoch {e:4d}: L - L* = "
              f"{log_loss(j[e], Xc, y) - Lstar:.1e}")
