"""Bab 11: penalti L1 dari nol.

(1) Batas C: semua bobot nol tepat ketika
    C <= 1 / max_j |sum_i (y_i - ybar) x_ij|.
(2) ista (Listing 11.1): proximal gradient untuk ||w||_1 + C sum loss,
    dengan intersep tak dipenalti, dicocokkan dengan saga.
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab06_turunan import rancang
from bab11_data import data_jarang


def lunak(v, a):
    """Operator soft-thresholding: prox dari a ||v||_1."""
    return np.sign(v) * np.maximum(np.abs(v) - a, 0.0)


def ista(X, y, C, langkah):
    Xt = rancang(X)
    eta = 4 / (C * np.linalg.eigvalsh(Xt.T @ Xt).max())
    theta = np.zeros(Xt.shape[1])
    for _ in range(langkah):
        g = C * Xt.T @ (expit(Xt @ theta) - y)
        theta = theta - eta * g
        theta[1:] = lunak(theta[1:], eta)      # intersep tak disentuh
    return theta


def saga_l1(X, y, C):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(penalty="l1", C=C, solver="saga",
                                  tol=1e-12, max_iter=100_000,
                                  random_state=0).fit(X, y)


if __name__ == "__main__":
    X, y, _, _ = data_jarang()
    korelasi = np.abs(X.T @ (y - y.mean()))
    C_nol = 1 / korelasi.max()
    j = int(np.argmax(korelasi))
    print(f"(1) C_nol = 1 / max_j |sum (y - ybar) x_j| = {C_nol:.5f}")
    print(f"    dicapai oleh peubah ke-{j + 1}")
    for faktor in (0.99, 1.01, 1.2):
        m = saga_l1(X, y, faktor * C_nol)
        nz = np.flatnonzero(np.abs(m.coef_[0]) > 1e-10) + 1
        print(f"    C = {faktor:4.2f} C_nol: bobot bukan nol "
              f"{[int(v) for v in nz]}")
    C = 0.3
    m = saga_l1(X, y, C)
    t_sk = np.r_[m.intercept_, m.coef_[0]]
    for langkah in (30, 100, 1000):
        t = ista(X, y, C, langkah)
        beda = np.abs(t - t_sk).max()
        teks = f"{beda:.1e}" if beda > 1e-8 else "< 1e-8"
        print(f"(2) ISTA {langkah:4d} langkah: maks |beda dengan saga| "
              f"{teks}, bukan nol {int((np.abs(t[1:]) > 0).sum())}")
    print(f"    saga: bukan nol {int((np.abs(m.coef_) > 0).sum())}")
