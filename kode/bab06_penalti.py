"""Bab 6: pemisahan sempurna, penalti L2 dan L1, dan koreksi Firth.

Fungsi newton_l2, ista_l1, dan firth di sini dipakai lagi di bab-bab
berikutnya. Bobot bias w0 tidak pernah dipenalti.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian

Y_PISAH = np.array([0, 0, 0, 1, 1, 1])   # label mahasiswa 3 dan 6 ditukar


def newton_l2(X, y, lam, langkah=50):
    """Minimum L(w) + lam/2 (w1^2 + ... + wn^2) dengan metode Newton."""
    P = np.eye(X.shape[1])
    P[0, 0] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(langkah):
        g = gradien(w, X, y) + lam * P @ w
        H = hessian(w, X, y) + lam * P
        w = w - np.linalg.solve(H, g)
    return w


def lunak(u, a):
    """Soft-thresholding: sign(u) max(|u| - a, 0)."""
    return np.sign(u) * np.maximum(np.abs(u) - a, 0.0)


def ista_l1(X, y, lam, eta, langkah):
    """Proximal gradient untuk L(w) + lam (|w1| + ... + |wn|)."""
    w = np.zeros(X.shape[1])
    for _ in range(langkah):
        u = w - eta * gradien(w, X, y)
        w = np.r_[u[0], lunak(u[1:], eta * lam)]
    return w


def firth(X, y, langkah=200):
    """Taksiran Firth: skor termodifikasi X^T(y - p + h(1/2 - p)) = 0.

    Langkah Newton termodifikasi dipendekkan bila norma skor naik.
    """
    def skor(w):
        p = expit(X @ w)
        d = p * (1 - p)
        A = X.T @ (X * d[:, None])
        h = d * np.einsum("ij,jk,ik->i", X, np.linalg.inv(A), X)
        return X.T @ (y - p + h * (0.5 - p)), A, h

    w = np.zeros(X.shape[1])
    for _ in range(langkah):
        U, A, h = skor(w)
        if np.abs(U).max() < 1e-12:
            break
        s, t = np.linalg.solve(A, U), 1.0
        while np.linalg.norm(skor(w + t * s)[0]) > np.linalg.norm(U) \
                and t > 1e-6:
            t /= 2
        w = w + t * s
    return w, skor(w)[2]


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    v = np.array([-3.5, 1.0, 0.0])
    print("(1) data terpisah y = (0,0,0,1,1,1), L(t v), v = (-3.5, 1, 0):")
    for t in [1, 2, 5, 10, 20]:
        print(f"    t = {t:2d}: L = {log_loss(t * v, X, Y_PISAH):.6f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = np.zeros(3)
        for k in range(1, 16):
            w = w - np.linalg.solve(hessian(w, X, Y_PISAH),
                                    gradien(w, X, Y_PISAH))
            if k in (5, 10, 15):
                print(f"    Newton langkah {k:2d}: |w| = "
                      f"{np.linalg.norm(w):.1f}")
    with warnings.catch_warnings(record=True) as cat:
        warnings.simplefilter("always")
        r = sm.Logit(Y_PISAH, X).fit(disp=0)
    print("    peringatan statsmodels:")
    for nama in sorted({c.category.__name__ for c in cat}):
        print("      " + nama)

    print("(2) penalti L2, lam = 1/(mC), dibandingkan dengan sklearn:")
    for C in [10.0, 1.0, 0.1]:
        lam = 1 / (6 * C)
        for nama, yy in [("mini", y), ("terpisah", Y_PISAH)]:
            w = newton_l2(X, yy, lam)
            m = LogisticRegression(C=C, tol=1e-12).fit(Xf, yy)
            sama = np.allclose(w, np.r_[m.intercept_, m.coef_[0]],
                               atol=1e-7)
            print(f"    C = {C:4.1f} {nama:<8} w = ({w[0]:7.4f},"
                  f" {w[1]:.4f}, {w[2]:7.4f}) {sama}")

    g0 = gradien(np.zeros(3), X, y)
    print("(3) penalti L1:")
    print("    g(model nol) =", np.round(g0, 4) + 0.0)
    print(f"    lam_maks = {np.abs(g0[1:]).max():.4f}")
    for lam in [0.3, 0.2, 0.1, 0.02]:
        w = ista_l1(X, y, lam, 0.2, 30000)
        m = LogisticRegression(penalty="l1", C=1 / (6 * lam),
                               solver="saga", tol=1e-12, max_iter=100000,
                               random_state=0).fit(Xf, y)
        sama = np.abs(w[1:] - m.coef_[0]).max() < 1e-4
        w = np.round(w, 4) + 0.0
        print(f"    lam = {lam:4.2f}: w = ({w[0]:7.4f}, {w[1]:.4f},"
              f" {w[2]:7.4f}) {sama}")

    w, h = firth(X, Y_PISAH)
    p = expit(X @ w)
    U = X.T @ (Y_PISAH - p + h * (0.5 - p))
    print("(4) Firth pada data terpisah:")
    print(f"    w = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})")
    print(f"    skor termodifikasi nol: {np.abs(U).max() < 1e-10}")
    print("    peluang:", np.round(p, 4))
