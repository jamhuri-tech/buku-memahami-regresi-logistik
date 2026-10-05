"""Bab 6: pemisahan sempurna, penalti L2 dan L1, dan koreksi Firth.

Fungsi newton_l2, ista_l1, dan firth di sini dipakai lagi di bab-bab
berikutnya. Intersep tidak pernah dipenalti.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian

Y_PISAH = np.array([0, 0, 0, 1, 1, 1])   # data mini, label 3 dan 4 ditukar


def newton_l2(X, y, lam, langkah=50):
    """Minimum L(theta) + lam/2 |w|^2 dengan metode Newton."""
    P = np.eye(X.shape[1])
    P[0, 0] = 0.0
    theta = np.zeros(X.shape[1])
    for _ in range(langkah):
        g = gradien(theta, X, y) + lam * P @ theta
        H = hessian(theta, X, y) + lam * P
        theta = theta - np.linalg.solve(H, g)
    return theta


def lunak(u, a):
    """Soft-thresholding: sign(u) max(|u| - a, 0)."""
    return np.sign(u) * np.maximum(np.abs(u) - a, 0.0)


def ista_l1(X, y, lam, eta, langkah):
    """Proximal gradient untuk L(theta) + lam |w|_1."""
    theta = np.zeros(X.shape[1])
    for _ in range(langkah):
        u = theta - eta * gradien(theta, X, y)
        theta = np.r_[u[0], lunak(u[1:], eta * lam)]
    return theta


def firth(X, y, langkah=100):
    """Taksiran Firth: skor termodifikasi X^T(y - p + h(1/2 - p)) = 0."""
    theta = np.zeros(X.shape[1])
    for _ in range(langkah):
        p = expit(X @ theta)
        d = p * (1 - p)
        A = X.T @ (X * d[:, None])
        h = d * np.einsum("ij,jk,ik->i", X, np.linalg.inv(A), X)
        U = X.T @ (y - p + h * (0.5 - p))
        theta = theta + np.linalg.solve(A, U)
    return theta, h


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    v = np.array([-3.5, 1.0])
    print("(1) data terpisah y = (0,0,0,1,1,1), L(t v), v = (-3.5, 1):")
    for t in [1, 2, 5, 10, 20]:
        print(f"    t = {t:2d}: L = {log_loss(t * v, X, Y_PISAH):.6f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jalur = []
        theta = np.zeros(2)
        for k in range(1, 26):
            theta = theta - np.linalg.solve(hessian(theta, X, Y_PISAH),
                                            gradien(theta, X, Y_PISAH))
            if k in (5, 10, 15):
                jalur.append((k, theta[1]))
    for k, w in jalur:
        print(f"    Newton langkah {k:2d}: w = {w:.2f}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = sm.Logit(Y_PISAH, X).fit(disp=0)
    jenis = sorted({c.category.__name__ for c in w})
    print("    peringatan statsmodels:")
    for j in jenis:
        print("      " + j)
    print(f"    statsmodels |w| > 10: {abs(r.params[1]) > 10}")

    print("(2) penalti L2, lam = 1/(nC), dibandingkan dengan sklearn:")
    for C in [10.0, 1.0, 0.1]:
        lam = 1 / (6 * C)
        for nama, yy in [("mini", y), ("terpisah", Y_PISAH)]:
            th = newton_l2(X, yy, lam)
            m = LogisticRegression(C=C, tol=1e-12).fit(x[:, None], yy)
            cocok = np.allclose(th, np.r_[m.intercept_, m.coef_[0]],
                                atol=1e-7)
            print(f"    C = {C:4.1f} {nama:<9} b = {th[0]:7.4f},"
                  f" w = {th[1]:.4f}, sama: {cocok}")

    g0 = gradien(np.zeros(2), X, y)
    print("(3) penalti L1:")
    print(f"    lam_maks = |g_w(model nol)| = {abs(g0[1]):.4f}")
    print(f"    C_min = 1/(n lam_maks) = {1 / (6 * abs(g0[1])):.4f}")
    for lam in [0.7, 0.5, 0.1]:
        th = ista_l1(X, y, lam, 0.25, 20000)
        m = LogisticRegression(penalty="l1", C=1 / (6 * lam),
                               solver="saga", tol=1e-12, max_iter=100000,
                               random_state=0).fit(x[:, None], y)
        cocok = abs(th[1] - m.coef_[0, 0]) < 1e-4
        print(f"    lam = {lam}: b = {th[0]:7.4f}, w = {th[1]:.4f},"
              f" saga sama: {cocok}")

    th, h = firth(X, Y_PISAH)
    p = expit(X @ th)
    U = X.T @ (Y_PISAH - p + h * (0.5 - p))
    print("(4) Firth pada data terpisah:")
    print(f"    b = {th[0]:.4f}, w = {th[1]:.4f}, x50 = {-th[0] / th[1]:.4f}")
    print(f"    skor termodifikasi nol: {np.abs(U).max() < 1e-10}")
    print("    peluang:", np.round(p, 4))
