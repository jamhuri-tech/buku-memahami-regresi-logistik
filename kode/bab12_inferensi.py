"""Bab 12: galat baku, uji, dan selang kepercayaan dari nol.

mle (Newton) mengembalikan taksiran dan Hessian jumlah log-loss;
galat baku adalah akar diagonal invers Hessian itu (Listing 12.1).
selang_profil (Listing 12.2) mencari batas selang kemungkinan profil
dengan brentq; parameter lain dioptimumkan ulang dengan BFGS.
firth (Listing 12.3) menyelesaikan persamaan skor yang dimodifikasi.
"""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import chi2

KRITIS = chi2.ppf(0.95, df=1)          # 3.8415 = 1.96^2


def log_kem(theta, X, y):
    z = X @ theta
    return np.sum(y * z - np.logaddexp(0, z))


def mle(X, y, tol=1e-10, maks=50):
    theta = np.zeros(X.shape[1])
    for _ in range(maks):
        p = expit(X @ theta)
        H = (X.T * (p * (1 - p))) @ X
        d = np.linalg.solve(H, X.T @ (p - y))
        theta = theta - d
        if np.max(np.abs(d)) < tol:
            break
    return theta, H


def galat_baku(H):
    return np.sqrt(np.diag(np.linalg.inv(H)))


def profil(X, y, j, v, awal):
    """Log-kemungkinan terbesar bila theta_j = v."""
    lain = [i for i in range(X.shape[1]) if i != j]
    geser = X[:, j] * v
    A = X[:, lain]

    def f(u):
        z = A @ u + geser
        return -np.sum(y * z - np.logaddexp(0, z))

    def g(u):
        return -A.T @ (y - expit(A @ u + geser))

    r = minimize(f, awal[lain], jac=g, method="BFGS",
                 options={"gtol": 1e-9})
    return -r.fun


def selang_profil(X, y, j, theta, se):
    l_maks = log_kem(theta, X, y)

    def h(v):
        turun = l_maks - profil(X, y, j, v, theta)
        return 2 * turun - KRITIS

    bawah = brentq(h, theta[j] - 10 * se, theta[j])
    atas = brentq(h, theta[j], theta[j] + 10 * se)
    return bawah, atas


def firth(X, y, tol=1e-10, maks=100):
    theta = np.zeros(X.shape[1])
    for _ in range(maks):
        p = expit(X @ theta)
        w = p * (1 - p)
        H = (X.T * w) @ X
        A = np.sqrt(w)[:, None] * X
        h = np.einsum("ij,jk,ik->i", A, np.linalg.inv(H), A)
        skor = X.T @ (y - p + h * (0.5 - p))
        d = np.linalg.solve(H, skor)
        theta = theta + d
        if np.max(np.abs(d)) < tol:
            break
    return theta
