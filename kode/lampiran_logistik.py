"""Lampiran B: regresi logistik dalam satu berkas NumPy.

Menaksir, menguji, meramal, dan menilai; setiap fungsi
berasal dari bab yang disebut di komentarnya.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm


def rancang(X):                                # Bab 1
    X = np.asarray(X, float).reshape(len(X), -1)
    return np.column_stack([np.ones(len(X)), X])


def log_kem(w, X, y):                          # Bab 2
    z = X @ w
    return np.sum(y * z - np.logaddexp(0, z))


def newton(X, y, lam=0.0, it=50):              # Bab 5, 6
    P = np.eye(X.shape[1])
    P[0, 0] = 0                                # w0 tidak dipenalti
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = expit(X @ w)
        g = X.T @ (p - y) + lam * P @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * P
        s = np.linalg.solve(H, g)
        u, l0 = 1.0, log_kem(w, X, y)
        while log_kem(w - u * s, X, y) < l0 - 1e-12:
            u /= 2
        w = w - u * s
    return w


def ringkasan(w, X, y):                        # Bab 10
    p = expit(X @ w)
    I = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(I)))
    z = w / se
    return se, z, 2 * norm.sf(np.abs(z))


def uji_lr(X, y, buang):                       # Bab 10
    sisa = [j for j in range(X.shape[1]) if j not in buang]
    G = 2 * (log_kem(newton(X, y), X, y)
             - log_kem(newton(X[:, sisa], y), X[:, sisa], y))
    return G, chi2.sf(G, len(buang))


def selang_p(w, X, X0, z=1.96):                # Bab 7
    p = expit(X @ w)
    C = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    e = X0 @ w
    s = np.sqrt(np.einsum("ij,jk,ik->i", X0, C, X0))
    return expit(e), expit(e - z * s), expit(e + z * s)


def auc(y, s):                                 # Bab 8
    d = s[y == 1][:, None] - s[y == 0][None, :]
    return np.mean((d > 0) + 0.5 * (d == 0))


def ukuran(y, p):                              # Bab 8
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll, np.mean((p - y) ** 2), auc(y, p)


if __name__ == "__main__":
    tabel = np.array([[1, 0, 0], [2, 1, 0], [3, 0, 1],
                      [4, 2, 1], [5, 3, 1], [6, 3, 0]])
    X, y = rancang(tabel[:, :2]), tabel[:, 2]
    w = newton(X, y)
    se, z, pv = ringkasan(w, X, y)
    print("w", w.round(4), "SE", se.round(4))
    print("nilai-p Wald", pv.round(4))
    print("LR w1 = w2 = 0:", np.round(uji_lr(X, y, [1, 2]), 4))
    print("p(4, 1), selang:",
          np.round(selang_p(w, X, rancang([[4.0, 1.0]])), 4).ravel())
    p = np.round(expit(X @ w), 12)
    print("log-loss, Brier, AUC:", np.round(ukuran(y, p), 4))
