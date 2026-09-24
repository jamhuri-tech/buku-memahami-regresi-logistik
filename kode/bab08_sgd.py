"""Bab 8: SGD dan SAGA pada data sintetis 10 ribu titik, 10 peubah.

Semua metode dibandingkan per epoch (n gradien satu titik): gradient
descent penuh, SGD dengan laju tetap dan laju meluruh, dan SAGA
(Listing 8.2). Sesudah itu SAGA scikit-learn dicocokkan dengan MLE,
dan dijalankan pada jam belajar mentah yang tidak dibakukan.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab04_data import BENIH, jam_belajar

EPOCH = 20


def data_besar(n=10_000, d=10, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(n, d))
    theta = rng.normal(size=d + 1)
    Xt = np.column_stack([np.ones(n), X])
    y = (rng.random(n) < expit(Xt @ theta)).astype(float)
    return Xt, y


def loss(theta, X, y):
    z = X @ theta
    return np.mean(np.logaddexp(0, z) - y * z)


def sgd(X, y, laju, epoch, rng):
    n, m = X.shape
    theta, k, catat = np.zeros(m), 0, []
    for _ in range(epoch):
        for i in rng.integers(0, n, size=n):
            k += 1
            g = (expit(X[i] @ theta) - y[i]) * X[i]
            theta = theta - laju(k) * g
        catat.append(theta.copy())
    return catat


def saga(X, y, eta, epoch, rng):
    n, m = X.shape
    theta = np.zeros(m)
    sisa = np.zeros(n)          # p_i - y_i tersimpan
    g_rata = np.zeros(m)        # rata-rata gradien
    catat = []
    for _ in range(epoch):
        for i in rng.integers(0, n, size=n):
            baru = expit(X[i] @ theta) - y[i]
            beda = (baru - sisa[i]) * X[i]
            theta = theta - eta * (beda + g_rata)
            g_rata += beda / n
            sisa[i] = baru
        catat.append(theta.copy())
    return catat


def semua_metode(X, y):
    n = len(y)
    L = np.linalg.eigvalsh(X.T @ X / n).max() / 4
    L_maks = (X ** 2).sum(axis=1).max() / 4
    hasil, theta = {}, np.zeros(X.shape[1])
    hasil["GD, eta = 1/L"] = []
    for _ in range(EPOCH):
        theta = theta - (X.T @ (expit(X @ theta) - y) / n) / L
        hasil["GD, eta = 1/L"].append(theta.copy())
    for nama, laju in (("SGD, eta = 0.05", lambda k: 0.05),
                       ("SGD, eta = 0.01", lambda k: 0.01),
                       ("SGD, eta = 0.5/(1+k/n)",
                        lambda k: 0.5 / (1 + k / n))):
        hasil[nama] = sgd(X, y, laju, EPOCH, np.random.default_rng(1))
    hasil["SAGA, eta = 1/(3 L_maks)"] = saga(
        X, y, 1 / (3 * L_maks), EPOCH, np.random.default_rng(1))
    return hasil, L, L_maks


if __name__ == "__main__":
    X, y = data_besar()
    ref = sm.Logit(y, X).fit(disp=0).params
    L_opt = loss(ref, X, y)
    hasil, L, L_maks = semua_metode(X, y)
    print(f"n = {len(y)}, d = {X.shape[1] - 1}, log-loss minimum "
          f"{L_opt:.6f}")
    print(f"L = {L:.4f}, L_maks = max ||x_i||^2 / 4 = {L_maks:.4f}")
    print("selisih log-loss dari minimum sesudah epoch ke-")
    print("                              1        5       10       20")
    for nama, catat in hasil.items():
        sel = [loss(catat[e - 1], X, y) - L_opt for e in (1, 5, 10, 20)]
        print(f"{nama:25s}" + "".join(f"{v:9.1e}" for v in sel))
    Xp = X[:, 1:]
    m = LogisticRegression(penalty=None, solver="saga", tol=1e-8,
                           random_state=0, max_iter=1000).fit(Xp, y)
    beda = np.abs(np.r_[m.intercept_, m.coef_[0]] - ref).max()
    print(f"scikit-learn saga: {m.n_iter_[0]} epoch, "
          f"maks |theta - MLE| < 1e-6: {beda < 1e-6}")
    x, yj = jam_belajar()
    for nama, u in (("mentah", x), ("dibakukan", (x - x.mean()) / x.std())):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = LogisticRegression(penalty=None, solver="saga",
                                   random_state=0,
                                   max_iter=100).fit(u[:, None], yj)
        print(f"jam belajar {nama}: saga {m.n_iter_[0]} epoch, "
              f"w = {m.coef_[0, 0]:.4f}, peringatan {len(w)}")
