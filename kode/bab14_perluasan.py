"""Bab 14: kelas tak seimbang (bobot kelas, koreksi kasus-kontrol)
dan regresi logistik multikelas (softmax).
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bab01_data import BENIH
from bab09_kalibrasi import ece


def data_jarang(m, benih):
    """Dua fitur, prevalensi sekitar 5 persen."""
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(m, 2))
    z = -3.5 + X @ np.array([1.0, 0.5])
    y = (rng.random(m) < expit(z)).astype(int)
    return X, y


def model(**kw):
    return LogisticRegression(penalty=None, solver="newton-cholesky",
                              tol=1e-10, **kw)


def softmax_loss(Wf, X, Y):
    """Cross-entropy rata-rata dan gradiennya; W berukuran (n+1, K)."""
    W = Wf.reshape(X.shape[1], Y.shape[1])
    Z = X @ W
    L = np.mean(logsumexp(Z, axis=1) - np.sum(Y * Z, axis=1))
    G = X.T @ (softmax(Z, axis=1) - Y) / len(X)
    return L, G.ravel()


if __name__ == "__main__":
    X, y = data_jarang(20000, BENIH)
    Xu, yu = data_jarang(20000, BENIH + 1)
    m1, m0 = y.sum(), len(y) - y.sum()
    print(f"(1) data jarang: {m1} positif dari {len(y)}"
          f" ({y.mean():.4f})")
    a = model().fit(X, y)
    b = model(class_weight="balanced").fit(X, y)
    print("    model         w0      w1      w2    rata p  ECE    AUC")
    for nama, m in [("tanpa bobot", a), ("balanced", b)]:
        p = m.predict_proba(Xu)[:, 1]
        print(f"    {nama:<11} {m.intercept_[0]:7.4f} {m.coef_[0, 0]:.4f}"
              f"  {m.coef_[0, 1]:.4f}  {p.mean():.4f}  {ece(yu, p):.4f}"
              f" {roc_auc_score(yu, p):.4f}")
    print(f"    selisih w0 {b.intercept_[0] - a.intercept_[0]:.4f},"
          f" log(m0/m1) = {np.log(m0 / m1):.4f}")

    rng = np.random.default_rng(BENIH + 5)
    kas = np.flatnonzero(y == 1)
    kon = rng.choice(np.flatnonzero(y == 0), size=len(kas), replace=False)
    i = np.r_[kas, kon]
    c = model().fit(X[i], y[i])
    r1, r0 = 1.0, len(kon) / m0
    print("(2) sampel kasus-kontrol 1:1:")
    print(f"    w0 sampel = {c.intercept_[0]:.4f},"
          f" koreksi log(r1/r0) = {np.log(r1 / r0):.4f}")
    print(f"    w0 terkoreksi = {c.intercept_[0] - np.log(r1 / r0):.4f},"
          f" w0 data penuh = {a.intercept_[0]:.4f}")
    print(f"    (w1, w2) sampel = ({c.coef_[0, 0]:.4f},"
          f" {c.coef_[0, 1]:.4f})")

    rng = np.random.default_rng(BENIH + 7)
    mm = 3000
    Xm = rng.normal(size=(mm, 2))
    Wb = np.array([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0], [0.0, -0.5, 1.5]])
    A = np.c_[np.ones(mm), Xm]
    P = softmax(A @ Wb, axis=1)
    ym = np.array([rng.choice(3, p=q) for q in P])
    Y = np.eye(3)[ym]
    h = minimize(softmax_loss, np.zeros(9), args=(A, Y), jac=True,
                 method="BFGS", options={"gtol": 1e-10})
    W = h.x.reshape(3, 3)
    print("(3) softmax tiga kelas, m = 3000:")
    print(f"    banyaknya per kelas: {np.bincount(ym)}")
    W0 = W - W[:, :1]
    print("    W (kelas 0 sebagai rujukan):")
    for j in range(3):
        print("     ", np.round(W0[j], 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = LogisticRegression(penalty=None, tol=1e-10,
                                max_iter=10000).fit(Xm, ym)
    print("    peluang sama dengan sklearn:",
          np.allclose(sk.predict_proba(Xm), softmax(A @ W, axis=1),
                      atol=1e-6))
    mn = sm.MNLogit(ym, A).fit(disp=0, method="newton")
    print("    koefisien sama dengan MNLogit:",
          np.allclose(mn.params, W0[:, 1:], atol=1e-5))
    print(f"    log-loss = {h.fun:.4f}")
