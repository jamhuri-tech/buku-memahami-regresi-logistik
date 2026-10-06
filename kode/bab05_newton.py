"""Bab 5: metode Newton, IRLS, dan Newton teredam pada data mini.

Fungsi newton dan irls di sini dipakai lagi di bab-bab berikutnya.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian


def newton(X, y, w0=None, langkah=20, redam=False, c=1e-4):
    """Newton (teredam bila redam=True, pencarian garis Armijo).

    Mengembalikan jejak w dan panjang langkah t yang dipakai.
    """
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    jejak, panjang = [w.copy()], []
    for _ in range(langkah):
        g, H = gradien(w, X, y), hessian(w, X, y)
        s = -np.linalg.solve(H, g)
        t = 1.0
        if redam:
            L0 = log_loss(w, X, y)
            while log_loss(w + t * s, X, y) > L0 + c * t * (g @ s):
                t /= 2
        w = w + t * s
        jejak.append(w.copy())
        panjang.append(t)
        if np.abs(gradien(w, X, y)).max() < 1e-12:
            break
    return np.array(jejak), panjang


def irls(X, y, langkah=20):
    """IRLS: kuadrat terkecil berbobot pada respons kerja."""
    w = np.zeros(X.shape[1])
    for _ in range(langkah):
        z = X @ w
        p = expit(z)
        d = p * (1 - p)
        r = z + (y - p) / d
        w = np.linalg.solve(X.T @ (X * d[:, None]), X.T @ (d * r))
    return w


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    v = np.array([-2.0, 1.0, -1.0])
    j, _ = newton(X, y, langkah=8)
    print("(1) Newton dari w = (0, 0, 0):")
    print("     k     w0       w1       w2       L      |w - MLE|")
    for k, w in enumerate(j):
        e = np.abs(w - m).max()
        teks = f"{e:.1e}" if e > 1e-10 else "< 1e-10"
        print(f"    {k:2d}  {w[0]:7.4f}  {w[1]:7.4f}  {w[2]:7.4f}"
              f"  {log_loss(w, X, y):.6f}  {teks}")
    c = [w[1] for w in j[1:5]]
    print("    w_k = c_k (-2, 1, -1):")
    print("      c_k =", np.round(c, 6))
    print(f"    sejajar (-2, 1, -1): "
          f"{all(np.allclose(w, w[1] * v) for w in j[1:])}")

    r0 = 4 * y - 2
    print("(2) IRLS langkah pertama:")
    print("    respons kerja r =", r0)
    print("    X^T X w = X^T r, X^T r =", (X.T @ r0).astype(float))
    print("    hasil:", np.round(irls(X, y, 1), 4) + 0.0)
    print("    IRLS 10 langkah = MLE:", np.allclose(irls(X, y, 10), m))

    w0 = np.array([-6.0, 2.0, -2.0])
    print("(3) dari w0 = (-6, 2, -2):")
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        j, _ = newton(X, y, w0, langkah=2)
    for k, w in enumerate(j[:2]):
        print(f"    murni  k = {k}: w = ({w[0]:.2f}, {w[1]:.2f},"
              f" {w[2]:.2f}), L = {log_loss(w, X, y):.4f}")
    print(f"    murni  k = 2: L = {log_loss(j[2], X, y):.4f} (naik)")
    j, t = newton(X, y, w0, langkah=30, redam=True)
    print("    teredam, langkah t:", t)
    print(f"    teredam: {len(t)} langkah, w = "
          f"({j[-1][0]:.4f}, {j[-1][1]:.4f}, {j[-1][2]:.4f})")

    print("(4) pustaka (semua dibandingkan dengan MLE):")
    for s in ["lbfgs", "newton-cg", "newton-cholesky"]:
        mm = LogisticRegression(penalty=None, solver=s,
                                tol=1e-10).fit(Xf, y)
        w = np.r_[mm.intercept_, mm.coef_[0]]
        print(f"    sklearn {s:<16} |selisih| < 1e-5: "
              f"{np.abs(w - m).max() < 1e-5}")
    rs = sm.Logit(y, X).fit(disp=0)
    print(f"    statsmodels Logit (newton)   |selisih| < 1e-8: "
          f"{np.abs(rs.params - m).max() < 1e-8}")
