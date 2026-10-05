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


def newton(X, y, theta0=None, langkah=20, redam=False, c=1e-4):
    """Newton (teredam bila redam=True, pencarian garis Armijo).

    Mengembalikan jejak theta dan panjang langkah t yang dipakai.
    """
    theta = np.zeros(X.shape[1]) if theta0 is None else theta0.copy()
    jejak, panjang = [theta.copy()], []
    for _ in range(langkah):
        g, H = gradien(theta, X, y), hessian(theta, X, y)
        s = -np.linalg.solve(H, g)
        t = 1.0
        if redam:
            L0 = log_loss(theta, X, y)
            while log_loss(theta + t * s, X, y) > L0 + c * t * (g @ s):
                t /= 2
        theta = theta + t * s
        jejak.append(theta.copy())
        panjang.append(t)
        if np.abs(gradien(theta, X, y)).max() < 1e-12:
            break
    return np.array(jejak), panjang


def irls(X, y, langkah=20):
    """IRLS: regresi kuadrat terkecil berbobot pada respons kerja."""
    theta = np.zeros(X.shape[1])
    for _ in range(langkah):
        z = X @ theta
        p = expit(z)
        d = p * (1 - p)
        r = z + (y - p) / d
        A = X.T @ (X * d[:, None])
        theta = np.linalg.solve(A, X.T @ (d * r))
    return theta


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    j, _ = newton(X, y, langkah=8)
    print("(1) Newton dari theta = (0, 0):")
    print("     k        b         w        L       |theta - MLE|")
    for k, th in enumerate(j):
        e = np.abs(th - m).max()
        teks = f"{e:.1e}" if e > 1e-12 else "< 1e-12"
        print(f"    {k:2d}  {th[0]:8.4f}  {th[1]:8.4f}"
              f"   {log_loss(th, X, y):.6f}   {teks}")

    z0, p0 = X @ np.zeros(2), expit(X @ np.zeros(2))
    r0 = z0 + (y - p0) / (p0 * (1 - p0))
    print("(2) IRLS langkah pertama:")
    print("    respons kerja r =", r0)
    print("    kuadrat terkecil berbobot:", np.round(irls(X, y, 1), 4))
    print("    IRLS 10 langkah = MLE:", np.allclose(irls(X, y, 10), m))

    print("(3) dari theta0 = (-6, 3):")
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        j, _ = newton(X, y, np.array([-6.0, 3.0]), langkah=2)
    for k, th in enumerate(j[:2]):
        print(f"    murni  k = {k}: theta = ({th[0]:.2f}, {th[1]:.2f}),"
              f" L = {log_loss(th, X, y):.4f}")
    print(f"    murni  k = 2: |theta| > 1e5: {np.abs(j[2]).max() > 1e5}")
    j, t = newton(X, y, np.array([-6.0, 3.0]), langkah=30, redam=True)
    print("    teredam, langkah t:", t)
    print(f"    teredam: {len(t)} langkah, theta = "
          f"({j[-1][0]:.4f}, {j[-1][1]:.4f})")

    print("(4) pustaka (semua dibandingkan dengan MLE):")
    for s in ["lbfgs", "newton-cg", "newton-cholesky"]:
        mm = LogisticRegression(penalty=None, solver=s,
                                tol=1e-10).fit(x[:, None], y)
        th = np.r_[mm.intercept_, mm.coef_[0]]
        print(f"    sklearn {s:<16} |selisih| < 1e-5: "
              f"{np.abs(th - m).max() < 1e-5}")
    rs = sm.Logit(y, X).fit(disp=0)
    print(f"    statsmodels Logit (newton)   |selisih| < 1e-8: "
          f"{np.abs(rs.params - m).max() < 1e-8}")
