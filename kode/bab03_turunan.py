"""Bab 3: gradien dan Hessian log-loss, diperiksa dengan beda hingga
dan dengan statsmodels.

Fungsi gradien dan hessian di sini dipakai lagi di bab-bab berikutnya.
Keduanya untuk log-loss RATA-RATA L = -ell / n.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss


def gradien(theta, X, y):
    """g = X^T (p - y) / n."""
    p = expit(X @ theta)
    return X.T @ (p - y) / len(y)


def hessian(theta, X, y):
    """H = X^T D X / n, D = diag(p (1 - p))."""
    p = expit(X @ theta)
    d = p * (1 - p)
    return X.T @ (X * d[:, None]) / len(y)


def beda_hingga(f, theta, h=1e-6):
    e = np.eye(len(theta))
    return np.array([(f(theta + h * e[j]) - f(theta - h * e[j])) / (2 * h)
                     for j in range(len(theta))])


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    print("(1) gradien dan Hessian (rata-rata):")
    for nama, th in [("theta = (0, 0)", np.zeros(2)),
                     ("theta = (-2.8, 0.8)", np.array([-2.8, 0.8])),
                     ("theta = MLE", mle_mini())]:
        g, H = gradien(th, X, y), hessian(th, X, y)
        g = np.round(g, 4) + 0.0          # tanpa -0.0000
        ev = np.linalg.eigvalsh(H)
        print(f"    {nama}")
        print(f"      g = ({g[0]:8.4f}, {g[1]:8.4f})")
        print(f"      H = [[{H[0, 0]:.4f}, {H[0, 1]:.4f}],"
              f" [{H[1, 0]:.4f}, {H[1, 1]:.4f}]]")
        print(f"      nilai eigen H: {ev[0]:.4f}, {ev[1]:.4f}")

    th = np.array([-2.8, 0.8])
    gb = beda_hingga(lambda t: log_loss(t, X, y), th)
    Hb = np.array([beda_hingga(lambda t: gradien(t, X, y)[j], th)
                   for j in range(2)])
    print("(2) beda hingga di (-2.8, 0.8):")
    print(f"    selisih gradien < 1e-8: "
          f"{np.abs(gb - gradien(th, X, y)).max() < 1e-8}")
    print(f"    selisih Hessian < 1e-8: "
          f"{np.abs(Hb - hessian(th, X, y)).max() < 1e-8}")
    m = sm.Logit(y, X)
    print("    statsmodels: score = -n g dan hessian = -n H:",
          np.allclose(m.score(th), -6 * gradien(th, X, y)),
          np.allclose(m.hessian(th), -6 * hessian(th, X, y)))

    t = mle_mini()
    p = expit(X @ t)
    print("(3) persamaan skor di MLE:")
    print(f"    sum p = {p.sum():.6f}, sum y = {y.sum()}")
    print(f"    sum x p = {(x * p).sum():.6f}, sum x y = {(x * y).sum():.0f}")
    print(f"    |b + 3.5 w| < 1e-6: {abs(t[0] + 3.5 * t[1]) < 1e-6}")
