"""Bab 4: dua jalan yang sampai ke bentuk logistik.

(1) Dua kelas bersebaran normal dengan kovarians sama: peluang
    posteriornya tepat berbentuk logistik dengan logit linear.
(2) Peubah laten: y = 1 jika w x + b + e > 0. Galat e bersebaran
    logistik memberi regresi logistik; galat normal memberi probit.
(3) Fungsi logistik berskala 1.702 hampir berimpit dengan probit.
"""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from bab04_data import BENIH
from bab04_sigmoid import sigmoid


def bobot_lda(mu0, mu1, S, pi1):
    """Bobot dan intersep posterior dua kelas normal, kovarians S."""
    Si = np.linalg.inv(S)
    w = Si @ (mu1 - mu0)
    b = (-0.5 * mu1 @ Si @ mu1 + 0.5 * mu0 @ Si @ mu0
         + np.log(pi1 / (1 - pi1)))
    return w, b


def cocokkan(X, y):
    m = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000)
    m.fit(X, y)
    return m.coef_[0], m.intercept_[0]


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    n = 200_000
    mu0, mu1 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    pi1 = 0.3
    w, b = bobot_lda(mu0, mu1, S, pi1)
    y = (rng.random(n) < pi1).astype(int)
    L = np.linalg.cholesky(S)
    X = rng.normal(size=(n, 2)) @ L.T + np.where(y[:, None] == 1,
                                                 mu1, mu0)
    wt, bt = cocokkan(X, y)
    print("(1) dua kelas normal, kovarians sama, pi1 = 0.3")
    print(f"    rumus   : w = ({w[0]:.4f}, {w[1]:.4f}), b = {b:.4f}")
    print(f"    taksiran: w = ({wt[0]:.4f}, {wt[1]:.4f}), "
          f"b = {bt:.4f}")
    x = rng.normal(size=n)
    for nama, galat in (("logistik", rng.logistic(size=n)),
                        ("normal  ", rng.normal(size=n))):
        y = (1.5 * x - 0.5 + galat > 0).astype(int)
        wt, bt = cocokkan(x[:, None], y)
        print(f"(2) galat {nama}: w = {wt[0]:.4f}, b = {bt:.4f}")
    z = np.linspace(-8, 8, 16001)
    for c in (1.0, 1.6, 1.702):
        beda = np.abs(sigmoid(c * z) - norm.cdf(z))
        print(f"(3) c = {c:<5}: maks |sigma(c z) - Phi(z)| = "
              f"{beda.max():.4f}")
    print(f"    rasio simpangan baku logistik/normal = "
          f"{np.pi / np.sqrt(3):.4f}")
