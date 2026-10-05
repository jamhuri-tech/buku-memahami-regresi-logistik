"""Bab 12: diagnostik -- residu Pearson dan deviance, leverage, jarak
Cook, dfbeta, VIF, dan linearitas logit. Fungsi-fungsi di sini dipakai
lagi di Bab 15.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import chi2

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab10_inferensi import log_kem, mle


def residu(y, p):
    pearson = (y - p) / np.sqrt(p * (1 - p))
    dev = np.sign(y - p) * np.sqrt(-2 * (y * np.log(p)
                                         + (1 - y) * np.log(1 - p)))
    return pearson, dev


def leverage(X, p):
    d = p * (1 - p)
    Ii = np.linalg.inv(X.T @ (X * d[:, None]))
    return d * np.einsum("ij,jk,ik->i", X, Ii, X), Ii


def cook(y, p, h, k):
    rp = (y - p) / np.sqrt(p * (1 - p))
    return rp ** 2 * h / (k * (1 - h) ** 2)


def terpisah_satu_peubah(x, y):
    return x[y == 1].min() > x[y == 0].max() or \
        x[y == 1].max() < x[y == 0].min()


def vif(X):
    """VIF dari matriks korelasi peubah (tanpa kolom satu)."""
    return np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    p = expit(X @ m)
    rp, rd = residu(y, p)
    h, Ii = leverage(X, p)
    D = cook(y, p, h, 2)
    print("(1) diagnostik titik (data mini):")
    print("     i  Pearson  deviance  leverage   Cook")
    for i in range(6):
        print(f"    {i + 1:2d}  {rp[i]:7.4f}  {rd[i]:8.4f}   {h[i]:.4f}"
              f"   {D[i]:.4f}")
    print(f"    sum Pearson^2 = {(rp ** 2).sum():.4f},"
          f" sum deviance^2 = {(rd ** 2).sum():.4f}")
    print(f"    sum leverage = {h.sum():.4f}")
    inf = sm.GLM(y, X, family=sm.families.Binomial()).fit().get_influence()
    print("    sama dengan statsmodels:",
          np.allclose(inf.hat_matrix_diag, h),
          np.allclose(inf.cooks_distance[0], D))

    print("(2) dfbeta: satu langkah lawan buang-satu yang eksak:")
    satu = (Ii @ (X * (y - p)[:, None]).T).T / (1 - h)[:, None]
    for i in range(6):
        k = np.arange(6) != i
        if terpisah_satu_peubah(x[k], y[k]):
            eksak = "data terpisah: MLE tidak ada"
        else:
            t = m - mle(X[k], y[k])
            eksak = f"({t[0]:7.4f}, {t[1]:7.4f})"
        print(f"    i = {i + 1}: ({satu[i, 0]:7.4f}, {satu[i, 1]:7.4f})"
              f"  {eksak}")

    X3 = np.c_[X, x * np.log(x)]
    t3 = mle(X3, y)
    G = 2 * (log_kem(t3, X3, y) - log_kem(m, X, y))
    print(f"(3) Box-Tidwell (suku x log x): G = {G:.4f},"
          f" p = {chi2.sf(G, 1):.4f}")

    rng = np.random.default_rng(BENIH)
    n = 1000
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + np.sqrt(1 - 0.81) * rng.normal(size=n)
    x3 = rng.normal(size=n)
    eta = -0.5 + x1 + 0.5 * x2 + 0.8 * x3 - 0.6 * x3 ** 2
    yy = (rng.random(n) < expit(eta)).astype(int)
    print("(4) data sintetis n = 1000 (x1, x2 berkorelasi 0.9):")
    print("    VIF:", np.round(vif(np.c_[x1, x2, x3]), 2))
    for nama, A in [("x1 saja", np.c_[x1, x3, x3 ** 2]),
                    ("x1 dan x2", np.c_[x1, x2, x3, x3 ** 2])]:
        r = sm.Logit(yy, sm.add_constant(A)).fit(disp=0)
        print(f"    {nama:<9}: w1 = {r.params[1]:.4f},"
              f" SE(w1) = {r.bse[1]:.4f}")
    A1 = sm.add_constant(np.c_[x1, x2, x3])
    A2 = sm.add_constant(np.c_[x1, x2, x3, x3 ** 2])
    r1 = sm.Logit(yy, A1).fit(disp=0)
    r2 = sm.Logit(yy, A2).fit(disp=0)
    G = 2 * (r2.llf - r1.llf)
    teks = f"{chi2.sf(G, 1):.1e}" if chi2.sf(G, 1) > 1e-10 else "< 1e-10"
    print(f"    linearitas x3: G (suku x3^2) = {G:.2f}, p {teks}")
