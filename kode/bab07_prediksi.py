"""Bab 7: prediksi peluang, selang metode delta, dan ambang.

Fungsi kovarians dan prediksi_selang di sini dipakai lagi di
bab-bab berikutnya.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.stats import norm

from bab01_data import data_mini, mle_mini, rancang

Z975 = norm.ppf(0.975)


def kovarians(theta, X):
    """Cov(theta) = (X^T D X)^{-1}, invers informasi Fisher."""
    p = expit(X @ theta)
    d = p * (1 - p)
    return np.linalg.inv(X.T @ (X * d[:, None]))


def prediksi_selang(theta, C, X0, z=Z975):
    """Peluang dan selang metode delta (dibangun di skala logit)."""
    eta = X0 @ theta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X0, C, X0))
    return expit(eta), expit(eta - z * se), expit(eta + z * se), se


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    C = kovarians(m, X)
    print("(1) kovarians MLE (X^T D X)^{-1}:")
    print(f"    [[{C[0, 0]:.4f}, {C[0, 1]:.4f}], [{C[1, 0]:.4f},"
          f" {C[1, 1]:.4f}]]")
    x0 = np.array([2.5, 4.5, 5.0, 7.0])
    p, lo, hi, se = prediksi_selang(m, C, rancang(x0))
    print("(2) prediksi dan selang 95% (metode delta):")
    print("      x0      z      SE(z)     p      bawah   atas")
    for i in range(len(x0)):
        print(f"    {x0[i]:4.1f}  {m[0] + m[1] * x0[i]:7.4f}  {se[i]:6.4f}"
              f"   {p[i]:.4f}  {lo[i]:.4f}  {hi[i]:.4f}")
    r = sm.Logit(y, X).fit(disp=0)
    f = r.get_prediction(rancang(x0)).summary_frame(alpha=0.05)
    print("    sama dengan statsmodels get_prediction:",
          np.allclose(f["ci_lower"], lo) and np.allclose(f["ci_upper"], hi))

    print("(3) ambang t dan jam belajar minimum untuk tebakan lulus:")
    for t in [0.5, 0.3, 0.2]:
        xt = (logit(t) - m[0]) / m[1]
        yh = (expit(X @ m) >= t).astype(int)
        print(f"    t = {t}: x >= {xt:.4f}, tebakan = {yh}")
