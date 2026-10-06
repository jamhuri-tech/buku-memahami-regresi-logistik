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


def kovarians(w, X):
    """Cov(w) = (X^T D X)^{-1}, invers informasi Fisher."""
    p = expit(X @ w)
    d = p * (1 - p)
    return np.linalg.inv(X.T @ (X * d[:, None]))


def prediksi_selang(w, C, X0, z=Z975):
    """Peluang dan selang metode delta (dibangun di skala logit)."""
    eta = X0 @ w
    se = np.sqrt(np.einsum("ij,jk,ik->i", X0, C, X0))
    return expit(eta), expit(eta - z * se), expit(eta + z * se), se


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    C = kovarians(m, X)
    print("(1) kovarians MLE (X^T D X)^{-1}, dikali 417:")
    for baris in C * 417:
        print("    " + " ".join(f"{v:9.2f}" for v in baris))
    print("    galat baku:", np.round(np.sqrt(np.diag(C)), 4))
    X0 = np.array([[4.0, 1.0], [2.0, 0.0], [6.0, 3.0], [8.0, 1.0]])
    p, lo, hi, se = prediksi_selang(m, C, rancang(X0))
    print("(2) prediksi dan selang 95% (metode delta):")
    print("    x1  x2     z      SE(z)    p      bawah   atas")
    for i in range(len(X0)):
        print(f"    {X0[i, 0]:2.0f}  {X0[i, 1]:2.0f}  {np.round(rancang(X0)[i] @ m, 4) + 0.0:6.4f}"
              f"  {se[i]:.4f}  {p[i]:.4f}  {lo[i]:.4f}  {hi[i]:.4f}")
    r = sm.Logit(y, X).fit(disp=0)
    f = r.get_prediction(rancang(X0)).summary_frame(alpha=0.05)
    print("    sama dengan statsmodels get_prediction:",
          np.allclose(f["ci_lower"], lo) and np.allclose(f["ci_upper"], hi))

    print("(3) ambang t: tebak lulus bila x1 - x2 >= 2 + logit(t)/ln 3")
    for t in [0.5, 0.25, 0.2]:
        batas = 2 + logit(t) / np.log(3)
        yh = (expit(X @ m) >= t - 1e-12).astype(int)
        print(f"    t = {t:4.2f}: x1 - x2 >= {batas:.4f},"
              f" tebakan = {yh}")
