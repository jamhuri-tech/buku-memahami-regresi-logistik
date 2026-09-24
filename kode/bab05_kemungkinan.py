"""Bab 5: kemungkinan dan log-kemungkinan pada data jam belajar.

Menghitung kemungkinan (hasil kali peluang) dan log-kemungkinan
untuk beberapa pasangan (b, w), lalu mencari maksimumnya dengan
pencarian grid dan mencocokkannya dengan statsmodels.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab04_data import jam_belajar


def kemungkinan(b, w, x, y):
    p = expit(b + w * x)
    return np.prod(np.where(y == 1, p, 1 - p))


def log_kemungkinan(b, w, x, y):
    p = expit(b + w * x)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    x, y = jam_belajar()
    hasil = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    b_mle, w_mle = hasil.params
    print("   b        w      kemungkinan   log-kemungkinan")
    for b, w in ((0.0, 0.0), (-2.0, 0.3), (-5.0, 1.0),
                 (b_mle, w_mle)):
        print(f"{b:7.4f} {w:7.4f}   {kemungkinan(b, w, x, y):.4e}"
              f"   {log_kemungkinan(b, w, x, y):9.4f}")
    bb = np.linspace(-8, 1, 901)
    ww = np.linspace(-0.2, 1.5, 851)
    B, W = np.meshgrid(bb, ww)
    P = expit(B[..., None] + W[..., None] * x)
    LL = np.sum(np.where(y == 1, np.log(P), np.log(1 - P)), axis=-1)
    i = np.unravel_index(np.argmax(LL), LL.shape)
    print(f"grid 901 x 851: maks {LL[i]:.4f} di b = {B[i]:.2f}, "
          f"w = {W[i]:.3f}")
    print(f"statsmodels   : llf  {hasil.llf:.4f} di b = {b_mle:.4f}, "
          f"w = {w_mle:.4f}")
