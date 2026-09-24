"""Bab 9: satu langkah Newton sama dengan kuadrat terkecil berbobot.

irls_langkah (Listing 9.3) menghitung langkah Newton sebagai
penyelesaian kuadrat terkecil dengan bobot d_i = p_i (1 - p_i) dan
respons kerja z_i = x_i^T theta + (y_i - p_i) / d_i. Bagian utama
membandingkannya dengan langkah Newton, lalu dengan GLM statsmodels
yang memakai IRLS.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab04_data import jam_belajar
from bab06_turunan import rancang
from bab09_newton import newton


def irls_langkah(X, y, theta):
    p = expit(X @ theta)
    d = p * (1 - p)
    z = X @ theta + (y - p) / d          # respons kerja
    akar = np.sqrt(d)
    return np.linalg.lstsq(akar[:, None] * X, akar * z,
                           rcond=None)[0]


if __name__ == "__main__":
    x, y = jam_belajar()
    X = rancang(x)
    t_newton = newton(X, y, np.zeros(2), 6)
    theta, beda = np.zeros(2), 0.0
    for k in range(6):
        theta = irls_langkah(X, y, theta)
        beda = max(beda, np.abs(theta - t_newton[k + 1]).max())
    print(f"IRLS lawan Newton, 6 iterasi: maks selisih < 1e-10: "
          f"{beda < 1e-10}")
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    logit = sm.Logit(y, X).fit(disp=0)
    print(f"statsmodels GLM (IRLS): {glm.fit_history['iteration']} "
          f"iterasi, b = {glm.params[0]:.4f}, w = {glm.params[1]:.4f}")
    print(f"statsmodels Logit (Newton): {logit.mle_retvals['iterations']}"
          f" iterasi, b = {logit.params[0]:.4f}, "
          f"w = {logit.params[1]:.4f}")
