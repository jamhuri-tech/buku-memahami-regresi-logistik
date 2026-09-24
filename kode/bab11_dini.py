"""Bab 11: penghentian dini lawan penalti L2.

(1) Data jarang: gradient descent dari nol dengan eta = 1/L; log-loss
    uji dicatat setiap langkah, dan langkah terbaiknya dibandingkan
    dengan penalti L2 terbaik dari bab11_jalur.
(2) Data dua peubah: jarak setiap titik lintasan gradient descent ke
    jalur penalti L2 (dengan norma bobot yang sama).
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from bab04_data import dua_peubah
from bab06_turunan import gradien, rancang
from bab11_data import data_jarang

warnings.simplefilter("ignore")


def lintasan_gd(X, y, langkah):
    Xt = rancang(X)
    eta = 4 * len(y) / np.linalg.eigvalsh(Xt.T @ Xt).max()
    theta, jalur = np.zeros(Xt.shape[1]), []
    for _ in range(langkah):
        theta = theta - eta * gradien(theta, Xt, y)
        jalur.append(theta.copy())
    return np.array(jalur)


def jalur_l2(X, y, Cs):
    hasil = []
    for C in Cs:
        m = LogisticRegression(C=C, tol=1e-10, max_iter=10_000)
        m.fit(X, y)
        hasil.append(np.r_[m.intercept_, m.coef_[0]])
    return np.array(hasil)


if __name__ == "__main__":
    X, y, Xu, yu = data_jarang()
    jalur = lintasan_gd(X, y, 3000)
    uji = [log_loss(yu, expit(rancang(Xu) @ t)) for t in jalur]
    k = int(np.argmin(uji))
    print(f"(1) data jarang: langkah GD terbaik {k + 1}, log-loss uji "
          f"{uji[k]:.3f}")
    print(f"    sesudah 3000 langkah: log-loss uji {uji[-1]:.3f}")
    X2, y2 = dua_peubah()
    gd = lintasan_gd(X2, y2, 400)
    l2 = jalur_l2(X2, y2, np.geomspace(1e-4, 1e3, 400))
    print("(2) dua peubah: jarak lintasan GD ke jalur L2")
    print("    langkah   ||w||   jarak ke jalur L2")
    for t in (1, 5, 20, 100, 400):
        w = gd[t - 1]
        jarak = np.min(np.linalg.norm(l2 - w, axis=1))
        print(f"    {t:6d}   {np.linalg.norm(w[1:]):5.3f}   {jarak:.4f}")
