"""Bab 11: penalti L2 dan L1 pada data jarang.

Untuk setiap C pada grid, model L2 (lbfgs) dan L1 (saga) dilatih pada
100 titik dan dinilai dengan log-loss pada 10 ribu titik uji. Dicatat
juga banyaknya bobot bukan nol dan berapa dari lima peubah yang
sebenarnya berpengaruh ikut terpilih.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import expit

from bab07_deteksi import pemisahan
from bab11_data import W_JARANG, data_jarang

CS = np.geomspace(1e-3, 1e2, 21)


def latih(penalti, C, X, y):
    solver = "lbfgs" if penalti == "l2" else "saga"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(penalty=penalti, C=C, solver=solver,
                                  tol=1e-6, max_iter=20_000,
                                  random_state=0).fit(X, y)


def jalur(X, y, Xu, yu):
    hasil = {"l2": [], "l1": []}
    for pen in hasil:
        for C in CS:
            m = latih(pen, C, X, y)
            hasil[pen].append({
                "C": C, "coef": m.coef_[0].copy(),
                "latih": log_loss(y, m.predict_proba(X)),
                "uji": log_loss(yu, m.predict_proba(Xu))})
    return hasil


if __name__ == "__main__":
    X, y, Xu, yu = data_jarang()
    print(f"data latih: n = {len(y)}, d = {X.shape[1]}, "
          f"{pemisahan(X, y)} terpisah")
    m = LogisticRegression(penalty=None, max_iter=10_000).fit(X, y)
    ll = log_loss(yu, m.predict_proba(Xu))
    print(f"tanpa penalti: log-loss uji {ll:.3f}"
          f", maks |w| = {np.abs(m.coef_).max():.1f}")
    print(f"model sebenarnya: log-loss uji "
          f"{log_loss(yu, expit(Xu @ W_JARANG)):.3f}")
    hasil = jalur(X, y, Xu, yu)
    for pen, catat in hasil.items():
        k = int(np.argmin([c["uji"] for c in catat]))
        c = catat[k]
        nz = np.abs(c["coef"]) > 1e-8
        print(f"{pen.upper()} terbaik: C = {c['C']:.3g}, log-loss uji "
              f"{c['uji']:.3f}, latih {c['latih']:.3f}")
        print(f"    bobot bukan nol {nz.sum()}, dari lima peubah "
              f"penting terpilih {nz[:5].sum()}")
