"""Bab 14: fitur polinomial dan spline.

(1) Data cincin: model linear lawan polinomial derajat 2 dan lebih.
(2) Data bulan: derajat 1 sampai 9, tanpa penalti lawan penalti L2
    yang C-nya dipilih validasi silang berskor log-loss.
(3) Data bentuk U satu peubah: linear lawan spline kubik.
Semua dinilai dengan log-loss pada 5000 titik uji.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer,
                                   StandardScaler)

from bab14_data import bentuk_u, bulan, cincin

warnings.simplefilter("ignore")
CS = np.geomspace(1e-3, 1e3, 13)


def polinomial(derajat, penalti):
    if penalti:
        akhir = LogisticRegressionCV(Cs=CS, scoring="neg_log_loss",
                                     max_iter=100_000)
    else:
        akhir = LogisticRegression(penalty=None, max_iter=100_000)
    return make_pipeline(PolynomialFeatures(derajat, include_bias=False),
                         StandardScaler(), akhir)


if __name__ == "__main__":
    X, y, _ = cincin(300)
    Xu, yu, pu = cincin(5000, benih=1)
    print(f"(1) cincin, log-loss uji model sebenarnya {log_loss(yu, pu):.3f}")
    for d in (1, 2, 3, 6):
        m = polinomial(d, False).fit(X, y)
        print(f"    derajat {d}, tanpa penalti: "
              f"{log_loss(yu, m.predict_proba(Xu)):.3f}")
    X, y = bulan(200)
    Xu, yu = bulan(5000, benih=1)
    print("(2) bulan, log-loss uji:")
    print("    derajat  tanpa penalti  penalti CV   C terpilih")
    for d in (1, 2, 3, 4, 6, 9):
        a = polinomial(d, False).fit(X, y)
        b = polinomial(d, True).fit(X, y)
        print(f"    {d:5d}    {log_loss(yu, a.predict_proba(Xu)):11.3f}"
              f"  {log_loss(yu, b.predict_proba(Xu)):10.3f}   "
              f"{b[-1].C_[0]:g}")
    x, y, _ = bentuk_u(300)
    xu, yu, pu = bentuk_u(5000, benih=1)
    lin = LogisticRegression(penalty=None).fit(x, y)
    spl = make_pipeline(SplineTransformer(n_knots=5, degree=3),
                        LogisticRegression(penalty=None,
                                           max_iter=10_000)).fit(x, y)
    print(f"(3) bentuk U, log-loss uji: sebenarnya {log_loss(yu, pu):.3f}")
    print(f"    linear {log_loss(yu, lin.predict_proba(xu)):.3f}, "
          f"spline {log_loss(yu, spl.predict_proba(xu)):.3f}")
