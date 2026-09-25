"""Bab 17: pembobotan kelas dan ambang pada kejadian langka.

(1) class_weight="balanced": bobot peubah hampir tidak berubah,
    intersep bergeser log(n0 / n1).
(2) Tebakan model berbobot dengan ambang 0.5 sama dengan tebakan model
    tanpa bobot dengan ambang yang digeser.
(3) Biaya pada data uji baru untuk c_FN = 50, c_FP = 1.
"""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bab17_data import populasi

C_FN, C_FP = 50.0, 1.0


def latih(X, y, **opsi):
    return LogisticRegression(penalty=None, tol=1e-10, max_iter=1000,
                              **opsi).fit(X, y)


def biaya(y, p, t):
    tebak = p >= t
    return (C_FN * np.sum(~tebak & (y == 1))
            + C_FP * np.sum(tebak & (y == 0))) / len(y)


if __name__ == "__main__":
    X, y, _ = populasi()
    Xu, yu, pu = populasi(benih=1)
    a, b = latih(X, y), latih(X, y, class_weight="balanced")
    geser = np.log((len(y) - y.sum()) / y.sum())
    for awal, nama, m in (("(1)", "tanpa bobot", a),
                          ("   ", "balanced   ", b)):
        w = " ".join(f"{v:+.3f}" for v in m.coef_[0])
        print(f"{awal} {nama}: b = {m.intercept_[0]:.4f}")
        print(f"                 w = {w}")
    print(f"    selisih intersep {b.intercept_[0] - a.intercept_[0]:.4f}"
          f", log(n0 / n1) = {geser:.4f}")
    pa, pb = a.predict_proba(Xu)[:, 1], b.predict_proba(Xu)[:, 1]
    t = expit(-geser)
    sama = np.mean((pb >= 0.5) == (pa >= t))
    print(f"(2) ambang setara tanpa bobot: sigma(-log(n0/n1)) = {t:.4f}")
    print(f"    tebakan sama pada {sama:.4f} data uji")
    print(f"    AUC uji: tanpa bobot {roc_auc_score(yu, pa):.4f}, "
          f"balanced {roc_auc_score(yu, pb):.4f}")
    t_biaya = C_FP / (C_FP + C_FN)
    print(f"(3) biaya per titik, c_FN = {C_FN:g}, c_FP = {C_FP:g}:")
    print(f"    tanpa bobot, ambang {t_biaya:.4f}: {biaya(yu, pa, t_biaya):.4f}")
    print(f"    balanced,    ambang 0.5   : {biaya(yu, pb, 0.5):.4f}")
    print(f"    balanced,    ambang {t_biaya:.4f}: {biaya(yu, pb, t_biaya):.4f}")
    print(f"    peluang sebenarnya, {t_biaya:.4f}: {biaya(yu, pu, t_biaya):.4f}")
    print(f"    selalu negatif           : {biaya(yu, pa, 2.0):.4f}")
