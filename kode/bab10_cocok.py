"""Bab 10: mencocokkan RegresiLogistikKita dengan scikit-learn.

(1) Penalti L2 untuk beberapa C dan tanpa penalti, pada dua data.
(2) predict_proba, decision_function, dan predict, dengan label teks.
(3) sample_weight bulat sama dengan menggandakan baris.
(4) C mengalikan jumlah loss: menggandakan data sama dengan
    menggandakan C.
(5) liblinear: intersep ikut dipenalti, dan intercept_scaling.
(6) Tanpa penalti: sama dengan statsmodels.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from bab04_data import dua_peubah
from bab08_sgd import data_besar
from bab10_kita import RegresiLogistikKita


def theta(m):
    return np.r_[m.intercept_, m.coef_[0]]


def beda(a, b):
    return np.abs(theta(a) - theta(b)).max()


def sk(**opsi):
    return LogisticRegression(tol=1e-12, max_iter=10_000, **opsi)


if __name__ == "__main__":
    Xb, yb = data_besar()
    semua = {"dua peubah": dua_peubah(), "sintetis": (Xb[:, 1:], yb)}
    print("(1) maks |theta_kita - theta_sklearn| < 1e-6:")
    for nama, (X, y) in semua.items():
        hasil = []
        for C in (0.01, 1.0, 100.0, None):
            if C is None:
                a = RegresiLogistikKita(penalti=None).fit(X, y)
                b = sk(penalty=None).fit(X, y)
            else:
                a = RegresiLogistikKita(C=C).fit(X, y)
                b = sk(C=C).fit(X, y)
            label = "None" if C is None else f"{C:g}"
            hasil.append(f"{label}: {beda(a, b) < 1e-6}")
        print(f"    {nama:10s} C = " + ", ".join(hasil))
    X, y = dua_peubah()
    teks = np.where(y == 1, "lulus", "gagal")
    a = RegresiLogistikKita().fit(X, teks)
    b = sk().fit(X, teks)
    print(f"(2) classes_ kita {[str(c) for c in a.classes_]}")
    print(f"    classes_ sklearn {[str(c) for c in b.classes_]}")
    print(f"    maks beda predict_proba < 1e-8: "
          f"{np.abs(a.predict_proba(X) - b.predict_proba(X)).max() < 1e-8}")
    print(f"    predict sama untuk semua titik: "
          f"{np.array_equal(a.predict(X), b.predict(X))}")
    s = np.random.default_rng(0).integers(1, 4, size=len(y))
    Xg, yg = np.repeat(X, s, axis=0), np.repeat(y, s)
    for nama, kelas in (("kita", RegresiLogistikKita), ("sklearn", sk)):
        m1 = kelas().fit(X, y, sample_weight=s)
        m2 = kelas().fit(Xg, yg)
        awal = "(3)" if nama == "kita" else "   "
        print(f"{awal} {nama:7s}: bobot sampel = baris digandakan: "
              f"{beda(m1, m2) < 1e-6}")
    X2, y2 = np.vstack([X, X]), np.r_[y, y]
    m1 = sk(C=1.0).fit(X, y)
    m2 = sk(C=1.0).fit(X2, y2)
    m3 = sk(C=2.0).fit(X, y)
    print(f"(4) C = 1, data asli    : w = {m1.coef_[0].round(4)}")
    print(f"    C = 1, data ganda   : w = {m2.coef_[0].round(4)}")
    print(f"    C = 2, data asli    : w = {m3.coef_[0].round(4)}")
    print("(5) liblinear, C = 0.01:")
    for skala in (1.0, 100.0):
        a = RegresiLogistikKita(C=0.01, penalti_intersep=True,
                                skala_intersep=skala).fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = sk(C=0.01, solver="liblinear",
                   intercept_scaling=skala).fit(X, y)
        print(f"    intercept_scaling = {skala:5g}: b kita = "
              f"{a.intercept_[0]:.4f}, sklearn {b.intercept_[0]:.4f}")
    b = sk(C=0.01).fit(X, y)
    print(f"    lbfgs (intersep tak dipenalti): b = {b.intercept_[0]:.4f}")
    a = RegresiLogistikKita(penalti=None).fit(X, y)
    t = sm.Logit(y, sm.add_constant(X)).fit(disp=0).params
    print(f"(6) tanpa penalti lawan statsmodels < 1e-8: "
          f"{np.abs(theta(a) - t).max() < 1e-8}")
    print(f"    iterasi Newton kita: {a.n_iter_[0]}")
