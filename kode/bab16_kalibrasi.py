"""Bab 16: kalibrasi peluang regresi logistik.

(1) Kalibrasi rata-rata: di data latih, model tanpa penalti dengan
    intersep memberi rata-rata peluang tepat sama dengan proporsi
    kelas 1 (persamaan skor, Bab 6).
(2) ECE dan log-loss uji empat model: lengkap, penalti sangat kuat,
    class_weight="balanced", dan tanpa usia.
(3) Platt scaling dari nol (Listing 16.2), dicocokkan dengan
    CalibratedClassifierCV(method="sigmoid", ensemble=False).
(4) Rekalibrasi keempat model dengan sigmoid dan isotonic.
(5) Biaya pada ambang 1/6 (Bab 15) sebelum dan sesudah rekalibrasi.
"""
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bab15_ambang import biaya
from bab15_data import PEUBAH, bagi_pasien, model_pasien

warnings.simplefilter("ignore")
MODEL = [("lengkap", {}, PEUBAH),
         ("C = 0.001", {"penalty": "l2", "C": 0.001}, PEUBAH),
         ("balanced", {"class_weight": "balanced"}, PEUBAH),
         ("tanpa usia", {}, ["imt", "perokok", "wilayah"])]


def ece(y, p, k=10):
    kotak = np.minimum((p * k).astype(int), k - 1)
    total = 0.0
    for j in range(k):
        m = kotak == j
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def platt(s, y):
    """Mencari a, b sehingga sigma(a s + b) terkalibrasi."""
    n1 = y.sum()
    n0 = len(y) - n1
    t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))

    def f(ab):
        z = ab[0] * s + ab[1]
        return np.sum(np.logaddexp(0, z) - t * z)

    def g(ab):
        r = expit(ab[0] * s + ab[1]) - t
        return np.array([r @ s, r.sum()])

    return minimize(f, np.zeros(2), jac=g, method="BFGS").x


if __name__ == "__main__":
    latih, uji = bagi_pasien()
    y, yl = uji.penyakit.to_numpy(), latih.penyakit.to_numpy()
    print("(1) data latih, tol = 1e-10: rata-rata peluang")
    print(f"    proporsi kelas 1: {yl.mean():.6f}")
    for nama, opsi, peubah in MODEL:
        m = model_pasien(peubah, tol=1e-10, **opsi)
        m.fit(latih[peubah], yl)
        rata = m.predict_proba(latih[peubah])[:, 1].mean()
        print(f"    {nama:10s}: {rata:.6f}")
    print("(2) data uji     rata p   ECE     log-loss  AUC")
    for nama, opsi, peubah in MODEL:
        m = model_pasien(peubah, **opsi).fit(latih[peubah], yl)
        p = m.predict_proba(uji[peubah])[:, 1]
        print(f"    {nama:10s}  {p.mean():.4f}  {ece(y, p):.4f}  "
              f"{log_loss(y, p):.4f}    {roc_auc_score(y, p):.4f}")
    nama, opsi, peubah = MODEL[2]
    lipat = StratifiedKFold(5)
    s = cross_val_predict(model_pasien(peubah, **opsi), latih[peubah], yl,
                          cv=lipat, method="decision_function")
    a, b = platt(s, yl)
    m = model_pasien(peubah, **opsi).fit(latih[peubah], yl)
    p_kita = expit(a * m.decision_function(uji[peubah]) + b)
    sk = CalibratedClassifierCV(model_pasien(peubah, **opsi),
                                method="sigmoid", cv=lipat,
                                ensemble=False).fit(latih[peubah], yl)
    p_sk = sk.predict_proba(uji[peubah])[:, 1]
    print(f"(3) Platt kita untuk model balanced: a = {a:.4f}, b = {b:.4f}")
    print(f"    maks beda dengan CalibratedClassifierCV < 1e-4: "
          f"{np.abs(p_kita - p_sk).max() < 1e-4}")
    print("(4) sesudah rekalibrasi (cv = 5):")
    print("    model       sigmoid ECE  log-loss  isotonic ECE  log-loss")
    for nama, opsi, peubah in MODEL:
        baris = f"    {nama:10s}"
        for metode in ("sigmoid", "isotonic"):
            c = CalibratedClassifierCV(model_pasien(peubah, **opsi),
                                       method=metode, cv=5)
            q = c.fit(latih[peubah], yl).predict_proba(uji[peubah])[:, 1]
            baris += f"   {ece(y, q):.4f}      {log_loss(y, q):.4f}"
        print(baris)
    print("(5) biaya per pasien pada ambang 1/6:")
    for nama, opsi, peubah in (MODEL[0], MODEL[2]):
        m = model_pasien(peubah, **opsi).fit(latih[peubah], yl)
        p = m.predict_proba(uji[peubah])[:, 1]
        c = CalibratedClassifierCV(model_pasien(peubah, **opsi),
                                   method="sigmoid", cv=5)
        q = c.fit(latih[peubah], yl).predict_proba(uji[peubah])[:, 1]
        print(f"    {nama:10s}: {biaya(y, p, 1 / 6):.4f}, sesudah sigmoid "
              f"{biaya(y, q, 1 / 6):.4f}")
