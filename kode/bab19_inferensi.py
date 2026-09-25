"""Bab 19: model untuk tafsiran, dengan statsmodels.

Semua 13 peubah masuk, dengan satuan yang dipilih sebelum melihat
hasil: usia per 10 tahun, tensi per 10 mmHg, kolesterol per 50 mg/dl,
nadi maksimum per 10 denyut/menit. Rujukan kategori: nyeri 'tanpa
gejala', ekg 'normal', lereng 'naik', thal 'normal'. Odds ratio
dilaporkan dengan selang Wald; peubah kategori diuji bersama dengan
uji rasio kemungkinan.
"""
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2

from bab06_turunan import rancang
from bab07_deteksi import pemisahan
from bab19_data import baca_jantung

RUMUS = ("sakit ~ I(usia / 10) + pria + "
         "C(nyeri, Treatment('tanpa gejala')) + I(tensi / 10) + "
         "I(kolesterol / 50) + gula + C(ekg, Treatment('normal')) + "
         "I(nadi_maks / 10) + angina_latih + depresi_st + "
         "C(lereng_st, Treatment('naik')) + pembuluh + "
         "C(thal, Treatment('normal'))")


def nama_pendek(k):
    for awal in ("C(nyeri", "C(ekg", "C(lereng_st", "C(thal"):
        if k.startswith(awal):
            return awal[2:] + ": " + k.split("T.")[1].rstrip("]")
    return k.replace("I(", "").replace(")", "")


if __name__ == "__main__":
    d = baca_jantung()
    h = smf.logit(RUMUS, d).fit(disp=0)
    X = h.model.exog[:, 1:]
    print(f"pemisahan: {pemisahan(X, d.sakit.to_numpy())}; "
          f"n = {int(h.nobs)}, parameter {len(h.params)}")
    ci = h.conf_int()
    print("peubah                         OR     selang 95%")
    for k in h.params.index[1:]:
        print(f"{nama_pendek(k):28s} {np.exp(h.params[k]):6.2f}   "
              f"[{np.exp(ci.loc[k, 0]):5.2f}, {np.exp(ci.loc[k, 1]):6.2f}]")
    print("uji rasio kemungkinan per kelompok kategori:")
    for nama, bagian in (("nyeri", "C(nyeri, Treatment('tanpa gejala')) + "),
                         ("ekg", "C(ekg, Treatment('normal')) + "),
                         ("lereng_st", "C(lereng_st, Treatment('naik')) + "),
                         ("thal", " + C(thal, Treatment('normal'))")):
        kecil = smf.logit(RUMUS.replace(bagian, ""), d).fit(disp=0)
        G = 2 * (h.llf - kecil.llf)
        db = len(h.params) - len(kecil.params)
        print(f"    {nama:10s} G = {G:6.2f}, db {db}, nilai-p "
              f"{chi2.sf(G, db):.4f}")
