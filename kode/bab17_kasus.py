"""Bab 17: sampel kasus-kontrol dan koreksi intersep.

Semua kasus (kelas 1) diambil, kontrol (kelas 0) diambil acak sebanyak
k kali banyaknya kasus. Bobot peubah tetap sahih; intersep bergeser
sebesar log(r1 / r0), dengan r1 dan r0 fraksi kasus dan kontrol yang
diambil. Peluang yang dikoreksi dinilai pada populasi uji baru.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab16_kalibrasi import ece
from bab17_data import populasi

if __name__ == "__main__":
    X, y, _ = populasi()
    Xu, yu, _ = populasi(benih=1)
    rng = np.random.default_rng(1)
    kasus, kontrol = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    penuh = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    print(f"populasi penuh (n = {len(y)}): b = {penuh.params[0]:.3f}, "
          f"w1 = {penuh.params[1]:.3f} (SE {penuh.bse[1]:.3f})")
    print(f"proporsi kelas 1 di populasi uji: {yu.mean():.4f}")
    for k in (1, 5):
        ambil = rng.choice(kontrol, size=k * len(kasus), replace=False)
        i = np.r_[kasus, ambil]
        h = sm.Logit(y[i], sm.add_constant(X[i])).fit(disp=0)
        r0 = len(ambil) / len(kontrol)
        koreksi = np.log(1 / r0)                 # r1 = 1
        b_baru = h.params[0] - koreksi
        print(f"kasus : kontrol = 1 : {k} (n = {len(i)}):")
        print(f"    b = {h.params[0]:.3f}, log(r1/r0) = {koreksi:.3f}, "
              f"b dikoreksi = {b_baru:.3f}")
        print(f"    w1 = {h.params[1]:.3f} (SE {h.bse[1]:.3f}), "
              f"SE / SE penuh = {h.bse[1] / penuh.bse[1]:.2f}")
        z = sm.add_constant(Xu) @ h.params
        p_mentah, p_koreksi = expit(z), expit(z - koreksi)
        print(f"    uji: rata p mentah {p_mentah.mean():.4f}, "
              f"dikoreksi {p_koreksi.mean():.4f}")
        print(f"    ECE uji: mentah {ece(yu, p_mentah):.4f}, "
              f"dikoreksi {ece(yu, p_koreksi):.4f}")
