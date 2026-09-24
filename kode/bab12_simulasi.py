"""Bab 12: cakupan selang kepercayaan dan bias taksiran, diukur.

Untuk setiap skenario, label dibangkitkan ulang R kali dari model
yang parameternya diketahui, dengan matriks rancangan tetap. Data yang
terpisah (MLE tidak ada) dicatat dan dikeluarkan dari perhitungan
cakupan; taksiran Firth dihitung untuk semua ulangan.
"""
import numpy as np
from scipy.special import expit

from bab04_data import B_BENAR, BENIH, W_BENAR, dua_peubah, jam_belajar
from bab06_turunan import rancang
from bab07_deteksi import pemisahan
from bab12_inferensi import firth, galat_baku, mle, selang_profil


def skenario(X, benar, j, R, rng):
    catat = {"pisah": 0, "wald": 0, "profil": 0, "bawah": 0, "atas": 0,
             "mle": [], "firth": []}
    for _ in range(R):
        y = (rng.random(len(X)) < expit(X @ benar)).astype(float)
        catat["firth"].append(firth(X, y)[j])
        if pemisahan(X[:, 1:], y) != "tumpang tindih":
            catat["pisah"] += 1
            continue
        theta, H = mle(X, y)
        se = galat_baku(H)[j]
        catat["mle"].append(theta[j])
        catat["bawah"] += theta[j] + 1.96 * se < benar[j]
        catat["atas"] += theta[j] - 1.96 * se > benar[j]
        lo, hi = selang_profil(X, y, j, theta, se)
        catat["profil"] += lo <= benar[j] <= hi
    ada = R - catat["pisah"]
    catat["wald"] = 1 - (catat["bawah"] + catat["atas"]) / ada
    catat["profil"] /= ada
    return catat


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    x, _ = jam_belajar()
    X2, _ = dua_peubah()
    daftar = [
        ("jam belajar, n = 16", rancang(x), np.array([-3.5903, 0.6393])),
        ("garis, n = 40, w = 3", rancang(np.linspace(-2, 2, 40)),
         np.array([0.0, 3.0])),
        ("dua peubah, n = 200", rancang(X2), np.r_[B_BENAR, W_BENAR]),
    ]
    R = 1000
    for nama, X, benar in daftar:
        c = skenario(X, benar, 1, R, rng)
        ada = R - c["pisah"]
        print(f"{nama} ({R} ulangan), w benar = {benar[1]:g}:")
        print(f"  terpisah {c['pisah'] / R:.3f}; cakupan Wald "
              f"{c['wald']:.3f}, profil {c['profil']:.3f}")
        print(f"  Wald meleset: di bawah w {c['bawah'] / ada:.3f}, "
              f"di atas w {c['atas'] / ada:.3f}")
        print(f"  rata-rata w: MLE {np.mean(c['mle']):.3f}, "
              f"Firth {np.mean(c['firth']):.3f}")
