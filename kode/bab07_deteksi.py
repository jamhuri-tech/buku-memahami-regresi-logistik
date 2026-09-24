"""Bab 7: mendeteksi pemisahan dengan program linear.

pemisahan (Listing 7.1) memakai dua program linear. Yang pertama
mencari theta dengan margin terkecil t > 0 (pemisahan lengkap); yang
kedua mencari theta dengan semua margin >= 0 dan jumlah margin > 0
(pemisahan kuasi). Bagian utama memeriksa lima data, lalu mengukur
peluang data berlabel acak terpisah sebagai fungsi banyaknya peubah,
dan membandingkannya dengan rumus Cover (1965).
"""
import numpy as np
from scipy.optimize import linprog
from scipy.special import comb

from bab04_data import BENIH, dua_peubah, jam_belajar
from bab06_turunan import rancang
from bab07_data import gumpalan_terpisah, jam_kuasi, jam_terpisah


def pemisahan(X, y, tol=1e-9):
    A = (2 * y - 1)[:, None] * rancang(X)   # baris s_i x_i
    n, m = A.shape
    kotak = [(-1, 1)] * m
    # (1) maksimumkan t dengan syarat A theta >= t
    r = linprog(np.r_[np.zeros(m), -1.0],
                A_ub=np.c_[-A, np.ones(n)], b_ub=np.zeros(n),
                bounds=kotak + [(None, 1)])
    if -r.fun > tol:
        return "lengkap"
    # (2) maksimumkan jumlah margin dengan syarat A theta >= 0
    r = linprog(-A.sum(axis=0), A_ub=-A, b_ub=np.zeros(n),
                bounds=kotak)
    return "kuasi" if -r.fun > tol else "tumpang tindih"


def peluang_cover(n, D):
    k = np.arange(D)
    return min(1.0, 2 * comb(n - 1, k).sum() / 2.0 ** n)


def percobaan_cover(n=60, ulang=200, daftar_d=(5, 15, 20, 25, 29, 35,
                                              40, 50)):
    """Frekuensi data berlabel acak yang terpisah, per banyak peubah."""
    rng = np.random.default_rng(BENIH)
    hasil = {}
    for d in daftar_d:
        hit = 0
        for _ in range(ulang):
            X = rng.normal(size=(n, d))
            y = rng.integers(0, 2, size=n)
            hit += pemisahan(X, y) != "tumpang tindih"
        hasil[d] = hit / ulang
    return hasil


if __name__ == "__main__":
    data = [("jam belajar", jam_belajar()),
            ("jam terpisah", jam_terpisah()),
            ("jam kuasi", jam_kuasi()),
            ("dua peubah", dua_peubah()),
            ("gumpalan terpisah", gumpalan_terpisah())]
    for nama, (X, y) in data:
        print(f"{nama:18s}: {pemisahan(X, y)}")
    n, ulang = 60, 200
    print(f"label acak, n = {n}, {ulang} ulangan per d:")
    print("   d   terpisah   rumus Cover")
    for d, frek in percobaan_cover(n, ulang).items():
        print(f"  {d:2d}   {frek:8.3f}   {peluang_cover(n, d + 1):11.3f}")
