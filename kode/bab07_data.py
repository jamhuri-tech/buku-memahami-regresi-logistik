"""Bab 7: data yang terpisah.

jam_terpisah(): jam belajar dengan label diubah sehingga semua yang
    belajar paling banyak 5 jam tidak lulus dan sisanya lulus
    (pemisahan lengkap).
jam_kuasi(): seperti itu, tetapi batasnya di 5 jam dan dua mahasiswa
    yang belajar tepat 5 jam berbeda label (pemisahan kuasi).
gumpalan_terpisah(): 40 titik di bidang dari dua gumpalan yang dapat
    dipisahkan sebuah garis.
"""
import numpy as np

from bab04_data import BENIH, JAM


def jam_terpisah():
    return JAM.copy(), (JAM >= 6).astype(int)


def jam_kuasi():
    y = (JAM >= 5).astype(int)
    y[6] = 0                       # dua mahasiswa 5 jam: 0 dan 1
    return JAM.copy(), y


def gumpalan_terpisah(benih=BENIH):
    rng = np.random.default_rng(benih)
    X = np.vstack([rng.normal((-1.5, -0.5), 0.6, size=(20, 2)),
                   rng.normal((1.5, 0.5), 0.6, size=(20, 2))])
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    return X, y


if __name__ == "__main__":
    for nama, (x, y) in (("terpisah", jam_terpisah()),
                         ("kuasi   ", jam_kuasi())):
        print(f"jam {nama}: lulus = {''.join(map(str, y))}")
