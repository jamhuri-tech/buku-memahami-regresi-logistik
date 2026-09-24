"""Bab 14: data yang batas keputusannya tidak lurus.

cincin(n): x seragam di [-2, 2]^2, logit = 4 (1.2 - |x|^2); model
    regresi logistik derajat dua yang tepat.
bulan(n): dua bulan sabit berderau dari make_moons.
bentuk_u(n): satu peubah x di [0, 10], logit = -1 + 0.25 (x - 5)^2;
    risiko tinggi di kedua ujung.
"""
import numpy as np
from scipy.special import expit
from sklearn.datasets import make_moons

from bab04_data import BENIH


def cincin(n, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.uniform(-2, 2, size=(n, 2))
    p = expit(4 * (1.2 - (X ** 2).sum(axis=1)))
    return X, (rng.random(n) < p).astype(int), p


def bulan(n, benih=BENIH):
    return make_moons(n, noise=0.3, random_state=benih % 2**32)


def bentuk_u(n, benih=BENIH):
    rng = np.random.default_rng(benih)
    x = rng.uniform(0, 10, n)
    p = expit(-1 + 0.25 * (x - 5) ** 2)
    return x[:, None], (rng.random(n) < p).astype(int), p


if __name__ == "__main__":
    for nama, f in (("cincin", cincin), ("bulan", bulan),
                    ("bentuk u", bentuk_u)):
        hasil = f(300)
        print(f"{nama:8s}: n = 300, kelas 1 = {hasil[1].sum()}")
