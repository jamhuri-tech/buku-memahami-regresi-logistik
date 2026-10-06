"""Memeriksa setiap bilangan Contoh Soal Bab 15."""
import numpy as np

from bab10_inferensi import mle
from bab15_data import baca_jantung, rancangan


def r(v, n=4):
    return round(float(v), n)


d = baca_jantung()
X, y, nama = rancangan(d)
w = mle(X, y)
j = nama.index("pembuluh")

# Contoh Soal 15.1: ukuran X dan EPV
assert X.shape == (297, 18) and np.all(X[:, 0] == 1)
assert (y.sum(), len(y) - y.sum(), X.shape[1] - 1) == (137, 160, 17)
assert r(137 / 17, 2) == 8.06 and r(10 * 17) == 170

# Contoh Soal 15.2: odds ratio pembuluh untuk dua pembuluh
assert j == 15
assert r(w[j]) == 1.3107 and r(np.exp(w[j]), 2) == 3.71
assert r(np.exp(2 * w[j]), 2) == 13.76 and r(3.71 ** 2, 2) == 13.76

# Contoh Soal 15.3: susut seragam
assert r(0.7828 * 1.3107) == 1.026 and r(np.exp(1.0260), 2) == 2.79
print("Contoh Soal Bab 15: semua bilangan cocok")
