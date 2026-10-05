"""Memeriksa setiap bilangan Contoh Soal Bab 15."""
import numpy as np

from bab10_inferensi import mle
from bab15_data import baca_jantung, rancangan


def r(v, n=4):
    return round(float(v), n)


d = baca_jantung()
X, y, nama = rancangan(d)
th = mle(X, y)
j = nama.index("pembuluh")

# Contoh Soal 15.1: EPV
assert (y.sum(), len(y) - y.sum(), X.shape[1] - 1) == (137, 160, 17)
assert r(137 / 17, 2) == 8.06 and r(10 * 17) == 170

# Contoh Soal 15.2: odds ratio pembuluh untuk dua pembuluh
assert r(th[j]) == 1.3107 and r(np.exp(th[j]), 2) == 3.71
assert r(np.exp(2 * th[j]), 2) == 13.76 and r(3.71 ** 2, 2) == 13.76

# Contoh Soal 15.3: susut seragam
assert r(0.7869 * 1.3107) == 1.0314 and r(np.exp(1.0314), 2) == 2.80
print("Contoh Soal Bab 15: semua bilangan cocok")
