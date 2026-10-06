"""Memeriksa setiap bilangan Contoh Soal Bab 13."""
from fractions import Fraction as F

import numpy as np

from bab08_performa import peluang_mini
from bab13_validasi import delong, komponen_delong, m_riley


def r(v, n=4):
    return round(float(v), n)


# Contoh Soal 13.1: ringkasan validasi silang
a = np.array([0.82, 0.76, 0.85, 0.79, 0.78])
assert r(a.mean()) == 0.80 and r(np.var(a, ddof=1), 5) == 0.00125
assert r(np.std(a, ddof=1)) == 0.0354 and r(0.0354 / np.sqrt(5)) == 0.0158

# Contoh Soal 13.2: galat baku bootstrap
w = np.array([1.0, 1.2, 1.3, 1.5, 1.5])
assert r(w.mean()) == 1.3 and r(np.var(w, ddof=1)) == 0.045
assert r(np.std(w, ddof=1)) == 0.2121

# Contoh Soal 13.3: koreksi optimisme
o = np.array([0.05, 0.03, 0.04, 0.06, 0.02])
assert r(o.mean()) == 0.04 and r(0.87 - o.mean()) == 0.83

# Contoh Soal 13.4: DeLong pada data mini (ada seri)
p, y = peluang_mini()
v10, v01 = komponen_delong(y, p)
assert np.allclose(v10, [5 / 6, 2 / 3, 2 / 3])
assert np.allclose(v01, [1, 1, 1 / 6])
s10 = (F(1, 9) ** 2 + 2 * F(1, 18) ** 2) / 2
s01 = (2 * F(5, 18) ** 2 + F(10, 18) ** 2) / 2
assert (s10, s01) == (F(1, 108), F(25, 108))
assert np.isclose(np.var(v10, ddof=1), 1 / 108)
assert np.isclose(np.var(v01, ddof=1), 25 / 108)
var = s10 / 3 + s01 / 3
assert var == F(13, 162)
auc, se = delong(y, p)
assert np.isclose(se, np.sqrt(13 / 162)) and r(se) == 0.2833
assert r(1.96 * 0.2833) == 0.5553
assert (r(auc - 1.96 * se, 3), r(auc + 1.96 * se, 3)) == (0.167, 1.277)

# Contoh Soal 13.5: banyak sampel Riley
assert r(np.log(1 - 0.2 / 0.9)) == -0.2513
assert r(m_riley(10, 0.2), 1) == 397.9
assert r(398 * 0.3, 1) == 119.4 and r(119.4 / 10, 2) == 11.94
print("Contoh Soal Bab 13: semua bilangan cocok")
