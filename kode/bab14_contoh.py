"""Memeriksa setiap bilangan Contoh Soal Bab 14."""
import numpy as np
from scipy.special import expit, logit, softmax

from bab01_data import BENIH
from bab14_perluasan import data_jarang


def r(v, n=4):
    return round(float(v), n)


X, y = data_jarang(20000, BENIH)
m, m1 = 20000, 954
m0 = m - m1
assert y.sum() == m1

# Contoh Soal 14.1: model nol berbobot balanced
s1, s0 = m / (2 * m1), m / (2 * m0)
assert r(s1 * m1 / (s1 * m1 + s0 * m0)) == 0.5
assert m1 / m == 0.0477 and r(logit(m1 / m)) == -2.9939

# Contoh Soal 14.2: pergeseran w0
assert r(np.log(m0 / m1)) == 2.9939 and r(s1 / s0, 2) == 19.96
assert m0 == 19046

# Contoh Soal 14.3: koreksi kasus-kontrol
r0 = m1 / m0
assert r(r0) == 0.0501 and r(np.log(1 / r0)) == 2.9939
assert r(-0.5638 - 2.9939) == -3.5577

# Contoh Soal 14.4: ambang setara
assert r(expit(-np.log(m0 / m1)), 5) == 0.0477

# Contoh Soal 14.5: softmax
z = np.array([1.0, 0.0, -1.0])
e = np.exp(z)
assert (r(e[0]), r(e[2]), r(e.sum())) == (2.7183, 0.3679, 4.0862)
assert [r(v) for v in softmax(z)] == [0.6652, 0.2447, 0.0900]
assert r(np.log(e.sum()) - z[1]) == 1.4076 and r(-np.log(0.2447)) == 1.4077

# Contoh Soal 14.6: dua kelas = logistik; gradien di x = (1, 2, 1)
for z1, z0 in [(0.3, -0.2), (2.0, 1.5)]:
    assert np.isclose(softmax([z0, z1])[1], expit(z1 - z0))
g = np.outer([1, 2, 1], softmax(z) - np.array([0, 1, 0]))
assert [r(v) for v in g[1]] == [1.3305, -1.5105, 0.1801]
assert [r(v) for v in g[0]] == [0.6652, -0.7553, 0.0900]
assert np.allclose(g.sum(axis=1), 0)
print("Contoh Soal Bab 14: semua bilangan cocok")
