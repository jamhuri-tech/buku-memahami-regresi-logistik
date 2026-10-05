"""Memeriksa setiap bilangan Contoh Soal Bab 14."""
import numpy as np
from scipy.special import expit, logit, softmax


def r(v, n=4):
    return round(float(v), n)


# Contoh Soal 14.1: model nol berbobot balanced
n, n1 = 20000, 981
n0 = n - n1
s1, s0 = n / (2 * n1), n / (2 * n0)
assert r(s1 * n1 / (s1 * n1 + s0 * n0)) == 0.5
assert r(logit(0.05)) == -2.9444 and r(logit(n1 / n)) == -2.9646

# Contoh Soal 14.2: pergeseran intersep
assert r(np.log(n0 / n1)) == 2.9646 and r(s1 / s0, 2) == 19.39

# Contoh Soal 14.3: koreksi kasus-kontrol
r0 = n1 / n0
assert r(r0) == 0.0516 and r(np.log(1 / r0)) == 2.9646
assert r(-0.5002 - 2.9646) == -3.4648

# Contoh Soal 14.4: ambang setara
assert r(expit(-np.log(n0 / n1)), 5) == 0.04905 and n1 / n == 0.04905

# Contoh Soal 14.5: softmax
z = np.array([1.0, 0.0, -1.0])
e = np.exp(z)
assert (r(e[0]), r(e[2]), r(e.sum())) == (2.7183, 0.3679, 4.0862)
assert [r(v) for v in softmax(z)] == [0.6652, 0.2447, 0.0900]
assert r(np.log(e.sum()) - z[1]) == 1.4076 and r(-np.log(0.2447)) == 1.4077

# Contoh Soal 14.6: dua kelas = logistik
for z1, z0 in [(0.3, -0.2), (2.0, 1.5)]:
    assert np.isclose(softmax([z0, z1])[1], expit(z1 - z0))

# Contoh Soal 14.7: gradien satu titik, y = kelas 1, x = (1, 2)
g = np.outer([1, 2], softmax(z) - np.array([0, 1, 0]))
assert [r(v) for v in g.ravel()] == [0.6652, -0.7553, 0.0900, 1.3305,
                                     -1.5105, 0.1801]
print("Contoh Soal Bab 14: semua bilangan cocok")
