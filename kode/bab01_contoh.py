"""Memeriksa setiap bilangan Contoh Soal Bab 1."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang


def r(v, n=4):
    return round(float(v), n)


X, y = data_mini()
A = rancang(X)

# Contoh Soal 1.1: dari tabel ke matriks
assert A.shape == (6, 3) and list(A[:, 0]) == [1] * 6
assert list(A[3]) == [1, 4, 2] and list(y) == [0, 0, 1, 1, 1, 0]

# Contoh Soal 1.2: satu sampel, simbolik lalu angka (i = 3, w = 0)
p = 0.5
assert r(-np.log(p)) == 0.6931
g = (p - 1) * np.array([1.0, 3.0, 0.0])
assert list(g) == [-0.5, -1.5, 0.0]
assert list(np.zeros(3) - 0.5 * g) == [0.25, 0.75, 0.0]

# Contoh Soal 1.3: turunan fungsi logistik
for v in (-1.0, 0.0, 1.0):
    h = 1e-6
    beda = (expit(v + h) - expit(v - h)) / (2 * h)
    assert abs(beda - expit(v) * (1 - expit(v))) < 1e-9
assert r(expit(1) * (1 - expit(1))) == 0.1966

# Contoh Soal 1.4: z = X w untuk w = (-2, 1, -1)
z = A @ np.array([-2.0, 1.0, -1.0])
assert list(z) == [-1, -1, 1, 0, 0, 1]
assert (r(np.e), r(expit(1)), r(expit(-1))) == (2.7183, 0.7311, 0.2689)

# Contoh Soal 1.5: odds dan odds ratio
assert r(np.exp(-1)) == 0.3679 and r(np.exp(1)) == 2.7183

# Contoh Soal 1.6: tafsiran MLE
w = mle_mini()
assert np.allclose(w, np.log(3) * np.array([-2, 1, -1]))
assert (r(w[0]), r(w[1]), r(w[2])) == (-2.1972, 1.0986, -1.0986)
assert (r(np.exp(w[1])), r(np.exp(w[2])), r(np.exp(w[0]))) == (3.0, 0.3333,
                                                                0.1111)
assert r(expit(w @ [1, 4, 1])) == 0.75
assert r(expit(w @ [1, 2, 0])) == 0.5

# Contoh Soal 1.7: batas keputusan x2 = x1 - 2
assert list(A @ np.array([-2, 1, -1]) >= 0) == [False, False, True, True,
                                                 True, True]
print("Contoh Soal Bab 1: semua bilangan cocok")
