"""Memeriksa setiap bilangan Contoh Soal Bab 11."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians
from bab10_inferensi import mle


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)
m = mle_mini()
se = np.sqrt(np.diag(kovarians(m, X)))

# Contoh Soal 11.1: skala fitur
assert r(np.exp(2 * m[1])) == 9.0
assert (r(np.exp(-2 * 1.2836)), r(np.exp(2 * 3.4808), 1)) == (0.0768,
                                                               1055.3)
assert sum((Xf[:, 1] - 1.5) ** 2) == 9.5 and r(np.sqrt(9.5 / 5)) == 1.3784
assert r(np.std(Xf[:, 0], ddof=1)) == 1.8708
assert r(1.0986 * 1.8708) == 2.0553 and r(np.exp(2.0553), 3) == 7.809
assert r(-1.0986 * 1.3784) == -1.5143 and r(np.exp(-1.5143)) == 0.22

# Contoh Soal 11.2: penyesuaian, langkah Newton pertama model x1 saja
A = X[:, :2]
U = A.T @ (y - 0.5)
assert np.allclose(U, [0, 1.5])
assert np.allclose(np.linalg.inv(A.T @ A) * 105, [[91, -21], [-21, 6]])
s = 4 * np.linalg.solve(A.T @ A, U)
assert np.allclose(s, [-1.2, 36 / 105]) and r(36 / 105) == 0.3429
w1 = mle(A, y)
assert (r(w1[1]), r(np.exp(w1[1]))) == (0.3613, 1.4352)
assert r(np.corrcoef(Xf.T)[0, 1]) == 0.8919

# Contoh Soal 11.3: tabel 2 x 2
assert 30 * 35 / (20 * 15) == 3.5 and r(np.log(3.5)) == 1.2528
assert r(1/30 + 1/20 + 1/15 + 1/35) == 0.1786
assert r(np.sqrt(1/30 + 1/20 + 1/15 + 1/35)) == 0.4226
lo, hi = np.log(3.5) - 1.96 * 0.4226, np.log(3.5) + 1.96 * 0.4226
assert (r(np.exp(lo)), r(np.exp(hi))) == (1.5288, 8.0129)
assert r(15 / 35) == 0.4286

# Contoh Soal 11.4: interaksi
w0 = logit(10 / 50)
assert r(w0) == -1.3863
assert r(logit(20 / 40) - w0) == 1.3863
assert r(logit(45 / 50)) == 2.1972
assert r(logit(45 / 50) - w0 - 2 * 1.3862944) == 0.8109
assert r(np.exp(logit(45 / 50) - w0 - 2 * np.log(4))) == 2.25

# Contoh Soal 11.5: AME
p = expit(X @ m)
d = p * (1 - p)
assert 4 * F(3, 16) + 2 * F(1, 4) == F(5, 4)
assert np.isclose(d.sum(), 1.25)
assert r(np.log(3) * 5 / 24) == 0.2289 and r(np.log(3) / 4) == 0.2747
assert r(m[1] * d.mean()) == 0.2289

# Contoh Soal 11.6: odds ratio lawan rasio risiko
def p1(p0, OR):
    o = OR * p0 / (1 - p0)
    return o / (1 + o)
assert r(p1(0.2, 3)) == 0.4286 and r(p1(0.2, 3) / 0.2, 2) == 2.14
assert r(p1(0.01, 3)) == 0.0294 and r(p1(0.01, 3) / 0.01, 2) == 2.94
assert p1(0.25, 3) == 0.5        # data mini: 1/4 -> 1/2

# Contoh Soal 11.7: ketidakruntuhan
assert r((0.5 / 0.5) / (0.2 / 0.8)) == 4.0 and r((0.8 / 0.2) / 1) == 4.0
assert r((0.65 / 0.35) / (0.35 / 0.65)) == 3.4490
print("Contoh Soal Bab 11: semua bilangan cocok")
