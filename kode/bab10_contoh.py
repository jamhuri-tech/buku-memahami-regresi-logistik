"""Memeriksa setiap bilangan Contoh Soal Bab 10."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians
from bab10_inferensi import log_kem, mle, profil, selang_profil


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)
m = mle_mini()
C = kovarians(m, X)
se = np.sqrt(np.diag(C))
ln3 = np.log(3)

# Contoh Soal 10.1: Var(U) = I
p = expit(X @ m)
I = X.T @ (X * (p * (1 - p))[:, None])
assert np.allclose(np.linalg.inv(I), C)

# Contoh Soal 10.2: Wald untuk w1 dan w2
z1, z2 = m[1] / se[1], m[2] / se[2]
assert (r(z1), r(2 * norm.sf(z1))) == (0.9039, 0.366)
assert (r(z2), r(2 * norm.sf(-z2))) == (-0.6777, 0.498)
assert r(1.96 * 1.2154) == 2.3822
assert (r(ln3 - 2.3822), r(ln3 + 2.3822)) == (-1.2836, 3.4808)
assert (r(np.exp(-1.2836), 3), r(np.exp(3.4808), 1)) == (0.277, 32.5)

# Contoh Soal 10.3: Wald gabungan, det blok = 320 * 417
assert 616 * 1096 - 736 ** 2 == 133440 == 320 * 417
assert 1096 - 2 * 736 + 616 == 240 and F(240, 320) == F(3, 4)
W = m[1:] @ np.linalg.solve(C[1:, 1:], m[1:])
assert np.isclose(W, 0.75 * ln3 ** 2) and r(W) == 0.9052
assert r(ln3 ** 2) == 1.2069
assert r(chi2.sf(W, 2)) == 0.636 and np.isclose(chi2.sf(W, 2),
                                                np.exp(-W / 2))

# Contoh Soal 10.4: rasio kemungkinan
l1, l0 = log_kem(m, X, y), log_kem(np.zeros(3), X, y)
assert (r(l1), r(l0)) == (-3.6356, -4.1589)
G = 2 * (l1 - l0)
assert np.isclose(G, 6 * ln3 - 8 * np.log(2)) and r(G) == 1.0465
assert np.isclose(G, np.log(729 / 256))
assert r(chi2.sf(G, 2)) == 0.5926 and r(np.exp(-G / 2)) == 0.5926
assert (r(-2 * l1), r(-2 * l0)) == (7.2713, 8.3178)
lx1 = log_kem(mle(X[:, :2], y), X[:, :2], y)
lx2 = log_kem(mle(X[:, [0, 2]], y), X[:, [0, 2]], y)
assert (r(lx1), r(lx2)) == (-3.8950, -4.1060)
assert (r(2 * (l1 - lx1)), r(2 * (l1 - lx2))) == (0.5188, 0.9408)

# Contoh Soal 10.5: uji skor = U^T (langkah Newton pertama)
U = X.T @ (y - 0.5)
assert np.allclose(U, [0, 1.5, 0.5])
s = np.linalg.solve(X.T @ X / 4, U)
assert np.allclose(s, [-2, 1, -1])
assert np.isclose(U @ s, 1) and r(chi2.sf(1, 2)) == r(np.exp(-0.5)) == 0.6065

# Contoh Soal 10.6: selang profil w1 dan w2
a1, b1 = selang_profil(X, y, 1)
a2, b2 = selang_profil(X, y, 2)
assert (r(a1), r(b1), r(a2), r(b2)) == (-1.0869, 4.4200, -5.8454,
                                        1.9142)
for j, v in [(1, a1), (1, b1), (2, a2), (2, b2)]:
    assert r(2 * (l1 - profil(X, y, j, v))) == 3.8415
assert (r(-ln3 - 1.96 * se[2]), r(-ln3 + 1.96 * se[2])) == (-4.2762, 2.0789)

# Contoh Soal 10.7: AIC dan BIC
assert r(np.log(6)) == 1.7918
assert (r(-2 * l0 + 2), r(-2 * l0 + np.log(6))) == (10.3178, 10.1095)
assert (r(-2 * l1 + 6), r(-2 * l1 + 3 * np.log(6))) == (13.2713,
                                                        12.6465)
assert r(-2 * lx1 + 4) == 11.7900
print("Contoh Soal Bab 10: semua bilangan cocok")
