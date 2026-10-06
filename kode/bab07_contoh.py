"""Memeriksa setiap bilangan Contoh Soal Bab 7."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit, logit

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)
m = mle_mini()

# Contoh Soal 7.1: peluang titik baru
assert r(expit(m @ [1, 4, 1])) == 0.75 and r(expit(m @ [1, 2, 0])) == 0.5

# Contoh Soal 7.2: informasi Fisher dan kovarians lewat 16 I
I16 = np.array([[20, 72, 32], [72, 314, 152], [32, 152, 82]])
p = expit(X @ m)
assert np.allclose(16 * X.T @ (X * (p * (1 - p))[:, None]), I16)
c11 = 314 * 82 - 152 ** 2
c12 = -(72 * 82 - 152 * 32)
c13 = 72 * 152 - 314 * 32
assert (c11, c12, c13) == (2644, -1040, 896)
det16 = 20 * c11 + 72 * c12 + 32 * c13
assert det16 == 6672 == 16 * 417
adj = np.array([[2644, -1040, 896], [-1040, 616, -736], [896, -736, 1096]])
assert np.allclose(I16 @ adj, det16 * np.eye(3))
assert np.allclose(kovarians(m, X), adj / 417)
assert [r(np.sqrt(v / 417)) for v in (2644, 616, 1096)] == [2.5180, 1.2154,
                                                            1.6212]

# Contoh Soal 7.3: metode delta di (4, 1)
x0 = np.array([1, 4, 1])
assert x0 @ adj @ x0 == 1180
assert 2644 + 16 * 616 + 1096 + 2 * 4 * -1040 + 2 * 896 + 2 * 4 * -736 == 1180
var = 1180 / 417
assert (r(var), r(np.sqrt(var))) == (2.8297, 1.6822)
lo, hi = np.log(3) - 1.96 * np.sqrt(var), np.log(3) + 1.96 * np.sqrt(var)
assert (r(lo), r(hi)) == (-2.1985, 4.3957)
assert (r(expit(lo)), r(expit(hi))) == (0.0999, 0.9878)

# Contoh Soal 7.4: selang simetris keluar dari [0, 1]
se_p = 0.1875 * np.sqrt(var)
assert r(se_p) == 0.3154 and r(1.96 * se_p) == 0.6182
assert (r(0.75 - 1.96 * se_p), r(0.75 + 1.96 * se_p)) == (0.1318, 1.3682)

# Contoh Soal 7.5: garis batas untuk ambang t
assert r(logit(0.25)) == -1.0986
assert r(2 + logit(0.25) / np.log(3)) == 1.0

# Contoh Soal 7.6: ambang dari biaya
assert 1 / (1 + 4) == 0.2 and r(logit(0.2)) == -1.3863
assert r(2 + logit(0.2) / np.log(3)) == 0.7381
print("Contoh Soal Bab 7: semua bilangan cocok")
