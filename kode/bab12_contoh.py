"""Memeriksa setiap bilangan Contoh Soal Bab 12."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab10_inferensi import mle, terpisah
from bab12_diagnostik import cook, leverage, residu, vif


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)
m = mle_mini()
p = expit(X @ m)
rp, rd = residu(y, p)
ADJ = np.array([[2644, -1040, 896], [-1040, 616, -736],
                [896, -736, 1096]])

# Contoh Soal 12.1: residu mahasiswa ke-6 dan jumlah kuadrat
assert r(rp[5]) == r(-np.sqrt(3)) == -1.7321
assert r(rd[5]) == r(-np.sqrt(2 * np.log(4))) == -1.6651
assert r(2 * np.log(4)) == 2.7726
assert r((rp ** 2).sum()) == 6.0
assert F(1, 3) * 3 + 1 + 1 + 3 == 6
assert np.isclose((rd ** 2).sum(), 20 * np.log(2) - 6 * np.log(3))
assert r((rd ** 2).sum()) == 7.2713

# Contoh Soal 12.2: leverage dan Cook mahasiswa ke-3
x3 = X[2]
assert x3 @ ADJ @ x3 == 1948 == 2644 + 9 * 616 - 6 * 1040
h, _ = leverage(X, p)
assert F(3, 16) * F(1948, 417) == F(487, 556) and r(h[2]) == 0.8759
assert r(F(487, 556)) == 0.8759
D3 = F(1, 3) * F(487, 556) / (3 * (1 - F(487, 556)) ** 2)
assert D3 == F(270772, 42849) and r(D3) == 6.3192
assert r(cook(y, p, h, 3)[2]) == 6.3192
assert r(h.sum()) == 3.0
q = [xi @ ADJ @ xi for xi in X]
assert q == [1180, 892, 1948, 372, 804, 1084]

# Contoh Soal 12.3: dfbeta mahasiswa ke-4
x4 = X[3]
assert list(ADJ @ x4) == [276, -48, 144]
assert F(372, 1668) == 1 - F(108, 139) and 1668 == 4 * 417
satu = [F(v, 648) for v in (276, -48, 144)]
assert satu == [F(23, 54), F(-2, 27), F(2, 9)]
assert [r(v) for v in satu] == [0.4259, -0.0741, 0.2222]
k = np.arange(6) != 3
t = m - mle(X[k], y[k])
assert [r(v) for v in t] == [0.5534, -0.1259, 0.2682]
assert terpisah(X[np.arange(6) != 2], y[np.arange(6) != 2])
assert terpisah(X[np.arange(6) != 5], y[np.arange(6) != 5])

# Contoh Soal 12.4: VIF data mini dan VIF teoretis
assert F(35, 2) * F(19, 2) == F(16625, 100)
assert 17.5 * 9.5 - 11.5 ** 2 == 34
assert r(17.5 * 9.5 / 34) == 4.8897 and r(vif(Xf)[0]) == 4.8897
assert r(np.sqrt(4.8897)) == 2.2113
assert r(1 / (1 - 0.9 ** 2), 2) == 5.26 and r(np.sqrt(5.26), 2) == 2.29
print("Contoh Soal Bab 12: semua bilangan cocok")
