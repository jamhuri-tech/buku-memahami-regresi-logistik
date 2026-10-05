"""Memeriksa setiap bilangan Contoh Soal Bab 7."""
import numpy as np
from scipy.special import expit, logit

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians, prediksi_selang


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)
m = mle_mini()

# Contoh Soal 7.1: peluang untuk x0 = 4.5 dan 2.5
assert r(m[0] + 4.5 * m[1]) == 1.2140 and r(m[1]) == 1.2140
assert r(expit(m[0] + 4.5 * m[1])) == 0.7710
assert r(expit(m[0] + 2.5 * m[1])) == 0.2290

# Contoh Soal 7.2: informasi dan kovarians
p = expit(X @ m)
d = p * (1 - p)
assert [r(v) for v in d[:3]] == [0.0438, 0.1199, 0.2283]
I = X.T @ (X * d[:, None])
assert [r(v) for v in I.ravel()] == [0.7840, 2.7439, 2.7439, 10.8042]
det = I[0, 0] * I[1, 1] - I[0, 1] ** 2
assert r(det) == 0.9413
C = kovarians(m, X)
assert [r(v) for v in C.ravel()] == [11.4775, -2.9148, -2.9148, 0.8328]
assert (r(np.sqrt(C[0, 0])), r(np.sqrt(C[1, 1]))) == (3.3879, 0.9126)

# Contoh Soal 7.3: metode delta di x0 = 5
v = np.array([1.0, 5.0])
var = v @ C @ v
assert r(var) == 3.1494 and r(np.sqrt(var)) == 1.7747
assert r(C[0, 0] + 25 * C[1, 1] + 10 * C[0, 1]) == 3.1494
z0 = m[0] + 5 * m[1]
assert r(z0) == 1.8210
lo, hi = z0 - 1.96 * np.sqrt(var), z0 + 1.96 * np.sqrt(var)
assert (r(lo), r(hi)) == (-1.6573, 5.2994)
assert (r(expit(lo)), r(expit(hi))) == (0.1601, 0.9950)

# Contoh Soal 7.4: selang simetris di skala peluang keluar dari [0, 1]
p0 = expit(z0)
setengah = 1.96 * p0 * (1 - p0) * np.sqrt(var)
assert (r(p0), r(p0 * (1 - p0)), r(setengah)) == (0.8607, 0.1199, 0.4171)
assert (r(p0 - setengah), r(p0 + setengah)) == (0.4436, 1.2777)

# Contoh Soal 7.5: ambang dan jam belajar minimum
assert r(logit(0.3)) == -0.8473
assert r((logit(0.3) - m[0]) / m[1]) == 2.8021

# Contoh Soal 7.6: ambang dari biaya
assert 1 / (1 + 4) == 0.2
assert r(logit(0.2)) == -1.3863 and r((logit(0.2) - m[0]) / m[1]) == 2.3581
print("Contoh Soal Bab 7: semua bilangan cocok")
