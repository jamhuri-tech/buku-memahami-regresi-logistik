"""Memeriksa setiap bilangan Contoh Soal Bab 11."""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
m = mle_mini()
se = np.sqrt(np.diag(kovarians(m, rancang(x))))
Z = norm.ppf(0.975)

# Contoh Soal 11.1: skala peubah
assert r(np.exp(2 * m[1])) == 11.3368 and r(3.3670 ** 2) == 11.3367
assert (r(np.exp(2 * (m[1] - Z * se[1]))), r(np.exp(2 * (m[1] + Z * se[1])), 2)) == (0.3169, 405.58)
sx = np.std(x, ddof=1)
assert r(sx) == 1.8708 and r(m[1] * sx) == 2.2712 and r(np.exp(m[1] * sx)) == 9.6914

# Contoh Soal 11.2: tabel 2 x 2
assert 30 * 35 / (20 * 15) == 3.5 and r(np.log(3.5)) == 1.2528
assert r(1/30 + 1/20 + 1/15 + 1/35) == 0.1786
assert r(np.sqrt(1/30 + 1/20 + 1/15 + 1/35)) == 0.4226
lo, hi = np.log(3.5) - 1.96 * 0.4226, np.log(3.5) + 1.96 * 0.4226
assert (r(np.exp(lo)), r(np.exp(hi))) == (1.5288, 8.0129)
assert r(15 / 35) == 0.4286

# Contoh Soal 11.3: interaksi
b0 = logit(10 / 50)
assert r(b0) == -1.3863
assert r(logit(20 / 40) - b0) == 1.3863
assert r(logit(45 / 50)) == 2.1972
assert r(logit(45 / 50) - b0 - 2 * 1.3862944) == 0.8109
assert r(np.exp(logit(45 / 50) - b0 - 2 * np.log(4))) == 2.25

# Contoh Soal 11.4: AME
p = expit(rancang(x) @ m)
d = p * (1 - p)
assert r(d.sum()) == 0.7840 and r(m[1] * d.mean()) == 0.1586
assert r(m[1] / 4) == 0.3035

# Contoh Soal 11.5: odds ratio lawan rasio risiko
def p1(p0, OR):
    o = OR * p0 / (1 - p0)
    return o / (1 + o)
assert r(p1(0.2, 3)) == 0.4286 and r(p1(0.2, 3) / 0.2, 2) == 2.14
assert r(p1(0.01, 3)) == 0.0294 and r(p1(0.01, 3) / 0.01, 2) == 2.94

# Contoh Soal 11.6: ketidakruntuhan
assert r((0.5 / 0.5) / (0.2 / 0.8)) == 4.0 and r((0.8 / 0.2) / 1) == 4.0
assert r((0.65 / 0.35) / (0.35 / 0.65)) == 3.4490
print("Contoh Soal Bab 11: semua bilangan cocok")
