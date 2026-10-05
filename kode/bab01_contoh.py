"""Memeriksa setiap bilangan Contoh Soal Bab 1."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, rancang


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)

# Contoh Soal 1.1: skor dan peluang di theta = (-2.8, 0.8)
z = X @ np.array([-2.8, 0.8])
assert np.allclose(z, [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0])
assert [r(np.exp(v)) for v in (2.0, 1.2, 0.4)] == [7.3891, 3.3201, 1.4918]
p = expit(z)
assert [r(v) for v in p] == [0.1192, 0.2315, 0.4013, 0.5987, 0.7685,
                             0.8808]
assert np.allclose(p + p[::-1], 1)

# Contoh Soal 1.2: turunan fungsi logistik
for v in (-2.0, 0.0, 2.0):
    h = 1e-6
    beda = (expit(v + h) - expit(v - h)) / (2 * h)
    assert abs(beda - expit(v) * (1 - expit(v))) < 1e-9
assert r(expit(2) * (1 - expit(2))) == 0.1050

# Contoh Soal 1.3: odds, logit, dan rasio odds berurutan
odds = p / (1 - p)
assert r(odds[2]) == 0.6703 and r(np.log(odds[2]), 2) == -0.40
assert np.allclose(odds[1:] / odds[:-1], np.exp(0.8))
assert r(np.exp(0.8)) == 2.2255

# Contoh Soal 1.4: tafsiran MLE
m = LogisticRegression(penalty=None, tol=1e-10).fit(x[:, None], y)
b, w = m.intercept_[0], m.coef_[0, 0]
assert (r(b), r(w)) == (-4.2491, 1.2140)
assert r(-b / w) == 3.5 and r(np.exp(w)) == 3.3670 and r(w / 4) == 0.3035
p3, p4 = expit(b + 3 * w), expit(b + 4 * w)
assert (r(p3), r(p4), r(p4 - p3)) == (0.3527, 0.6473, 0.2945)

# Contoh Soal 1.5: dua peubah
z5 = 0.5 + 2 * 1 - 1 * 3
assert z5 == -0.5 and r(expit(z5)) == 0.3775
assert r(np.exp(2)) == 7.3891
print("Contoh Soal Bab 1: semua bilangan cocok")
