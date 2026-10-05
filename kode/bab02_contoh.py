"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, rancang
from bab02_loss import log_loss, loss_titik


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)

# Contoh Soal 2.1: theta = 0
assert 0.5 ** 6 == 0.015625
assert r(-6 * np.log(2)) == -4.1589 and r(np.log(2)) == 0.6931
assert r(log_loss(np.zeros(2), X, y)) == 0.6931

# Contoh Soal 2.2: theta = (-2.8, 0.8)
th = np.array([-2.8, 0.8])
p = expit(X @ th)
suku = np.where(y == 1, np.log(p), np.log(1 - p))
assert [r(v) for v in suku] == [-0.1269, -0.2633, -0.9130, -0.9130,
                                -0.2633, -0.1269]
ell = suku.sum()
assert r(ell) == -2.6065 and r(sum(r(v) for v in suku)) == -2.6064
assert r(np.exp(ell)) == 0.0738 and r(np.exp(ell) * 64, 2) == 4.72
assert r(-ell / 6) == 0.4344

# Contoh Soal 2.3: bentuk skor, titik ke-3
assert r(np.log(1 + np.exp(-0.4))) == 0.5130
assert r(loss_titik(-0.4, 1)) == 0.9130

# Contoh Soal 2.4: bentuk margin, titik ke-4 (s = -1, z = 0.4)
assert r(1 + np.exp(0.4)) == 2.4918 and r(np.log(1 + np.exp(0.4))) == 0.9130

# Contoh Soal 2.5: gradien terhadap skor, y = 0, z = 5
p5 = expit(5.0)
assert r(p5) == 0.9933 and r(1 - p5) == 0.0067
assert r(2 * p5 * p5 * (1 - p5)) == 0.0132

# Contoh Soal 2.6: harapan log-loss untuk y ~ Bern(0.7)
R = lambda q: -0.7 * np.log(q) - 0.3 * np.log(1 - q)
q = np.linspace(0.01, 0.99, 9801)
assert abs(q[np.argmin(R(q))] - 0.7) < 1e-9
assert r(R(0.7)) == 0.6109 and r(R(0.5)) == 0.6931 and r(R(0.9)) == 0.7645
print("Contoh Soal Bab 2: semua bilangan cocok")
