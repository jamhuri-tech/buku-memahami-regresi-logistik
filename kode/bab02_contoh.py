"""Memeriksa setiap bilangan Contoh Soal Bab 2."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss, loss_titik


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)

# Contoh Soal 2.1: w = 0
assert 0.5 ** 6 == 0.015625
assert r(-6 * np.log(2)) == -4.1589 and r(log_loss(np.zeros(3), X, y)) == 0.6931

# Contoh Soal 2.2: w = (-2, 1, -1)
w = np.array([-2.0, 1.0, -1.0])
p = expit(X @ w)
suku = np.where(y == 1, np.log(p), np.log(1 - p))
assert [r(v) for v in suku] == [-0.3133, -0.3133, -0.3133, -0.6931,
                                -0.6931, -1.3133]
assert r(suku.sum()) == -3.6393 and r(sum(r(v) for v in suku)) == -3.6394
assert r(np.exp(suku.sum())) == 0.0263 and r(np.exp(suku.sum()) * 64, 2) == 1.68
assert r(-suku.sum() / 6) == 0.6066

# Contoh Soal 2.3: bentuk skor, mahasiswa ke-6
assert r(np.log(1 + np.e)) == 1.3133 and r(loss_titik(1.0, 0)) == 1.3133

# Contoh Soal 2.4: bentuk margin, mahasiswa ke-3 (s = +1, z = 1)
assert r(np.log(1 + np.exp(-1))) == 0.3133

# Contoh Soal 2.5: MLE dalam bentuk tertutup
ell = np.sum(np.where(y == 1, np.log(expit(X @ mle_mini())),
                      np.log(1 - expit(X @ mle_mini()))))
assert np.isclose(ell, 3 * np.log(0.75) + 2 * np.log(0.5) + np.log(0.25))
assert np.isclose(ell, 3 * np.log(3) - 10 * np.log(2))
assert r(ell) == -3.6356 and r(-ell / 6) == 0.6059
assert (r(-np.log(0.75)), r(-np.log(0.25))) == (0.2877, 1.3863)

# Contoh Soal 2.6: gradien terhadap skor, y = 0, z = 5
p5 = expit(5.0)
assert r(p5) == 0.9933 and r(2 * p5 * p5 * (1 - p5)) == 0.0132

# Contoh Soal 2.7: harapan log-loss untuk y ~ Bern(0.7)
R = lambda q: -0.7 * np.log(q) - 0.3 * np.log(1 - q)
q = np.linspace(0.01, 0.99, 9801)
assert abs(q[np.argmin(R(q))] - 0.7) < 1e-9
assert r(R(0.7)) == 0.6109 and r(R(0.5)) == 0.6931 and r(R(0.9)) == 0.7645
print("Contoh Soal Bab 2: semua bilangan cocok")
