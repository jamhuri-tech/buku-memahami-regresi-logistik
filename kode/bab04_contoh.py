"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab04_gd import gd


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)

# Contoh Soal 4.1: dua langkah GD, eta = 0.25
j = gd(X, y, 0.25, 2)
assert np.allclose(j[1], [0, 1 / 16, 1 / 48])
assert (r(1 / 16), r(1 / 48)) == (0.0625, 0.0208)
z = X @ j[1]
assert [r(v) for v in z] == [0.0625, 0.1458, 0.1875, 0.2917, 0.375, 0.4375]
p = expit(z)
assert [r(v) for v in p] == [0.5156, 0.5364, 0.5467, 0.5724, 0.5927,
                             0.6077]
res = p - y
assert (r(res.sum()), r(Xf[:, 0] @ res), r(Xf[:, 1] @ res)) == (0.3715,
                                                               0.1276,
                                                               0.2822)
g = gradien(j[1], X, y)
assert [r(v) for v in g] == [0.0619, 0.0213, 0.0470]
assert [r(v) for v in j[2]] == [-0.0155, 0.0572, 0.0091]

# Contoh Soal 4.2: L dan batas teras
L = np.linalg.eigvalsh(X.T @ X / 24).max()
assert r(L) == 4.8555 and r(1 / L) == 0.2060
assert np.trace(X.T @ X) / 24 == 5.0

# Contoh Soal 4.3: lema penurunan untuk langkah pertama
g0 = gradien(np.zeros(3), X, y)
assert r(g0 @ g0) == 0.0694 and np.isclose(g0 @ g0, 1 / 16 + 1 / 144)
assert r(1 - L * 0.25 / 2) == 0.3931
batas = np.log(2) - 0.25 * (1 - L * 0.25 / 2) * (g0 @ g0)
assert r(batas) == 0.6863 and r(log_loss(j[1], X, y)) == 0.6857
assert log_loss(j[1], X, y) <= batas

# Contoh Soal 4.4: pemusatan
Xc = rancang(Xf - Xf.mean(0))
assert np.allclose(Xc.T @ Xc, [[6, 0, 0], [0, 17.5, 11.5], [0, 11.5, 9.5]])
assert 43 - 6 * 3.5 * 1.5 == 11.5 and 23 - 6 * 1.5 ** 2 == 9.5
assert r(11.5 / np.sqrt(17.5 * 9.5)) == 0.8919

# Contoh Soal 4.5: perkiraan banyak langkah
m = mle_mini()
mu = np.linalg.eigvalsh(hessian(m, X, y)).min()
assert r(mu) == 0.0194 and r(1 - mu / L) == 0.9960
assert round(np.log(1e-6) / np.log(1 - mu / L)) == 3454
mc = np.r_[m[0] + m[1:] @ Xf.mean(0), m[1:]]
muc = np.linalg.eigvalsh(hessian(mc, Xc, y)).min()
Lc = np.linalg.eigvalsh(Xc.T @ Xc / 24).max()
assert (r(muc), r(Lc)) == (0.0426, 1.0698) and r(1 - muc / Lc) == 0.9602
assert round(np.log(1e-6) / np.log(1 - muc / Lc)) == 340

# Contoh Soal 4.6: tiga pembaruan SGD, eta = 0.5
w = np.zeros(3)
hasil = []
for i in range(3):
    pi = expit(X[i] @ w)
    w = w - 0.5 * (pi - y[i]) * X[i]
    hasil.append([r(pi)] + [r(v) for v in w])
assert hasil == [[0.5, -0.25, -0.25, 0.0], [0.3208, -0.4104, -0.5708,
                                           -0.1604],
                 [0.1069, 0.0361, 0.7688, -0.1604]]
print("Contoh Soal Bab 4: semua bilangan cocok")
