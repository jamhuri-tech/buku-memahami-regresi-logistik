"""Memeriksa setiap bilangan Contoh Soal Bab 4."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab04_gd import gd


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)

# Contoh Soal 4.1: dua langkah GD dengan eta = 0.25
j = gd(X, y, 0.25, 2)
assert (r(j[1][0]), r(j[1][1])) == (0.0, 0.1458)
assert r(7 / 48) == 0.1458
p = expit(7 / 48 * x)
assert [r(v) for v in p] == [0.5364, 0.5724, 0.6077, 0.6418, 0.6746,
                             0.7058]
res = np.round(p, 4) - y
assert r(res.sum()) == 0.7387 and r((x * res).sum()) == -0.3207
g = gradien(j[1], X, y)
assert (r(g[0]), r(g[1])) == (0.1231, -0.0534)
assert (r(j[2][0]), r(j[2][1])) == (-0.0308, 0.1592)

# Contoh Soal 4.2: konstanta L dan laju 1/L
Lmax = np.linalg.eigvalsh(X.T @ X / 24).max()
assert r(Lmax) == 3.9960 and r(1 / Lmax) == 0.2502
assert np.allclose(hessian(np.zeros(2), X, y), X.T @ X / 24)

# Contoh Soal 4.3: lema penurunan untuk langkah pertama
g0 = gradien(np.zeros(2), X, y)
batas = log_loss(np.zeros(2), X, y) - 0.25 * (1 - Lmax * 0.25 / 2) * (
    g0 @ g0)
assert r(g0 @ g0) == 0.3403 and r(1 - 3.9960 * 0.25 / 2) == 0.5005
assert r(batas) == 0.6506 and r(log_loss(j[1], X, y)) == 0.6475
assert log_loss(j[1], X, y) <= batas

# Contoh Soal 4.4: pemusatan
Xc = rancang(x - 3.5)
A = Xc.T @ Xc / 24
assert np.allclose(A, np.diag([0.25, 17.5 / 24]))
assert r(17.5 / 24) == 0.7292 and r(24 / 17.5) == 1.3714

# Contoh Soal 4.5: perkiraan banyak langkah
m = mle_mini()
mu = np.linalg.eigvalsh(hessian(m, X, y)).min()
assert r(mu) == 0.0136 and r(1 - mu / Lmax) == 0.9966
assert round(np.log(1e-6) / np.log(1 - mu / Lmax)) == 4042
muc = np.linalg.eigvalsh(hessian(np.array([0, m[1]]), Xc, y)).min()
assert r(muc) == 0.1307 and r(1 - muc / (17.5 / 24)) == 0.8208
assert round(np.log(1e-6) / np.log(1 - muc / (17.5 / 24))) == 70

# Contoh Soal 4.6: tiga pembaruan SGD, eta = 0.5 tetap
th = np.zeros(2)
hasil = []
for i in range(3):
    pi = expit(X[i] @ th)
    th = th - 0.5 * (pi - y[i]) * X[i]
    hasil.append((r(pi), r(th[0]), r(th[1])))
assert hasil == [(0.5, -0.25, -0.25), (0.3208, -0.4104, -0.5708),
                 (0.1069, 0.0361, 0.7688)]
assert r(-0.4104 - 3 * 0.5708) == -2.1228
print("Contoh Soal Bab 4: semua bilangan cocok")
