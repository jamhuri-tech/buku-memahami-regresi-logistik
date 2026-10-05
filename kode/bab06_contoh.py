"""Memeriksa setiap bilangan Contoh Soal Bab 6."""
import numpy as np

from bab01_data import data_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab06_penalti import Y_PISAH, firth, ista_l1, lunak, newton_l2


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)

# Contoh Soal 6.1: kerugian di sepanjang sinar t v
v = np.array([-3.5, 1.0])
s = 2 * Y_PISAH - 1
assert list(s * (X @ v)) == [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]
suku = [r(np.log1p(np.exp(-a))) for a in (2.5, 1.5, 0.5)]
assert suku == [0.0789, 0.2014, 0.4741]
assert r(log_loss(v, X, Y_PISAH)) == 0.2515
t = np.arange(1, 50)
L = [log_loss(tt * v, X, Y_PISAH) for tt in t]
assert all(a > b > 0 for a, b in zip(L, L[1:]))

# Contoh Soal 6.2: langkah Newton berpenalti, data terpisah, lam = 1/6
g0 = gradien(np.zeros(2), X, Y_PISAH)
assert np.allclose(g0, [0, -0.75])
H = hessian(np.zeros(2), X, Y_PISAH) + np.diag([0, 1 / 6])
assert np.allclose(H * 24, [[6, 21], [21, 95]])
assert 6 * 95 - 21 ** 2 == 129
step = np.linalg.solve(H, g0)
assert (r(step[0]), r(step[1])) == (2.9302, -0.8372)
assert np.allclose(24 / 129 * np.array([15.75, -4.5]), step)
assert np.allclose(newton_l2(X, Y_PISAH, 1 / 6, langkah=1),
                   [-2.9302, 0.8372], atol=5e-5)

# Contoh Soal 6.3: C dan lambda
assert r(1 / (6 * 1.0)) == 0.1667

# Contoh Soal 6.4: soft-thresholding dan satu langkah ISTA
assert np.allclose(lunak(np.array([0.3, -1.2, 0.05]), 0.1),
                   [0.2, -1.1, 0.0])
th = ista_l1(X, y, 0.5, 0.25, 1)
assert (r(th[0]), r(th[1])) == (0.0, 0.0208)
assert r(7 / 48 - 0.125) == 0.0208

# Contoh Soal 6.5: lambda_maks
gw = abs(gradien(np.zeros(2), X, y)[1])
assert r(gw) == 0.5833 and r(1 / (6 * gw)) == 0.2857

# Contoh Soal 6.6: leverage di theta = 0
h0 = 1 / 6 + (x - 3.5) ** 2 / 17.5
assert [r(v) for v in h0] == [0.5238, 0.2952, 0.1810, 0.1810, 0.2952,
                              0.5238]
assert r(h0.sum()) == 2.0
p = np.full(6, 0.5)
A = X.T @ (X * 0.25)
hk = 0.25 * np.einsum("ij,jk,ik->i", X, np.linalg.inv(A), X)
assert np.allclose(hk, h0)
th, h = firth(X, Y_PISAH)
assert (r(th[0]), r(th[1])) == (-3.9512, 1.1289)
print("Contoh Soal Bab 6: semua bilangan cocok")
