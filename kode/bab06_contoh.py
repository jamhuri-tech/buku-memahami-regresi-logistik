"""Memeriksa setiap bilangan Contoh Soal Bab 6."""
import numpy as np

from bab01_data import data_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab06_penalti import Y_PISAH, firth, ista_l1, lunak, newton_l2


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)

# Contoh Soal 6.1: sinar t v pada data terpisah
v = np.array([-3.5, 1.0, 0.0])
s = 2 * Y_PISAH - 1
assert list(s * (X @ v)) == [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]
assert [r(np.log1p(np.exp(-a))) for a in (2.5, 1.5, 0.5)] == [0.0789,
                                                             0.2014,
                                                             0.4741]
assert r(log_loss(v, X, Y_PISAH)) == 0.2515
L = [log_loss(t * v, X, Y_PISAH) for t in range(1, 50)]
assert all(a > b > 0 for a, b in zip(L, L[1:]))

# Contoh Soal 6.2: langkah Newton berpenalti, lam = 1/6
g0 = gradien(np.zeros(3), X, Y_PISAH)
assert np.allclose(g0, [0, -0.75, -3.5 / 6])
assert list(-24 * g0) == [0, 18, 14]
A = X.T @ X + 4 * np.diag([0, 1, 1])
assert A.astype(int).tolist() == [[6, 21, 9], [21, 95, 43], [9, 43, 27]]
assert (21 * -3.5 + 95, 21 * -1.5 + 43, 9 * -3.5 + 43, 9 * -1.5 + 27) == (
    21.5, 11.5, 11.5, 13.5)
assert 21.5 * 13.5 - 11.5 ** 2 == 158
s1, s2 = (18 * 13.5 - 11.5 * 14) / 158, (21.5 * 14 - 11.5 * 18) / 158
assert (r(s1), r(s2), r(-3.5 * s1 - 1.5 * s2)) == (0.5190, 0.5949,
                                                   -2.7089)
assert np.allclose(newton_l2(X, Y_PISAH, 1 / 6, langkah=1),
                   [-3.5 * s1 - 1.5 * s2, s1, s2])

# Contoh Soal 6.3: C dan lambda
assert r(1 / 6) == 0.1667

# Contoh Soal 6.4: soft-thresholding dan satu langkah ISTA
assert np.allclose(lunak(np.array([0.3, -1.2, 0.05]), 0.1), [0.2, -1.1, 0])
w = ista_l1(X, y, 0.1, 0.2, 1)
assert np.allclose(w, [0, 0.03, 0])

# Contoh Soal 6.5: lambda_maks
g = gradien(np.zeros(3), X, y)
assert r(abs(g[1])) == 0.25 and r(abs(g[2])) == 0.0833
assert r(1 / (6 * 0.25)) == 0.6667

# Contoh Soal 6.6: leverage di w = 0
xc = Xf - Xf.mean(0)
Sinv34 = np.array([[9.5, -11.5], [-11.5, 17.5]])
q = np.einsum("ij,jk,ik->i", xc, Sinv34, xc)
assert list(q) == [12.5, 8.5, 24.5, 1.0, 9.0, 12.5]
h0 = 1 / 6 + q / 34
assert [r(v) for v in h0] == [0.5343, 0.4167, 0.8873, 0.1961, 0.4314,
                              0.5343]
assert r(h0.sum()) == 3.0
w, h = firth(X, Y_PISAH)
assert [r(v) for v in w] == [-2.6608, 0.5302, 0.6689]
print("Contoh Soal Bab 6: semua bilangan cocok")
