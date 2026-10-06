"""Memeriksa setiap bilangan Contoh Soal Bab 3."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit, logit

from bab01_data import data_mini, mle_mini, rancang
from bab03_turunan import gradien, hessian


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)

# Contoh Soal 3.1: gradien di w = 0
res = 0.5 - y
assert list(res) == [0.5, 0.5, -0.5, -0.5, -0.5, 0.5]
assert list(X.T @ res) == [0.0, -1.5, -0.5]
assert [r(v) for v in gradien(np.zeros(3), X, y)] == [0.0, -0.25, -0.0833]

# Contoh Soal 3.2: gradien di w = (-2, 1, -1)
s1 = expit(1.0)
assert r(s1) == 0.7311
assert np.isclose((X.T @ (expit(X @ [-2, 1, -1]) - y))[1], 6 * s1 - 4.5)
assert np.isclose((X.T @ (expit(X @ [-2, 1, -1]) - y))[2], 2 * s1 - 1.5)
assert (r(6 * s1 - 4.5), r(2 * s1 - 1.5)) == (-0.1136, -0.0379)
assert (r((6 * s1 - 4.5) / 6), r((2 * s1 - 1.5) / 6)) == (-0.0189, -0.0063)

# Contoh Soal 3.3: X^T X dan Hessian di w = 0
XtX = X.T @ X
assert XtX.astype(int).tolist() == [[6, 21, 9], [21, 91, 43], [9, 43, 23]]
det = 6 * (91 * 23 - 43 * 43) - 21 * (21 * 23 - 43 * 9) + 9 * (21 * 43 - 91 * 9)
assert det == 204 and (91 * 23 - 43 * 43, 21 * 23 - 43 * 9, 21 * 43 - 91 * 9) == (244, 96, 84)
assert np.allclose(hessian(np.zeros(3), X, y), XtX / 24)
assert r(204 / 24 ** 3, 5) == 0.01476
ev = np.linalg.eigvalsh(XtX / 24)
assert [r(v) for v in ev] == [0.0255, 0.1190, 4.8555]
assert r(ev.prod(), 5) == 0.01476

# Contoh Soal 3.4: Hessian di MLE (pecahan)
p = [F(1, 4), F(1, 4), F(3, 4), F(1, 2), F(1, 2), F(3, 4)]
d = [q * (1 - q) for q in p]
assert d == [F(3, 16)] * 3 + [F(1, 4)] * 2 + [F(3, 16)]
Xi = [[1, 1, 0], [1, 2, 1], [1, 3, 0], [1, 4, 2], [1, 5, 3], [1, 6, 3]]
I = [[sum(d[i] * Xi[i][j] * Xi[i][k] for i in range(6)) for k in range(3)]
     for j in range(3)]
assert I == [[F(5, 4), F(9, 2), F(2)], [F(9, 2), F(157, 8), F(19, 2)],
             [F(2), F(19, 2), F(41, 8)]]
assert np.allclose(np.array(I, float) / 6, hessian(mle_mini(), X, y))

# Contoh Soal 3.5: bentuk kuadrat
v = np.array([-2.0, 1.0, -1.0])
assert list(X @ v) == [-1, -1, 1, 0, 0, 1]
assert r(v @ (XtX / 24) @ v) == 0.1667

# Contoh Soal 3.6: persamaan skor di MLE
assert sum(p) == 3 and sum(q * x[1] for q, x in zip(p, Xi)) == 12
assert sum(q * x[2] for q, x in zip(p, Xi)) == 5
assert list(X.T @ y) == [3, 12, 5]
assert np.allclose(expit(X @ (np.log(3) * v)), np.array(p, float))

# Contoh Soal 3.7: model nol
assert r(logit(0.25)) == -1.0986 and r(np.log(3)) == 1.0986
print("Contoh Soal Bab 3: semua bilangan cocok")
