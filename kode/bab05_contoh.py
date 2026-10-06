"""Memeriksa setiap bilangan Contoh Soal Bab 5."""
import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab05_newton import irls, newton


def r(v, n=4):
    return round(float(v), n)


Xf, y = data_mini()
X = rancang(Xf)
m = mle_mini()
v = np.array([-2.0, 1.0, -1.0])

# Contoh Soal 5.2: langkah Newton pertama dengan eliminasi
assert list(X.T @ (4 * y - 2)) == [0, 6, 2]
# baris 1: a = -3.5 b - 1.5 c; sisa sistem 2 x 2
assert (21 * -3.5 + 91, 21 * -1.5 + 43) == (17.5, 11.5)
assert (9 * -3.5 + 43, 9 * -1.5 + 23) == (11.5, 9.5)
assert 17.5 * 9.5 - 11.5 ** 2 == 34
assert ((6 * 9.5 - 11.5 * 2) / 34, (17.5 * 2 - 11.5 * 6) / 34) == (1, -1)
assert -3.5 * 1 - 1.5 * -1 == -2
assert np.allclose(newton(X, y, langkah=1)[0][1], v)

# Contoh Soal 5.3: langkah kedua sepanjang garis c v
g, H = gradien(v, X, y), hessian(v, X, y)
d1 = expit(1) * expit(-1)
assert r(d1) == 0.1966
assert r(v @ g) == -0.0126 and np.isclose(v @ g, (6 * expit(1) - 4.5 - (2 * expit(1) - 1.5)) / 6 * 1)
assert np.isclose(v @ H @ v, 4 * d1 / 6) and r(v @ H @ v) == 0.1311
c2 = 1 - (v @ g) / (v @ H @ v)
assert r(c2) == 1.0963
assert np.allclose(newton(X, y, langkah=2)[0][2], c2 * v)

# Contoh Soal 5.4: MLE tepat c = ln 3
for c in (0.5, 1.0, np.log(3)):
    k = X @ v
    turunan = k @ (expit(c * k) - y)
    assert np.isclose(turunan, 2 * (2 * expit(c) - 1) - 1)
assert np.isclose(expit(np.log(3)), 0.75)

# Contoh Soal 5.5: IRLS langkah pertama
assert list(4 * y - 2) == [-2, -2, 2, 2, 2, -2]
assert np.allclose(irls(X, y, 1), v)

# Contoh Soal 5.6: konvergensi kuadratik
j, _ = newton(X, y, langkah=8)
e = [np.abs(w - m).max() for w in j]
assert (r(e[1]), float(f"{e[2]:.1e}"), float(f"{e[3]:.1e}")) == (0.1972,
                                                                4.5e-3,
                                                                2.6e-6)
assert (r(e[2] / e[1] ** 2, 3), r(e[3] / e[2] ** 2, 3)) == (0.117, 0.125)

# Contoh Soal 5.7: Armijo dari (-6, 2, -2)
w = np.array([-6.0, 2.0, -2.0])
g, H = gradien(w, X, y), hessian(w, X, y)
s = -np.linalg.solve(H, g)
assert [r(t, 2) for t in s] == [14.19, -4.33, 4.02] and r(g @ s) == -1.4210
assert r(log_loss(w, X, y)) == 0.9461
assert (r(log_loss(w + s, X, y)), r(log_loss(w + 0.5 * s, X, y))) == (2.1119,
                                                                    0.7775)
assert log_loss(w + 0.5 * s, X, y) <= log_loss(w, X, y) + 1e-4 * 0.5 * (g @ s)
print("Contoh Soal Bab 5: semua bilangan cocok")
