"""Memeriksa setiap bilangan Contoh Soal Bab 5."""
import numpy as np

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian
from bab05_newton import irls, newton


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)
m = mle_mini()

# Contoh Soal 5.2: langkah Newton pertama, eksak
Hinv = 24 / 105 * np.array([[91, -21], [-21, 6]])
assert np.allclose(Hinv, np.linalg.inv(hessian(np.zeros(2), X, y)))
g0 = np.array([0, -7 / 12])
assert np.allclose(Hinv @ g0, [2.8, -0.8])
assert np.allclose(newton(X, y, langkah=1)[0][1], [-2.8, 0.8])

# Contoh Soal 5.3: langkah kedua dengan angka empat desimal
H = np.array([[0.1744, 0.6103], [0.6103, 2.5084]])
g = np.array([0.0, -0.1153])
det = H[0, 0] * H[1, 1] - H[0, 1] ** 2
assert (r(H[0, 0] * H[1, 1]), r(H[0, 1] ** 2), r(det)) == (0.4375, 0.3725,
                                                          0.0650)
v = np.array([[2.5084, -0.6103], [-0.6103, 0.1744]]) @ g
assert (r(v[0]), r(v[1])) == (0.0704, -0.0201)
t2 = np.array([-2.8, 0.8]) - v / det
assert (r(t2[0]), r(t2[1])) == (-3.8826, 1.1094)
j, _ = newton(X, y, langkah=2)
assert (r(j[2][0]), r(j[2][1])) == (-3.8842, 1.1098)

# Contoh Soal 5.4: IRLS langkah pertama
r0 = (y - 0.5) / 0.25
assert list(r0) == [-2, -2, 2, -2, 2, 2]
xc = x - 3.5
assert (xc * r0).sum() == 14 and (xc ** 2).sum() == 17.5
assert np.allclose(irls(X, y, 1), [-2.8, 0.8])

# Contoh Soal 5.5: konvergensi kuadratik
j, _ = newton(X, y, langkah=8)
e = [np.abs(t - m).max() for t in j]
assert [float(f"{v:.1e}") for v in e[3:6]] == [2.8e-2, 1.7e-4, 6.1e-9]
assert (r(e[4] / e[3] ** 2, 2), r(e[5] / e[4] ** 2, 2)) == (0.22, 0.22)

# Contoh Soal 5.6: pencarian garis Armijo dari (-6, 3)
th = np.array([-6.0, 3.0])
g, H = gradien(th, X, y), hessian(th, X, y)
s = -np.linalg.solve(H, g)
assert (r(s[0], 2), r(s[1], 2)) == (32.95, -18.52)
assert r(g @ s) == -6.8808 and r(log_loss(th, X, y)) == 1.1322
nilai = [r(log_loss(th + t * s, X, y)) for t in (1, 0.5, 0.25, 0.125)]
assert nilai == [24.6369, 10.0899, 2.9214, 0.4754]
tt = 0.125
assert log_loss(th + tt * s, X, y) <= log_loss(th, X, y) + 1e-4 * tt * (
    g @ s)
assert np.allclose(th + 0.125 * s, [-1.8807, 0.6855], atol=5e-5)
print("Contoh Soal Bab 5: semua bilangan cocok")
