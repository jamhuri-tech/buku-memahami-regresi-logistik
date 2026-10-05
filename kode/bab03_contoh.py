"""Memeriksa setiap bilangan Contoh Soal Bab 3."""
import numpy as np
from scipy.special import expit, logit

from bab01_data import data_mini, mle_mini, rancang
from bab02_loss import log_loss
from bab03_turunan import gradien, hessian


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)

# Contoh Soal 3.1: gradien di theta = 0
res = expit(X @ np.zeros(2)) - y
assert list(res) == [0.5, 0.5, -0.5, 0.5, -0.5, -0.5]
assert res.sum() == 0 and (x * res).sum() == -3.5
assert [r(v) for v in gradien(np.zeros(2), X, y)] == [0.0, -0.5833]

# Contoh Soal 3.2: gradien di theta = (-2.8, 0.8)
t1 = np.array([-2.8, 0.8])
res = expit(X @ t1) - y
assert [r(v) for v in res] == [0.1192, 0.2315, -0.5987, 0.5987, -0.2315,
                               -0.1192]
assert r((x * res).sum()) == -0.6918
assert [r(v) for v in gradien(t1, X, y)] == [0.0, -0.1153]

# Contoh Soal 3.3: Hessian di theta = 0
H0 = hessian(np.zeros(2), X, y)
assert np.allclose(H0, np.array([[6, 21], [21, 91]]) / 24)
assert [r(v) for v in H0.ravel()] == [0.25, 0.875, 0.875, 3.7917]
assert r(np.linalg.det(H0)) == 0.1823 and abs(105 / 576 - 0.18229) < 1e-5
ev = np.linalg.eigvalsh(H0)
assert (r(ev[0]), r(ev[1])) == (0.0456, 3.9960)
assert r(np.sqrt((0.25 - 91 / 24) ** 2 + 4 * 0.875 ** 2)) == 3.9504

# Contoh Soal 3.4: Hessian di theta = (-2.8, 0.8)
p = expit(X @ t1)
d = p * (1 - p)
assert [r(v) for v in d] == [0.1050, 0.1779, 0.2403, 0.2403, 0.1779,
                             0.1050]
assert r(d.sum()) == 1.0463 and r((d * x).sum()) == 3.6620
assert r((d * x * x).sum()) == 15.0502
assert [r(v) for v in hessian(t1, X, y).ravel()] == [0.1744, 0.6103,
                                                      0.6103, 2.5084]

# Contoh Soal 3.5: bentuk kuadrat v^T H v
v = np.array([-3.5, 1.0])
assert r(((v[0] + v[1] * x) ** 2).sum()) == 17.5
assert r(v @ H0 @ v) == 0.7292

# Contoh Soal 3.6: persamaan skor di MLE
ph = expit(X @ mle_mini())
assert [r(v) for v in ph] == [0.0459, 0.1393, 0.3527, 0.6473, 0.8607,
                              0.9541]
assert r(ph.sum()) == 3.0 and r((x * ph).sum()) == 14.0
assert (x * y).sum() == 14
assert r(sum(r(v) * xi for v, xi in zip(ph, x))) == 13.9999

# Contoh Soal 3.7: model nol
assert r(logit(0.25)) == -1.0986 and r(np.log(3)) == 1.0986

# Contoh Soal 3.8: simetri memaksa b = -3.5 w
for th in [np.array([0.3, -0.7]), np.array([-2.0, 1.5])]:
    bayang = np.array([-th[0] - 7 * th[1], th[1]])
    assert abs(log_loss(th, X, y) - log_loss(bayang, X, y)) < 1e-12
t = mle_mini()
assert abs(t[0] + 3.5 * t[1]) < 1e-6
print("Contoh Soal Bab 3: semua bilangan cocok")
