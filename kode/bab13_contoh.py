"""Memeriksa setiap bilangan Contoh Soal Bab 13."""
import numpy as np

from bab01_data import data_mini, mle_mini, rancang
from bab13_validasi import delong, komponen_delong, n_riley


def r(v, n=4):
    return round(float(v), n)


# Contoh Soal 13.1: ringkasan validasi silang
a = np.array([0.82, 0.76, 0.85, 0.79, 0.78])
assert r(a.mean()) == 0.80 and r(np.var(a, ddof=1), 5) == 0.00125
assert r(np.std(a, ddof=1)) == 0.0354 and r(0.0354 / np.sqrt(5)) == 0.0158

# Contoh Soal 13.2: koreksi optimisme
o = np.array([0.05, 0.03, 0.04, 0.06, 0.02])
assert r(o.mean()) == 0.04 and r(0.87 - o.mean()) == 0.83

# Contoh Soal 13.3: galat baku bootstrap
w = np.array([1.0, 1.2, 1.3, 1.5, 1.5])
assert r(w.mean()) == 1.3 and r(np.var(w, ddof=1)) == 0.045
assert r(np.std(w, ddof=1)) == 0.2121

# Contoh Soal 13.4: DeLong pada data mini
x, y = data_mini()
p = 1 / (1 + np.exp(-(rancang(x) @ mle_mini())))
v10, v01 = komponen_delong(y, p)
assert np.allclose(v10, [2 / 3, 1, 1]) and np.allclose(v01, [1, 1, 2 / 3])
assert np.isclose(np.var(v10, ddof=1), 1 / 27)
auc, se = delong(y, p)
assert np.isclose(se, np.sqrt(2) / 9) and r(se) == 0.1571
assert (r(auc - 1.96 * se, 2), r(auc + 1.96 * se, 2)) == (0.58, 1.20)

# Contoh Soal 13.5: ukuran sampel Riley
assert r(np.log(1 - 0.2 / 0.9)) == -0.2513
assert r(n_riley(10, 0.2), 1) == 397.9
assert r(398 * 0.3, 1) == 119.4 and r(119.4 / 10, 2) == 11.94
print("Contoh Soal Bab 13: semua bilangan cocok")
