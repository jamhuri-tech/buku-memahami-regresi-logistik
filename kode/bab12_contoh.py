"""Memeriksa setiap bilangan Contoh Soal Bab 12."""
import numpy as np
from scipy.special import expit
from scipy.stats import chi2

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians
from bab10_inferensi import mle
from bab12_diagnostik import cook, leverage, residu


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)
m = mle_mini()
p = expit(X @ m)
rp, rd = residu(y, p)

# Contoh Soal 12.1: residu mahasiswa ke-3
assert (r(1 - p[2]), r(np.sqrt(p[2] * (1 - p[2])))) == (0.6473, 0.4778)
assert r(rp[2]) == 1.3546 and r(0.6473 / 0.4778) == 1.3548
assert r(np.log(p[2])) == -1.0420 and r(rd[2]) == 1.4436
assert r(np.sqrt(2 * 1.0420)) == 1.4436

# Contoh Soal 12.2: jumlah kuadrat residu
assert r((rd ** 2).sum()) == 4.9560 and r((rp ** 2).sum()) == 4.0897

# Contoh Soal 12.3: leverage mahasiswa ke-3
C = kovarians(m, X)
q = np.array([1, 3.0]) @ C @ np.array([1, 3.0])
assert r(q) == 1.4838
assert r(11.4775 + 9 * 0.8328 + 6 * -2.9148) == 1.4839
h, _ = leverage(X, p)
assert r(h[2]) == 0.3388 and r(0.2283 * 1.4839) == 0.3388

# Contoh Soal 12.4: jarak Cook
assert r(rp[2] ** 2) == 1.8349
assert r(cook(y, p, h, 2)[2]) == 0.7109
assert r(1.8349 * 0.3388 / (2 * (1 - 0.3388) ** 2)) == 0.7110

# Contoh Soal 12.5: dfbeta satu langkah lawan eksak, titik 1
satu = C @ X[0] * (y[0] - p[0]) / (1 - h[0])
assert (r(satu[0]), r(satu[1])) == (-0.5482, 0.1333)
k = np.arange(6) != 0
t = mle(X[k], y[k])
assert (r(m[0] - t[0]), r(m[1] - t[1])) == (-0.5101, 0.1236)
assert (r(t[0]), r(t[1])) == (-3.7390, 1.0904)

# Contoh Soal 12.6: VIF teoretis
assert r(1 / (1 - 0.9 ** 2), 2) == 5.26 and r(np.sqrt(5.26), 2) == 2.29

# Contoh Soal 12.7: Box-Tidwell
assert r(chi2.sf(0.0175, 1)) == 0.8948
print("Contoh Soal Bab 12: semua bilangan cocok")
