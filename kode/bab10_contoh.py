"""Memeriksa setiap bilangan Contoh Soal Bab 10."""
import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians
from bab10_inferensi import log_kem, profil, selang_profil


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)
m = mle_mini()
se = np.sqrt(np.diag(kovarians(m, X)))

# Contoh Soal 10.1: varians asimtotik, Var(U) = I
p = expit(X @ m)
I = X.T @ (X * (p * (1 - p))[:, None])
assert np.allclose(np.linalg.inv(I), kovarians(m, X))

# Contoh Soal 10.2: Wald
z = m[1] / se[1]
assert (r(z), r(z ** 2), r(2 * norm.sf(z))) == (1.3303, 1.7697, 0.1834)
lo, hi = m[1] - 1.96 * se[1], m[1] + 1.96 * se[1]
assert (r(lo), r(hi)) == (-0.5746, 3.0027)
assert (r(np.exp(lo)), r(np.exp(hi), 2)) == (0.5629, 20.14)

# Contoh Soal 10.3: rasio kemungkinan
l1, l0 = log_kem(m, X, y), log_kem(np.zeros(2), X, y)
assert (r(l1), r(l0), r(2 * (l1 - l0))) == (-2.4780, -4.1589, 3.3618)
assert r(chi2.sf(2 * (l1 - l0), 1)) == 0.0667
assert (r(-2 * l1), r(-2 * l0)) == (4.9560, 8.3178)
assert r(chi2.ppf(0.95, 1)) == 3.8415

# Contoh Soal 10.4: uji skor
U = X.T @ (y - 0.5)
assert list(U) == [0.0, 3.5]
S = U @ np.linalg.solve(X.T @ X / 4, U)
assert r(S) == 2.8 and r(3.5 ** 2 * 24 / 105) == 2.8
assert r(chi2.sf(2.8, 1)) == 0.0943

# Contoh Soal 10.5: selang profil (diperiksa pada batasnya)
a, b = selang_profil(X, y, 1)
assert (r(a), r(b)) == (-0.0658, 4.1592)
for v in (a, b):
    assert r(2 * (l1 - profil(X, y, 1, v))) == 3.8415

# Contoh Soal 10.6: AIC dan BIC
assert (r(-2 * l1 + 4), r(-2 * l0 + 2)) == (8.9560, 10.3178)
assert r(np.log(6)) == 1.7918
assert (r(-2 * l1 + 2 * np.log(6)), r(-2 * l0 + np.log(6))) == (8.5395,
                                                                10.1095)
print("Contoh Soal Bab 10: semua bilangan cocok")
