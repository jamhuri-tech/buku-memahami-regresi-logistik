"""Memeriksa setiap bilangan Contoh Soal Bab 9."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit, logit
from scipy.stats import chi2

from bab01_data import data_mini, mle_mini, rancang
from bab08_performa import peluang_mini
from bab09_kalibrasi import ece, hosmer_lemeshow, intersep_kemiringan


def r(v, n=4):
    return round(float(v), n)


p, y = peluang_mini()

# Contoh Soal 9.1: ECE dua dan tiga kelompok
e2 = F(2, 6) * abs(F(1, 4) - 0) + F(4, 6) * abs(F(5, 8) - F(3, 4))
assert F(3, 4) + F(1, 2) + F(1, 2) + F(3, 4) == 4 * F(5, 8)
assert e2 == F(1, 6) and r(ece(y, p, 2)) == r(1 / 6)
e3 = F(1, 3) * (F(1, 4) + F(1, 2) + F(1, 4))
assert e3 == F(1, 3) and r(ece(y, p, 3)) == r(1 / 3)

# Contoh Soal 9.2: intersep 0 dan kemiringan 1 di data latih
X0, _ = data_mini()
X = rancang(X0)
w = mle_mini()
z = X @ w
q = expit(z)
assert abs((y - q).sum()) < 1e-10 and abs((z * (y - q)).sum()) < 1e-10
a, c = intersep_kemiringan(y, p)
assert abs(a) < 1e-6 and abs(c - 1) < 1e-6

# Contoh Soal 9.3: Hosmer-Lemeshow tiga kelompok
suku = []
for O, E in [(0, F(1, 2)), (2, F(1)), (1, F(3, 2))]:
    suku.append((O - E) ** 2 / (E * (1 - E / 2)))
assert suku == [F(2, 3), F(2), F(2, 3)] and sum(suku) == F(10, 3)
hl, df, pv = hosmer_lemeshow(y, p, 3)
assert r(hl) == 3.3333 and df == 1 and r(pv) == 0.0679
assert r(chi2.sf(10 / 3, 1)) == 0.0679
# G = 2 membelah seri: {1, 2, 4} dan {5, 3, 6}, O = E di keduanya
assert F(1, 4) + F(1, 4) + F(1, 2) == 1 and F(1, 2) + F(3, 2) == 2
assert abs(hosmer_lemeshow(y, p, 2)[0]) < 1e-10

# Contoh Soal 9.4: rekalibrasi a = 0, c = 0.5
assert r(logit(0.9)) == 2.1972 and r(expit(0.5 * logit(0.9))) == 0.75
assert r(expit(0.5 * logit(0.1))) == 0.25
assert r(expit(0.5 * logit(0.5))) == 0.5
assert r(np.sqrt(0.9)) == 0.9487
assert r(np.sqrt(0.9) + np.sqrt(0.1)) == 1.2649
print("Contoh Soal Bab 9: semua bilangan cocok")
