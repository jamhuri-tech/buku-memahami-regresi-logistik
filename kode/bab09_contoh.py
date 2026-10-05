"""Memeriksa setiap bilangan Contoh Soal Bab 9."""
import numpy as np
from scipy.special import expit, logit

from bab01_data import data_mini, mle_mini, rancang
from bab09_kalibrasi import ece, hosmer_lemeshow, intersep_kemiringan


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
X = rancang(x)
m = mle_mini()
p = expit(X @ m)

# Contoh Soal 9.1: ECE dua kelompok
assert r(p[:3].mean()) == 0.1793 and r(p[3:].mean()) == 0.8207
assert r(abs(p[:3].mean() - 1 / 3)) == 0.1540
assert r(ece(y, p, 2)) == 0.1540

# Contoh Soal 9.2: intersep 0 dan kemiringan 1 di data latih
z = X @ m
assert abs((y - p).sum()) < 1e-10 and abs((z * (y - p)).sum()) < 1e-10
a, c = intersep_kemiringan(y, p)
assert abs(a) < 1e-8 and abs(c - 1) < 1e-8

# Contoh Soal 9.3: Hosmer-Lemeshow dua kelompok
E1 = p[:3].sum()
assert r(E1) == 0.5379 and r((1 - E1) ** 2) == 0.2135
assert r(E1 * (1 - E1 / 3)) == 0.4415
assert r((1 - E1) ** 2 / (E1 * (1 - E1 / 3))) == 0.4837
assert r(hosmer_lemeshow(y, p, 2)[0]) == 0.9673

# Contoh Soal 9.4: rekalibrasi a = 0, c = 0.5
assert r(logit(0.9)) == 2.1972 and r(expit(0.5 * logit(0.9))) == 0.75
assert r(expit(0.5 * logit(0.1))) == 0.25
assert r(expit(0.5 * logit(0.5))) == 0.5
print("Contoh Soal Bab 9: semua bilangan cocok")
