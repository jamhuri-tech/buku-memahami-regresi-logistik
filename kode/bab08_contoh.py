"""Memeriksa setiap bilangan Contoh Soal Bab 8."""
from fractions import Fraction as F

import numpy as np
from scipy.special import expit

from bab01_data import data_mini, mle_mini, rancang
from bab08_performa import (auc_mw, average_precision, brier, kebingungan,
                            pseudo_r2, ukuran)


def r(v, n=4):
    return round(float(v), n)


x, y = data_mini()
p = expit(rancang(x) @ mle_mini())

# Contoh Soal 8.1 dan 8.2
assert kebingungan(y, p, 0.5) == (2, 1, 2, 1)
assert np.allclose(ukuran(2, 1, 2, 1), [2 / 3] * 5)
assert kebingungan(y, p, 0.3) == (3, 1, 2, 0)
assert np.allclose(ukuran(3, 1, 2, 0), [5 / 6, 3 / 4, 1, 2 / 3, 6 / 7])
assert r(6 / 7) == 0.8571 and r(5 / 6) == 0.8333

# Contoh Soal 8.3: ROC dan AUC dengan trapesium
luas = F(1, 3) * F(2, 3) + F(1, 3) * 1 + F(1, 3) * 1
assert luas == F(8, 9) and r(8 / 9) == 0.8889

# Contoh Soal 8.4: Mann-Whitney
assert auc_mw(y, p) == 8 / 9
assert auc_mw(y, x) == 8 / 9          # transformasi naik: AUC tetap

# Contoh Soal 8.5: average precision
ap = F(1, 3) * 1 + F(1, 3) * 1 + F(1, 3) * F(3, 4)
assert ap == F(11, 12) and r(average_precision(y, p)) == 0.9167

# Contoh Soal 8.6: Brier
kuadrat = [r((pi - yi) ** 2) for pi, yi in zip(p, y)]
assert kuadrat[:3] == [0.0021, 0.0194, 0.4189]
assert r(brier(y, p)) == 0.1468 and r(1 - brier(y, p) / 0.25) == 0.4127

# Contoh Soal 8.7: pseudo-R^2
mcf, cs, nag, tjur = pseudo_r2(y, p)
assert (r(mcf), r(cs), r(nag), r(tjur)) == (0.4042, 0.4290, 0.5719, 0.4450)
assert r(1 - 2.4780 / 4.1589) == 0.4042
assert r(np.exp(2 * (-4.1589 + 2.4780) / 6)) == 0.5710
assert r(p[y == 1].mean()) == 0.7225 and r(p[y == 0].mean()) == 0.2775
print("Contoh Soal Bab 8: semua bilangan cocok")
