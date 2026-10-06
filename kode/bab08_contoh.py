"""Memeriksa setiap bilangan Contoh Soal Bab 8."""
from fractions import Fraction as F

import numpy as np

from bab01_data import data_mini
from bab08_performa import (auc_mw, average_precision, brier, kebingungan,
                            peluang_mini, pseudo_r2, ukuran)


def r(v, n=4):
    return round(float(v), n)


p, y = peluang_mini()
assert np.allclose(p, [1 / 4, 1 / 4, 3 / 4, 1 / 2, 1 / 2, 3 / 4])

# Contoh Soal 8.1: dua ambang
assert kebingungan(y, p, 0.5) == (3, 1, 2, 0)
assert np.allclose(ukuran(3, 1, 2, 0), [5 / 6, 3 / 4, 1, 2 / 3, 6 / 7])
assert kebingungan(y, p, 0.7) == (1, 1, 2, 2)
assert np.allclose(ukuran(1, 1, 2, 2), [1 / 2, 1 / 2, 1 / 3, 2 / 3, 2 / 5])
assert F(2) * F(1, 2) * F(1, 3) / (F(1, 2) + F(1, 3)) == F(2, 5)
assert r(5 / 6) == 0.8333 and r(6 / 7) == 0.8571

# Contoh Soal 8.2: ROC dengan seri, luas trapesium
luas = F(1, 3) * (0 + F(1, 3)) / 2 + F(2, 3) * 1
assert luas == F(13, 18) and r(13 / 18) == 0.7222

# Contoh Soal 8.3: pasangan (Mann-Whitney)
assert F(2 + F(1, 2) + 2 + 2, 9) == F(13, 18)
assert auc_mw(y, p) == 13 / 18
X, _ = data_mini()
assert auc_mw(y, X[:, 0] - X[:, 1]) == 13 / 18   # urutan sama
assert auc_mw(y, X[:, 0]) == 2 / 3                # x1 saja

# Contoh Soal 8.4: average precision per ambang
ap = F(1, 3) * F(1, 2) + F(2, 3) * F(3, 4)
assert ap == F(2, 3) and r(average_precision(y, p)) == 0.6667

# Contoh Soal 8.5: Brier
jumlah = F(1, 16) * 3 + F(4, 16) * 2 + F(9, 16)
assert jumlah == F(5, 4) and jumlah / 6 == F(5, 24)
assert r(brier(y, p)) == 0.2083
assert 1 - F(5, 24) / F(1, 4) == F(1, 6)

# Contoh Soal 8.6: pseudo-R^2
ll = 3 * np.log(3) - 10 * np.log(2)
ll0 = 6 * np.log(0.5)
assert (r(ll), r(ll0)) == (-3.6356, -4.1589)
mcf, cs, nag, tjur = pseudo_r2(y, p)
assert np.isclose(mcf, np.log(27 / 16) / np.log(64))
assert np.isclose(cs, 1 - (16 / 27) ** (1 / 3))
assert r((16 / 27) ** (1 / 3)) == 0.8399
assert np.isclose(nag, cs / 0.75)
assert (r(mcf), r(cs), r(nag), r(tjur)) == (0.1258, 0.1601, 0.2134,
                                            0.1667)
assert F(3, 4) + F(1, 2) + F(1, 2) == 3 * F(7, 12)
assert F(1, 4) + F(1, 4) + F(3, 4) == 3 * F(5, 12)
print("Contoh Soal Bab 8: semua bilangan cocok")
