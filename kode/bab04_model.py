"""Bab 4: membaca bobot dan intersep model satu peubah.

Model regresi logistik pada data jam belajar ditaksir dengan
statsmodels dan scikit-learn (cara menaksirnya pokok Bab 5). Bagian
utama membaca intersep, bobot, odds ratio, titik setengah, dan
perubahan peluang untuk kenaikan satu jam di beberapa tempat.
"""
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from bab04_data import jam_belajar
from bab04_sigmoid import sigmoid


def taksir():
    x, y = jam_belajar()
    hasil = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    b, w = hasil.params
    return b, w


if __name__ == "__main__":
    x, y = jam_belajar()
    b, w = taksir()
    sk = LogisticRegression(penalty=None, tol=1e-10,
                            max_iter=1000).fit(x[:, None], y)
    print(f"statsmodels : b = {b:.4f}, w = {w:.4f}")
    print(f"scikit-learn: b = {sk.intercept_[0]:.4f}, "
          f"w = {sk.coef_[0, 0]:.4f}")
    print(f"odds ratio per jam: exp(w) = {np.exp(w):.4f}")
    x50 = -b / w
    print(f"titik setengah: x = -b/w = {x50:.3f} jam")
    print(f"kemiringan di sana: w/4 = {w / 4:.4f} per jam")
    print("  x -> x+1   p(x)    p(x+1)  beda p  odds ratio")
    for a in (1, 5, 10):
        p0, p1 = sigmoid(b + w * a), sigmoid(b + w * (a + 1))
        rasio = (p1 / (1 - p1)) / (p0 / (1 - p0))
        print(f"{a:3d} -> {a + 1:<3d} {p0:7.4f} {p1:7.4f} "
              f"{p1 - p0:7.4f} {rasio:8.4f}")
    print(f"peluang lulus tanpa belajar (x = 0): sigma(b) = "
          f"{sigmoid(b):.4f}")
