"""Bab 6: persamaan skor X^T (p - y) = 0 di MLE, dan penalti.

Di MLE, jumlah peluang sama dengan banyaknya kelas 1, dan setiap
peubah memenuhi sum x_ij p_i = sum x_ij y_i. Penalti L2 scikit-learn
merusak persamaan peubah menjadi X^T (p - y) = -w / C, tetapi tidak
merusak persamaan intersep, karena intersep tidak dipenalti;
liblinear ikut memenalti intersep sehingga persamaan itu pun rusak.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from bab04_data import W_BENAR, B_BENAR, dua_peubah

if __name__ == "__main__":
    X, y = dua_peubah()
    print(f"data dua peubah: n = {len(y)}, sum y = {y.sum()}")
    print(f"parameter benar: w = ({W_BENAR[0]:g}, {W_BENAR[1]:g}), "
          f"b = {B_BENAR:g}")
    model = [
        ("tanpa penalti", dict(penalty=None)),
        ("L2, C = 1", dict(C=1.0)),
        ("L2, C = 0.01", dict(C=0.01)),
        ("L2, C = 0.01, liblinear", dict(C=0.01, solver="liblinear")),
    ]
    for nama, opsi in model:
        m = LogisticRegression(tol=1e-10, max_iter=5000, **opsi)
        m.fit(X, y)
        p = m.predict_proba(X)[:, 1]
        g = np.round(X.T @ (p - y), 3) + 0.0
        w, b = m.coef_[0], m.intercept_[0]
        print(f"{nama}:")
        print(f"  w = ({w[0]:.3f}, {w[1]:.3f}), b = {b:.3f}, "
              f"sum p = {p.sum():.3f}")
        baris = f"  X^T(p - y) = ({g[0]:.3f}, {g[1]:.3f})"
        if "C" in opsi:
            C = opsi["C"]
            baris += f", -w/C = ({-w[0] / C:.3f}, {-w[1] / C:.3f})"
        print(baris)
