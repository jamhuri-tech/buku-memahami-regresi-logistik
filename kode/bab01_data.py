"""Data mini yang dipakai di seluruh buku, dan matriks rancangannya.

Enam mahasiswa: jam belajar x = 1, ..., 6 dan lulus (1) atau tidak
(0). Kedua kelas tumpang tindih di x = 3 dan x = 4, sehingga taksiran
kemungkinan maksimum ada. Data simetris terhadap x = 3,5 (x -> 7 - x
sekaligus y -> 1 - y), sehingga titik setengah MLE tepat 3,5.
"""
import numpy as np

BENIH = 20261005


def data_mini():
    x = np.arange(1.0, 7.0)
    y = np.array([0, 0, 1, 0, 1, 1])
    return x, y


def rancang(x):
    """Matriks rancangan: kolom satu untuk intersep, lalu peubah."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def mle_mini():
    """MLE data mini, theta = (b, w), dari scikit-learn tanpa penalti."""
    from sklearn.linear_model import LogisticRegression
    x, y = data_mini()
    m = LogisticRegression(penalty=None, tol=1e-12).fit(x[:, None], y)
    return np.array([m.intercept_[0], m.coef_[0, 0]])


if __name__ == "__main__":
    x, y = data_mini()
    print("x:", x)
    print("y:", y)
    print("X:\n", rancang(x))
