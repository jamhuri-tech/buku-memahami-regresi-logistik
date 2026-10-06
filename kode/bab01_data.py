"""Data mini yang dipakai di seluruh buku, dan matriks rancangannya.

Enam mahasiswa (sampel i = 1, ..., 6), dua fitur, satu target:
  x1 = jam belajar per minggu, x2 = banyak absen, y = lulus (1/0).
Kedua kelas tidak dapat dipisahkan oleh garis apa pun, sehingga
taksiran kemungkinan maksimum (MLE) ada. MLE-nya tepat
w = ln 3 * (-2, 1, -1), dengan peluang 1/4, 1/2, atau 3/4.
"""
import numpy as np

BENIH = 20261006

TABEL = np.array([
    # x1  x2  y
    [1, 0, 0],
    [2, 1, 0],
    [3, 0, 1],
    [4, 2, 1],
    [5, 3, 1],
    [6, 3, 0],
])


def data_mini():
    """Matriks fitur X (6 x 2) dan vektor target y (6,)."""
    return TABEL[:, :2].astype(float), TABEL[:, 2].copy()


def rancang(X):
    """Menambahkan kolom x0 = 1: X berukuran m x (n + 1)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


def mle_mini():
    """MLE data mini (w0, w1, w2) dengan metode Newton dari w = 0."""
    X, y = data_mini()
    A = rancang(X)
    w = np.zeros(3)
    for _ in range(20):
        p = 1 / (1 + np.exp(-(A @ w)))
        g = A.T @ (p - y)
        H = A.T @ (A * (p * (1 - p))[:, None])
        w = w - np.linalg.solve(H, g)
    return w


if __name__ == "__main__":
    X, y = data_mini()
    print("X:\n", X)
    print("y:", y)
    print("MLE:", mle_mini(), np.log(3) * np.array([-2, 1, -1]))
