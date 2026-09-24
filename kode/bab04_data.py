"""Data contoh berjalan untuk Bagian II (Bab 4-7).

jam_belajar(): 16 mahasiswa, jam belajar per minggu dan lulus (1)
atau tidak (0). Data rekaan, kecil, dan sengaja tumpang tindih
supaya penaksir kemungkinan maksimumnya ada.

dua_peubah(n, benih): data sintetis dua peubah dari model regresi
logistik yang parameternya kita ketahui: w = (2, -1), b = 0.5.
"""
import numpy as np

BENIH = 20260924

JAM = np.array([1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11],
               dtype=float)
LULUS = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])

W_BENAR = np.array([2.0, -1.0])
B_BENAR = 0.5


def jam_belajar():
    return JAM.copy(), LULUS.copy()


def dua_peubah(n=200, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-(X @ W_BENAR + B_BENAR)))
    y = (rng.random(n) < p).astype(int)
    return X, y


if __name__ == "__main__":
    x, y = jam_belajar()
    print("jam  :", " ".join(f"{v:g}" for v in x))
    print("lulus:", " ".join(str(v) for v in y))
    X, y2 = dua_peubah()
    print(f"dua peubah: n = {len(y2)}, kelas 1 = {y2.sum()}")
