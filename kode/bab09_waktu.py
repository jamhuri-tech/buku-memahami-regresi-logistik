"""Bab 9: biaya satu gradien lawan satu langkah Newton.

Waktu jalan tidak deterministik, sehingga keluaran program ini tidak
dicetak di naskah; angkanya hanya dikutip di prosa. Data: n = 20 ribu
titik, d peubah normal baku, label dari model regresi logistik.
"""
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from bab04_data import BENIH
from bab06_turunan import gradien, hessian


def ukur(f, ulang=5):
    t = time.perf_counter()
    for _ in range(ulang):
        f()
    return (time.perf_counter() - t) / ulang


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    n = 20_000
    for d in (10, 100, 500):
        X = np.column_stack([np.ones(n), rng.normal(size=(n, d))])
        th = rng.normal(size=d + 1) / np.sqrt(d)
        y = (rng.random(n) < expit(X @ th)).astype(float)
        t0 = np.zeros(d + 1)
        tg = ukur(lambda: gradien(t0, X, y))
        tn = ukur(lambda: np.linalg.solve(hessian(t0, X, y),
                                          gradien(t0, X, y)))
        waktu = {}
        for s in ("lbfgs", "newton-cholesky"):
            m = LogisticRegression(penalty=None, solver=s, tol=1e-8,
                                   max_iter=1000)
            t = time.perf_counter()
            m.fit(X[:, 1:], y)
            waktu[s] = (time.perf_counter() - t, m.n_iter_[0])
        print(f"d = {d:3d}: gradien {tg * 1e3:7.2f} ms, langkah Newton "
              f"{tn * 1e3:8.2f} ms ({tn / tg:5.1f} x)")
        for s, (t, k) in waktu.items():
            print(f"         {s:16s} {t * 1e3:8.1f} ms, {k} iterasi")
