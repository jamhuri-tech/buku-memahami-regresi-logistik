"""Bab 11: data dengan banyak peubah yang sebagian besar tidak berguna.

data_jarang(): 50 peubah normal baku; hanya lima yang benar-benar
memengaruhi label, dengan bobot (2, -2, 1.5, -1.5, 1) dan intersep 0.
Data latih 100 titik, data uji 10 ribu titik dari model yang sama.
"""
import numpy as np
from scipy.special import expit

from bab04_data import BENIH

D = 50
W_JARANG = np.r_[2.0, -2.0, 1.5, -1.5, 1.0, np.zeros(D - 5)]


def data_jarang(n_latih=100, n_uji=10_000, benih=BENIH):
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(n_latih + n_uji, D))
    y = (rng.random(len(X)) < expit(X @ W_JARANG)).astype(int)
    return X[:n_latih], y[:n_latih], X[n_latih:], y[n_latih:]


if __name__ == "__main__":
    X, y, Xu, yu = data_jarang()
    print(f"latih: {X.shape}, kelas 1 = {y.sum()}; "
          f"uji: {Xu.shape}, kelas 1 = {yu.sum()}")
