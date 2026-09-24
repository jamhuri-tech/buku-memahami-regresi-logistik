"""Bab 5: mengapa kuadrat galat tidak cocok untuk regresi logistik.

(1) Gradien terhadap skor z untuk satu titik berlabel 0: log-loss
    memberi sigma(z) - y, kuadrat galat 2 (sigma - y) sigma (1 - sigma).
(2) Gradient descent dengan laju belajar sama untuk kedua loss pada
    data jam belajar, dari dua titik awal.
(3) Nilai eigen terkecil Hessian (beda hingga): kuadrat galat
    mempunyai nilai eigen negatif, log-loss tidak.
"""
import numpy as np
from scipy.special import expit

from bab04_data import jam_belajar

x, y = jam_belajar()
X = np.column_stack([np.ones_like(x), x])


def log_loss(t):
    z = X @ t
    return np.mean(np.logaddexp(0, z) - y * z)


def grad_log_loss(t):
    return X.T @ (expit(X @ t) - y) / len(y)


def kuadrat(t):
    return np.mean((expit(X @ t) - y) ** 2)


def grad_kuadrat(t):
    p = expit(X @ t)
    return X.T @ (2 * (p - y) * p * (1 - p)) / len(y)


def gd(grad, awal, eta=0.05, langkah=(0, 1000, 20000, 100000)):
    t, catat = np.array(awal, dtype=float), {}
    for k in range(max(langkah) + 1):
        if k in langkah:
            catat[k] = t.copy()
        t = t - eta * grad(t)
    return catat


def eigen_min(f, t, h=1e-4):
    H, E = np.zeros((2, 2)), np.eye(2)
    for i in range(2):
        for j in range(2):
            H[i, j] = (f(t + h * E[i] + h * E[j])
                       - f(t + h * E[i] - h * E[j])
                       - f(t - h * E[i] + h * E[j])
                       + f(t - h * E[i] - h * E[j])) / (4 * h * h)
    return np.linalg.eigvalsh(H)[0]


if __name__ == "__main__":
    print("(1) label 0, gradien terhadap z:")
    print("     z   log-loss  kuadrat")
    for z in (0.0, 2.0, 5.0, 10.0):
        p = expit(z)
        print(f"  {z:4.0f}   {p:.5f}  {2 * p * p * (1 - p):.5f}")
    print("(2) gradient descent, eta = 0.05:")
    for awal in ((0.0, 0.0), (10.0, -2.0)):
        for nama, f, g in (("log-loss", log_loss, grad_log_loss),
                           ("kuadrat ", kuadrat, grad_kuadrat)):
            print(f"  awal {awal}, {nama}:")
            for k, t in gd(g, awal).items():
                print(f"    langkah {k:6d}: loss {f(t):.4f}, "
                      f"b = {t[0]:7.3f}, w = {t[1]:6.3f}")
    for nama, f, g in (("log-loss", log_loss, grad_log_loss),
                       ("kuadrat ", kuadrat, grad_kuadrat)):
        t, k = np.array([10.0, -2.0]), 0
        batas = 1.01 * f(gd(g, (0.0, 0.0))[100000])
        while f(t) > batas:
            t, k = t - 0.05 * g(t), k + 1
        print(f"  (10, -2) {nama}: dalam 1% dari minimum "
              f"pada langkah {k}")
    print("(3) nilai eigen terkecil Hessian:")
    for t in ((0.0, 0.0), (4.0, -1.0), (10.0, -2.0)):
        t = np.array(t)
        print(f"  (b, w) = ({t[0]:4.0f}, {t[1]:3.0f}): log-loss "
              f"{eigen_min(log_loss, t):+.5f}, kuadrat "
              f"{eigen_min(kuadrat, t):+.5f}")
