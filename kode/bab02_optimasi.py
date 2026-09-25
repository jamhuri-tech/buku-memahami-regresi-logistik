"""Bab 2: gradient descent dan metode Newton pada satu peubah.

f(t) = log(1 + e^t) - 0.7 t, dengan f'(t) = sigma(t) - 0.7 dan
f''(t) = sigma(t)(1 - sigma(t)) <= 1/4. Minimumnya di sigma(t) = 0.7,
yaitu t* = log(0.7 / 0.3).
"""
import numpy as np
from scipy.special import expit

T_BINTANG = np.log(0.7 / 0.3)


def turunan(t):
    s = expit(t)
    return s - 0.7, s * (1 - s)


def gd(t, eta, k):
    jejak = [t]
    for _ in range(k):
        t = t - eta * turunan(t)[0]
        jejak.append(t)
    return np.array(jejak)


def newton(t, k):
    jejak = [t]
    for _ in range(k):
        g, h = turunan(t)
        t = t - g / h
        jejak.append(t)
    return np.array(jejak)


if __name__ == "__main__":
    print(f"(1) minimum t* = log(0.7/0.3) = {T_BINTANG:.6f}")
    print("    galat |t - t*| dari t0 = 0:")
    print("     k    GD (eta = 4)   Newton")
    a, b = gd(0.0, 4.0, 6), newton(0.0, 6)
    for k in range(7):
        ga, gb = abs(a[k] - T_BINTANG), abs(b[k] - T_BINTANG)
        teks = f"{gb:.2e}" if gb > 1e-12 else "< 1e-12"
        print(f"    {k:2d}    {ga:.2e}      {teks}")
    for eta in [1.0, 4.0, 8.0]:
        j = gd(0.0, eta, 5000)
        k = np.argmax(np.abs(j - T_BINTANG) < 1e-10)
        print(f"    GD eta = {eta:g}: galat < 1e-10 di langkah {k}")
    print("(2) Newton dari t0 = 4:")
    j = newton(4.0, 2)
    print("    " + "  ->  ".join(f"{v:.4g}" for v in j))
    j = gd(4.0, 4.0, 5000)
    k = np.argmax(np.abs(j - T_BINTANG) < 1e-10)
    print(f"    GD eta = 4 dari t0 = 4: galat < 1e-10 di langkah {k}")
