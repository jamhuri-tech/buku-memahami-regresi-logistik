"""Bab 4: fungsi logistik dan sifat-sifatnya.

Fungsi sigmoid (Listing 4.1) menghitung fungsi logistik tanpa
meluap. Bagian utama memeriksa sifat-sifat fungsi itu secara numerik
dan membandingkan rumus polos dengan rumus stabil.
"""
import warnings

import numpy as np
from scipy.special import expit


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    hasil = np.empty_like(z)
    pos = z >= 0
    hasil[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])            # z < 0: exp(z) < 1, aman
    hasil[~pos] = ez / (1 + ez)
    return hasil


def sigmoid_polos(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


if __name__ == "__main__":
    z = np.linspace(-10, 10, 2001)
    s = sigmoid(z)
    print("(1) nilai di beberapa titik:")
    for t in (-4, -2, -1, 0, 1, 2, 4):
        print(f"    sigma({t:+d}) = {sigmoid(t):.4f}")
    print(f"(2) maks |sigma(-z) - (1 - sigma(z))| = "
          f"{np.max(np.abs(sigmoid(-z) - (1 - s))):.1e}")
    h = 1e-5
    beda = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
    print(f"(3) maks |beda hingga - sigma(1-sigma)| = "
          f"{np.max(np.abs(beda - s * (1 - s))):.1e}")
    print(f"    turunan terbesar {np.max(s * (1 - s)):.4f} "
          f"di z = {z[np.argmax(s * (1 - s))]:.1f}")
    for t in (0.5, 1.0, 2.0):
        print(f"    z = {t}: sigma = {sigmoid(t):.4f}, "
              f"1/2 + z/4 = {0.5 + t / 4:.4f}")
    print("(4) rumus polos lawan rumus stabil:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        polos = sigmoid_polos(-720.0)
        print(f"    polos(-720)  = {polos}, peringatan: "
              f"{w[0].category.__name__ if w else '-'}")
    print(f"    stabil(-720) = {sigmoid(-720.0):.3e}")
    print(f"    expit(-720)  = {expit(-720.0):.3e}")
    print(f"    maks |sigmoid - expit| = "
          f"{np.max(np.abs(sigmoid(z) - expit(z))):.1e}")
    print("(5) peluang kelas 0 untuk z = 40:")
    print(f"    1 - sigma(40) = {1 - sigmoid(40.0):.3e}")
    print(f"    sigma(-40)    = {sigmoid(-40.0):.3e}")
