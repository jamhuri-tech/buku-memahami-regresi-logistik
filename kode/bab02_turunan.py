"""Bab 2: gradien, Hessian, nilai eigen, dan hampiran Taylor.

f(u, v) = exp(u) + exp(-v) + (u - v)^2 / 2   (Hessian definit positif)
g(u, v) = u^4 - 2 u^2 + v^2                   (titik pelana di 0)
s(z)    = log(1 + exp(z)), dihampiri deret Taylor di z0 = 1.
"""
import numpy as np


def f(t):
    u, v = t
    return np.exp(u) + np.exp(-v) + (u - v) ** 2 / 2


def grad_f(t):
    u, v = t
    return np.array([np.exp(u) + (u - v), -np.exp(-v) - (u - v)])


def hess_f(t):
    u, v = t
    return np.array([[np.exp(u) + 1, -1.0], [-1.0, np.exp(-v) + 1]])


def hess_g(t):
    u, v = t
    return np.array([[12 * u ** 2 - 4, 0.0], [0.0, 2.0]])


def beda_hingga(fun, t, h=1e-5):
    """Gradien dengan beda pusat: satu koordinat digeser setiap kali."""
    e = np.eye(len(t))
    return np.array([(fun(t + h * e[i]) - fun(t - h * e[i])) / (2 * h)
                     for i in range(len(t))])


if __name__ == "__main__":
    t = np.array([0.5, -0.3])
    print("(1) f di (0.5, -0.3):")
    print("    gradien rumus      ", np.round(grad_f(t), 6))
    print("    gradien beda hingga", np.round(beda_hingga(f, t), 6))
    H = hess_f(t)
    Hb = np.array([beda_hingga(lambda s: grad_f(s)[i], t) for i in range(2)])
    print("    Hessian rumus      ", np.round(H, 6).tolist())
    print(f"    selisih Hessian beda hingga < 1e-8: "
          f"{np.abs(H - Hb).max() < 1e-8}")
    print("    nilai eigen Hessian", np.round(np.linalg.eigvalsh(H), 4))

    print("(2) nilai eigen Hessian g:")
    for titik in [(0.0, 0.0), (1.0, 0.0), (0.3, 0.0)]:
        ev = np.linalg.eigvalsh(hess_g(np.array(titik)))
        print(f"    di {titik}: {np.round(ev, 4)}")

    print("(3) s(z) = log(1 + e^z) di sekitar z0 = 1:")
    z0 = 1.0
    s0 = np.log1p(np.exp(z0))
    s1 = 1 / (1 + np.exp(-z0))
    s2 = s1 * (1 - s1)
    print(f"    s = {s0:.4f}, s' = {s1:.4f}, s'' = {s2:.4f}")
    print("       h   sebenarnya  orde satu  orde dua  galat orde dua")
    for h in [0.1, 0.5, 1.0, 2.0, -2.0]:
        benar = np.log1p(np.exp(z0 + h))
        satu = s0 + s1 * h
        dua = satu + s2 * h ** 2 / 2
        print(f"    {h:4.1f}   {benar:.4f}      {satu:.4f}     {dua:.4f}"
              f"    {dua - benar:+.1e}")
