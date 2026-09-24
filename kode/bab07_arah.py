"""Bab 7: gradient descent pada data terpisah menuju margin maksimum.

Gradient descent dengan laju 1/L pada gumpalan_terpisah(): norma
theta tumbuh seperti ||theta_svm|| log t, dan arahnya mendekati arah
pemisah bermargin maksimum (Soudry dkk., 2018). Pemisah itu dihitung
sebagai soal kuadratik: minimumkan ||theta||^2 / 2 dengan syarat
s_i x_i^T theta >= 1, dengan intersep ikut di dalam theta.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from bab06_turunan import rancang
from bab07_data import gumpalan_terpisah

LANGKAH = np.unique(np.geomspace(1, 10**6, 25).astype(int))


def margin_maksimum(X, y):
    A = (2 * y - 1)[:, None] * rancang(X)
    syarat = {"type": "ineq", "fun": lambda t: A @ t - 1,
              "jac": lambda t: A}
    r = minimize(lambda t: 0.5 * t @ t, np.zeros(A.shape[1]),
                 jac=lambda t: t, constraints=[syarat],
                 method="SLSQP", options={"ftol": 1e-14})
    return r.x


def jalankan_gd(X, y, langkah=LANGKAH):
    Xt = rancang(X)
    n = len(y)
    eta = 4 / np.linalg.eigvalsh(Xt.T @ Xt / n).max()
    theta, catat = np.zeros(Xt.shape[1]), {}
    for k in range(1, langkah.max() + 1):
        theta = theta - eta * Xt.T @ (expit(Xt @ theta) - y) / n
        if k in langkah:
            catat[k] = theta.copy()
    return catat


if __name__ == "__main__":
    X, y = gumpalan_terpisah()
    svm = margin_maksimum(X, y)
    u = svm / np.linalg.norm(svm)
    print(f"margin maksimum: theta = ({svm[0]:.4f}, {svm[1]:.4f}, "
          f"{svm[2]:.4f})")
    print(f"  ||theta_svm|| = {np.linalg.norm(svm):.4f}, "
          f"||theta_svm|| ln 10 = {np.linalg.norm(svm) * np.log(10):.4f}")
    catat = jalankan_gd(X, y)
    print("  langkah t   ||theta(t)||   sudut ke svm (derajat)")
    for k in (10, 100, 1000, 10**4, 10**5, 10**6):
        th = catat[k]
        nr = np.linalg.norm(th)
        sudut = np.degrees(np.arccos(min(1.0, th @ u / nr)))
        print(f"  {k:9d}   {nr:10.3f}   {sudut:10.3f}")
    naik = np.linalg.norm(catat[10**6]) - np.linalg.norm(catat[10**5])
    print(f"  kenaikan norma dari 1e5 ke 1e6 langkah: {naik:.4f}")
