"""Bab 8: gradient descent pada data jam belajar.

gd (Listing 8.1) berhenti ketika jarak ke MLE di bawah 1e-6; jarak
itu hanya dapat diukur karena MLE-nya kita ketahui dari statsmodels.
Bagian utama membandingkan x mentah dan x yang dipusatkan, beberapa
laju belajar c / L, dan percepatan Nesterov.
"""
import numpy as np
import statsmodels.api as sm

from bab04_data import jam_belajar
from bab06_turunan import gradien, hessian, rancang


def gd(X, y, eta, theta_hat, tol=1e-6, maks=50_000):
    theta = np.zeros(X.shape[1])
    for k in range(1, maks + 1):
        theta = theta - eta * gradien(theta, X, y)
        if np.linalg.norm(theta - theta_hat) < tol:
            return theta, k
    return theta, None


def nesterov(X, y, eta, theta_hat, tol=1e-6, maks=50_000):
    theta = v = np.zeros(X.shape[1])
    for k in range(1, maks + 1):
        baru = v - eta * gradien(v, X, y)
        v = baru + (k - 1) / (k + 2) * (baru - theta)
        theta = baru
        if np.linalg.norm(theta - theta_hat) < tol:
            return theta, k
    return theta, None


if __name__ == "__main__":
    x, y = jam_belajar()
    for nama, u in (("mentah", x), ("dipusatkan", x - x.mean())):
        X = rancang(u)
        n = len(y)
        L = np.linalg.eigvalsh(X.T @ X / n).max() / 4
        t_hat = sm.Logit(y, X).fit(disp=0).params
        e = np.linalg.eigvalsh(hessian(t_hat, X, y))
        jarak = np.linalg.norm(t_hat)
        print(f"x {nama}: L = {L:.4f}, lambda di MLE = "
              f"({e[0]:.4f}, {e[1]:.4f})")
        kappa = L / e[0]
        print(f"  kappa = L / lambda_min = {kappa:.1f}")
        print(f"  kappa ln(|theta_hat| / 1e-6) = "
              f"{kappa * np.log(jarak / 1e-6):.0f}")
        print(f"  batas stabil di MLE: eta < 2/lambda_max = "
              f"{2 * L / e[1]:.2f}/L")
        for c in (0.5, 1, 2, 4, 8):
            _, k = gd(X, y, c / L, t_hat)
            teks = "tidak sampai dalam 50000" if k is None else k
            print(f"  eta = {c:3g}/L: langkah {teks}")
        _, k = nesterov(X, y, 1 / L, t_hat)
        print(f"  Nesterov, eta = 1/L: langkah {k}")
        if nama == "mentah":
            a, _ = gd(X, y, 4 / L, t_hat)
            b_ = a - 4 / L * gradien(a, X, y)
            c_ = b_ - 4 / L * gradien(b_, X, y)
            print(f"  eta = 4/L, 50000 langkah: w berganti "
                  f"{a[1]:.4f}, {b_[1]:.4f}, {c_[1]:.4f}")
