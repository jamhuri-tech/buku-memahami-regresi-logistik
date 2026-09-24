"""Bab 9: metode Newton pada data jam belajar.

newton (Listing 9.1) adalah Newton murni; newton_teredam (Listing
9.2) memotong panjang langkah menjadi separuhnya sampai log-loss
turun cukup. Bagian utama mengukur galat setiap iterasi untuk x
mentah dan x dipusatkan, lalu mencoba titik awal yang jauh.
"""
import numpy as np
import statsmodels.api as sm

from bab04_data import jam_belajar
from bab06_turunan import gradien, hessian, log_loss, rancang


def newton(X, y, theta, iterasi):
    jalur = [theta]
    for _ in range(iterasi):
        arah = np.linalg.solve(hessian(theta, X, y),
                               gradien(theta, X, y))
        theta = theta - arah
        jalur.append(theta)
    return jalur


def newton_teredam(X, y, theta, tol=1e-10, maks=100):
    for k in range(1, maks + 1):
        g = gradien(theta, X, y)
        arah = np.linalg.solve(hessian(theta, X, y), g)
        t, f, j = 1.0, log_loss(theta, X, y), g @ arah / 4
        while log_loss(theta - t * arah, X, y) > f - t * j:
            t /= 2
        theta = theta - t * arah
        if np.linalg.norm(gradien(theta, X, y)) < tol:
            return theta, k
    return theta, None


if __name__ == "__main__":
    x, y = jam_belajar()
    Xm, Xp = rancang(x), rancang(x - x.mean())
    tm = sm.Logit(y, Xm).fit(disp=0).params
    jm = newton(Xm, y, np.zeros(2), 6)
    jp = newton(Xp, y, np.zeros(2), 6)
    print(" k   log-loss mentah  log-loss dipusat  |theta_k - MLE|")
    for k, (a, b) in enumerate(zip(jm, jp)):
        galat = np.linalg.norm(a - tm)
        teks = f"{galat:9.1e}" if galat > 1e-12 else "  < 1e-12"
        print(f"{k:2d}   {log_loss(a, Xm, y):.12f}  "
              f"{log_loss(b, Xp, y):.12f}  {teks}")
    print("Newton murni dari (b, w) = (10, -2):")
    theta = np.array([10.0, -2.0])
    theta = theta - np.linalg.solve(hessian(theta, Xm, y),
                                    gradien(theta, Xm, y))
    print(f"  iterasi 1: b = {theta[0]:.2f}, w = {theta[1]:.2f}")
    try:
        theta = theta - np.linalg.solve(hessian(theta, Xm, y),
                                        gradien(theta, Xm, y))
        meledak = np.linalg.norm(theta) > 1e15
    except np.linalg.LinAlgError:          # Hessian tepat singular
        meledak = True
    print(f"  iterasi 2: |theta| > 1e15 atau H singular "
          f"(meledak): {meledak}")
    theta, k = newton_teredam(Xm, y, np.array([10.0, -2.0]))
    print(f"teredam dari (10, -2): {k} iterasi, "
          f"b = {theta[0]:.4f}, w = {theta[1]:.4f}")
