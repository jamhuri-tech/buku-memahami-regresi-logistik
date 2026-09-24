"""Bab 6: kecembungan log-loss diperiksa secara numerik.

(1) Nilai eigen terkecil Hessian pada 10 ribu titik acak.
(2) Sepanjang sinar theta = t * MLE, kelengkungan meluruh ke nol:
    cembung tegas, tetapi tidak kuat cembung.
(3) Kolom yang bergantung linear (x dan 2x): Hessian singular dan
    minimumnya tidak tunggal.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from bab04_data import BENIH, jam_belajar
from bab04_model import taksir
from bab06_turunan import hessian, log_loss, rancang

if __name__ == "__main__":
    x, y = jam_belajar()
    X = rancang(x)
    rng = np.random.default_rng(BENIH)
    T = np.column_stack([rng.uniform(-10, 10, 10_000),
                         rng.uniform(-3, 3, 10_000)])
    lam = np.array([np.linalg.eigvalsh(hessian(t, X, y))[0]
                    for t in T])
    print(f"(1) 10000 titik acak: nilai eigen terkecil Hessian")
    print(f"    minimum {lam.min():.2e}, maksimum {lam.max():.2e}, "
          f"negatif: {np.sum(lam < 0)}")
    mle = np.array(taksir())
    print("(2) sinar theta = t * MLE:")
    print("     t   log-loss   eigen terkecil  eigen terbesar")
    for t in (1, 2, 5, 10, 20):
        e = np.linalg.eigvalsh(hessian(t * mle, X, y))
        print(f"    {t:2d}   {log_loss(t * mle, X, y):8.4f}"
              f"   {e[0]:14.2e}  {e[1]:14.2e}")
    X3 = np.column_stack([np.ones_like(x), x, 2 * x])
    print("(3) kolom x dan 2x:")
    try:
        sm.Logit(y, X3).fit(disp=0)
    except np.linalg.LinAlgError as galat:
        print(f"    statsmodels: LinAlgError ({galat})")
    sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000)
    sk.fit(X3[:, 1:], y)
    w1, w2 = sk.coef_[0]
    print(f"    scikit-learn: w1 = {w1:.4f}, w2 = {w2:.4f}, "
          f"w1 + 2 w2 = {w1 + 2 * w2:.4f}")
    b, w = mle
    for t in (np.array([b, w, 0]), np.array([b, 0, w / 2]),
              np.array([b, 3 * w, -w])):
        print(f"    (w1, w2) = ({t[1]:7.4f}, {t[2]:7.4f}): log-loss "
              f"{log_loss(t, X3, y):.6f}")
    e = np.linalg.eigvalsh(hessian(np.array([b, w, 0]), X3, y))
    print(f"    nilai eigen Hessian: {e[1]:.4f}, {e[2]:.4f}, dan")
    print(f"    satu lagi yang |.| < 1e-12: {abs(e[0]) < 1e-12}")
