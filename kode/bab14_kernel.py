"""Bab 14: regresi logistik kernel dari nol, lalu lawan SVM.

logistik_kernel (Listing 14.1) meminimumkan 1/2 a^T K a + C sum loss
dengan skor K a + b dan kernel RBF, memakai Newton. Hasilnya sama
dengan Nystroem(n_components = n) yang diikuti LogisticRegression.
Sesudah itu regresi logistik RBF dibandingkan dengan SVC pada data
bulan.
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from bab14_data import bulan

warnings.simplefilter("ignore")
GAMMA, C = 1.0, 1.0


def logistik_kernel(K, y, C, iterasi=30):
    n = len(y)
    a, b = np.zeros(n), 0.0
    for _ in range(iterasi):
        p = expit(K @ a + b)
        s = C * (p - y)
        g = np.r_[K @ (a + s), s.sum()]     # gradien (a, b)
        d = C * p * (1 - p)
        KD = K * d
        u = KD.sum(axis=1)
        H = np.block([[KD @ K + K, u[:, None]],
                      [u[None, :], np.array([[d.sum()]])]])
        langkah = np.linalg.lstsq(H, g, rcond=None)[0]
        a, b = a - langkah[:-1], b - langkah[-1]
    return a, b


if __name__ == "__main__":
    X, y = bulan(200)
    Xu, yu = bulan(5000, benih=1)
    K = rbf_kernel(X, X, gamma=GAMMA)
    a, b = logistik_kernel(K, y, C)
    p_kita = expit(rbf_kernel(Xu, X, gamma=GAMMA) @ a + b)
    ny = make_pipeline(Nystroem(gamma=GAMMA, n_components=len(y),
                                random_state=0),
                       LogisticRegression(C=C, tol=1e-10,
                                          max_iter=10_000)).fit(X, y)
    p_ny = ny.predict_proba(Xu)[:, 1]
    print("(1) kernel kita lawan Nystroem(200 komponen) + LR:")
    print(f"    maks beda peluang < 1e-4: "
          f"{np.abs(p_kita - p_ny).max() < 1e-4}")
    print(f"    log-loss uji {log_loss(yu, p_kita):.4f}, akurasi "
          f"{accuracy_score(yu, p_kita > 0.5):.4f}")
    for k in (10, 30, 100):
        m = make_pipeline(Nystroem(gamma=GAMMA, n_components=k,
                                   random_state=0),
                          LogisticRegression(C=C, max_iter=10_000))
        m.fit(X, y)
        print(f"    Nystroem {k:3d} komponen: log-loss uji "
              f"{log_loss(yu, m.predict_proba(Xu)):.4f}")
    svm = SVC(kernel="rbf", gamma=GAMMA, C=C, probability=True,
              random_state=0).fit(X, y)
    print("(2) data bulan, kernel RBF, gamma = 1, C = 1:")
    print(f"    SVC      : akurasi {accuracy_score(yu, svm.predict(Xu)):.3f}"
          f", log-loss (Platt) "
          f"{log_loss(yu, svm.predict_proba(Xu)):.4f}")
    print(f"    logistik : akurasi {accuracy_score(yu, p_kita > 0.5):.3f}"
          f", log-loss {log_loss(yu, p_kita):.4f}")
    print(f"    vektor pendukung SVC: {len(svm.support_)} dari {len(y)}")
    print(f"    koefisien a_i logistik yang |a_i| < 1e-6: "
          f"{int((np.abs(a) < 1e-6).sum())} dari {len(y)}")
