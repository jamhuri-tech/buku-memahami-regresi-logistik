"""Bab 18: regresi softmax dari nol.

log_softmax, entropi silang, dan gradien (Listing 18.1) dipakai
L-BFGS dengan dua parameterisasi: kelas 0 sebagai rujukan (dicocokkan
dengan statsmodels MNLogit) dan simetris dari titik nol (dicocokkan
dengan LogisticRegression tanpa penalti).
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from bab06_turunan import rancang
from bab18_data import tiga_kelas

warnings.simplefilter("ignore")


def log_softmax(Z):
    return Z - logsumexp(Z, axis=1, keepdims=True)


def loss_gradien(Theta, X, Y):
    """Cross-entropy total dan gradiennya (Theta: K x m)."""
    L = log_softmax(X @ Theta.T)
    P = np.exp(L)
    return -np.sum(Y * L), (P - Y).T @ X


def latih_softmax(X, y, K, rujukan=True):
    Xt, Y = rancang(X), np.eye(K)[y]
    m = Xt.shape[1]
    k_bebas = K - 1 if rujukan else K

    def f(v):
        Theta = v.reshape(k_bebas, m)
        if rujukan:
            Theta = np.vstack([np.zeros(m), Theta])
        nilai, g = loss_gradien(Theta, Xt, Y)
        return nilai, (g[1:] if rujukan else g).ravel()

    r = minimize(f, np.zeros(k_bebas * m), jac=True, method="L-BFGS-B",
                 options={"gtol": 1e-10, "ftol": 1e-15})
    return r.x.reshape(k_bebas, m), r.fun


if __name__ == "__main__":
    X, y, _ = tiga_kelas()
    T_ruj, f_ruj = latih_softmax(X, y, 3, rujukan=True)
    mn = sm.MNLogit(y, sm.add_constant(X)).fit(disp=0)
    print("(1) kelas 0 sebagai rujukan (baris: kelas 1, 2; b, w1, w2):")
    for k in range(2):
        print(f"    kelas {k + 1}: {np.round(T_ruj[k], 4)}")
    print(f"    maks beda dengan MNLogit < 1e-5: "
          f"{np.abs(T_ruj - mn.params.T).max() < 1e-5}")
    print(f"    log-kemungkinan kita {-f_ruj:.4f}, MNLogit {mn.llf:.4f}")
    T_sim, _ = latih_softmax(X, y, 3, rujukan=False)
    sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000)
    sk.fit(X, y)
    T_sk = np.column_stack([sk.intercept_, sk.coef_])
    print("(2) simetris dari nol:")
    print(f"    jumlah atas kelas (kita): "
          f"{np.round(T_sim.sum(axis=0), 8) + 0.0}")
    print(f"    maks beda dengan sklearn < 1e-5: "
          f"{np.abs(T_sim - T_sk).max() < 1e-5}")
    beda = T_sim[1:] - T_sim[0]
    print(f"    selisih terhadap kelas 0 = rujukan < 1e-5: "
          f"{np.abs(beda - T_ruj).max() < 1e-5}")
