"""Bab 9: solver quasi-Newton dan solver scikit-learn.

(1) scipy.optimize.minimize dengan BFGS, L-BFGS-B, dan Newton-CG pada
    jam belajar mentah dan data sintetis sepuluh ribu titik.
(2) Semua solver LogisticRegression tanpa penalti (liblinear dengan
    C = 1e12, karena liblinear tidak menerima penalty=None).
Setiap hasil dibandingkan dengan MLE statsmodels.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from bab04_data import jam_belajar
from bab06_turunan import gradien, hessian, log_loss, rancang
from bab08_sgd import data_besar


def data():
    x, y = jam_belajar()
    return {"jam belajar mentah": (rancang(x), y.astype(float)),
            "sintetis 10 ribu": data_besar()}


if __name__ == "__main__":
    for nama, (X, y) in data().items():
        ref = sm.Logit(y, X).fit(disp=0).params
        print(f"{nama}:")
        print("  scipy          iterasi  evaluasi  |theta - MLE| < 1e-6")
        for m in ("BFGS", "L-BFGS-B", "Newton-CG"):
            opsi = {"hess": hessian} if m == "Newton-CG" else {}
            r = minimize(log_loss, np.zeros(X.shape[1]), args=(X, y),
                         jac=gradien, method=m, **opsi)
            cocok = np.abs(r.x - ref).max() < 1e-6
            print(f"  {m:14s} {r.nit:7d}  {r.nfev:8d}  {cocok}")
        print("  scikit-learn     n_iter_  |theta - MLE| < 1e-6")
        for s in ("lbfgs", "newton-cg", "newton-cholesky", "sag",
                  "saga", "liblinear"):
            opsi = ({"C": 1e12, "intercept_scaling": 100}
                    if s == "liblinear" else {"penalty": None})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(solver=s, tol=1e-10,
                                       max_iter=10_000, random_state=0,
                                       **opsi).fit(X[:, 1:], y)
            t = np.r_[m.intercept_, m.coef_[0]]
            cocok = np.abs(t - ref).max() < 1e-6
            print(f"  {s:16s} {m.n_iter_[0]:7d}  {cocok}")
