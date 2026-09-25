"""Bab 3: minimize, LogisticRegression, statsmodels, dan benih acak.

Satu model, jam belajar, ditaksir dengan tiga cara. Lalu pengaruh
penalti bawaan scikit-learn, dan data sintetis make_classification.
"""

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from bab04_data import BENIH, jam_belajar

x, y = jam_belajar()
X = np.column_stack([np.ones(len(x)), x])


def loss(theta):
    z = X @ theta
    return np.mean(np.logaddexp(0, z) - y * z)


def grad(theta):
    p = 1 / (1 + np.exp(-(X @ theta)))
    return X.T @ (p - y) / len(y)


print("(1) jam belajar, tiga cara:               b         w")
h = minimize(loss, np.zeros(2), jac=grad, method="BFGS",
             options={"gtol": 1e-10})
print(f"    scipy minimize (BFGS)            {h.x[0]:8.4f}  {h.x[1]:8.4f}")
m = LogisticRegression(penalty=None, tol=1e-10).fit(x[:, None], y)
print(f"    LogisticRegression(penalty=None) "
      f"{m.intercept_[0]:8.4f}  {m.coef_[0, 0]:8.4f}")
s = sm.Logit(y, X).fit(disp=0)
print(f"    statsmodels Logit                {s.params[0]:8.4f}"
      f"  {s.params[1]:8.4f}")
d = LogisticRegression().fit(x[:, None], y)
print(f"    LogisticRegression() bawaan      "
      f"{d.intercept_[0]:8.4f}  {d.coef_[0, 0]:8.4f}")
print(f"    penalti bawaan: {d.penalty}, C = {d.C}")

print("(2) benih acak:")
a = np.random.default_rng(BENIH).normal(size=3)
b = np.random.default_rng(BENIH).normal(size=3)
c = np.random.default_rng(BENIH + 1).normal(size=3)
print("    benih sama:", np.round(a, 4))
print("    benih sama:", np.round(b, 4))
print("    benih lain:", np.round(c, 4))

print("(3) make_classification:")
Xs, ys = make_classification(n_samples=1000, n_features=5,
                             n_informative=3, n_redundant=1,
                             weights=[0.8], random_state=0)
print(f"    X {Xs.shape}, kelas 1: {ys.sum()} dari {len(ys)}")
r = np.linalg.matrix_rank(Xs)
print(f"    pangkat X = {r} (satu kolom kombinasi linear kolom lain)")
