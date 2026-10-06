"""Bab 1: alur satu sampel, skor, peluang, odds, dan logit."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, rancang

X, y = data_mini()
A = rancang(X)

print("(1) satu sampel (i = 3) dengan w = (0, 0, 0), eta = 0.5:")
w = np.zeros(3)
x3, y3 = A[2], y[2]
z = x3 @ w
p = expit(z)
ell = -(y3 * np.log(p) + (1 - y3) * np.log(1 - p))
grad = (p - y3) * x3 + 0.0
print(f"    z = {z:.4f}, p = {p:.4f}, loss = {ell:.4f}")
print("    turunan (p - y) x =", grad)
print("    bobot baru =", w - 0.5 * grad)

w = np.array([-2.0, 1.0, -1.0])
z = A @ w
p = expit(z)
print("(2) w = (-2, 1, -1):")
print("     i  x1  x2  y    z       p      odds    logit(p)")
for i in range(6):
    print(f"    {i + 1:2d}  {X[i, 0]:2.0f}  {X[i, 1]:2.0f}  {y[i]:d}"
          f"  {z[i]:5.1f}   {p[i]:.4f}  {p[i] / (1 - p[i]):.4f}"
          f"   {logit(p[i]):5.2f}")

m = LogisticRegression(penalty=None, tol=1e-12).fit(X, y)
w0, (w1, w2) = m.intercept_[0], m.coef_[0]
print("(3) MLE dari scikit-learn (penalty=None):")
print(f"    w0 = {w0:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}")
print(f"    ln 3 * (-2, 1, -1) = {np.round(np.log(3) * np.array([-2, 1, -1]), 4)}")
print(f"    e^w1 = {np.exp(w1):.4f}, e^w2 = {np.exp(w2):.4f},"
      f" e^w0 = {np.exp(w0):.4f}")
print("    peluang:", np.round(m.predict_proba(X)[:, 1], 4))
