"""Bab 1: skor, peluang, odds, dan logit pada data mini."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from bab01_data import data_mini, rancang

x, y = data_mini()
X = rancang(x)

theta = np.array([-2.8, 0.8])
z = X @ theta
p = expit(z)
print("(1) theta = (-2.8, 0.8):")
print("     i   x   y      z        p      odds    logit(p)")
for i in range(6):
    print(f"    {i + 1:2d}  {x[i]:2.0f}  {y[i]:2d}  {z[i]:6.2f}"
          f"   {p[i]:.4f}   {p[i] / (1 - p[i]):6.4f}"
          f"   {logit(p[i]):6.2f}")

m = LogisticRegression(penalty=None, tol=1e-10).fit(x[:, None], y)
b, w = m.intercept_[0], m.coef_[0, 0]
print("(2) MLE dari scikit-learn (penalty=None):")
print(f"    b = {b:.4f}, w = {w:.4f}")
print(f"    e^w = {np.exp(w):.4f}, -b/w = {-b / w:.4f},"
      f" w/4 = {w / 4:.4f}")
print("    peluang lulus:", np.round(m.predict_proba(x[:, None])[:, 1], 4))
