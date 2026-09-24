"""Bab 4: peluang, odds, dan log-odds (logit).

Mencetak tabel peluang -> odds -> log-odds, lalu memeriksa bahwa
logit dan fungsi logistik saling invers.
"""
import numpy as np
from scipy.special import logit

from bab04_sigmoid import sigmoid

if __name__ == "__main__":
    print("   p     odds   log-odds")
    for p in (0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99):
        odds = p / (1 - p)
        print(f"{p:5.2f} {odds:8.3f} {np.log(odds):+9.3f}")
    p = np.linspace(0.001, 0.999, 999)
    z = np.linspace(-30, 30, 601)
    print(f"maks |sigma(logit(p)) - p| = "
          f"{np.max(np.abs(sigmoid(logit(p)) - p)):.1e}")
    print(f"maks |logit(sigma(z)) - z| (|z| <= 30) = "
          f"{np.max(np.abs(logit(sigmoid(z)) - z)):.1e}")
    print(f"logit(1 - p) + logit(p) = "
          f"{np.max(np.abs(logit(1 - p) + logit(p))):.1e}")
