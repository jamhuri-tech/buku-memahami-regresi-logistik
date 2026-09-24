"""Bab 13: pengacau dan multikolinearitas.

(1) Pengacau: z memengaruhi paparan x dan label y. Tanpa z, koefisien
    x berganti tanda (paradoks Simpson).
(2) Multikolinearitas: dua peubah berkorelasi 0.95 lawan 0; galat baku
    membesar kira-kira sqrt(VIF), peluang hampir tidak berubah.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from bab04_data import BENIH

if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    n = 20_000
    z = rng.normal(size=n)                          # misalnya usia
    x = (rng.random(n) < expit(2 * z)).astype(int)  # paparan
    y = (rng.random(n) < expit(-1 - 1.0 * x + 2 * z)).astype(int)
    tanpa = sm.Logit(y, sm.add_constant(x)).fit(disp=0).params[1]
    X = sm.add_constant(np.column_stack([x, z]))
    dengan = sm.Logit(y, X).fit(disp=0).params[1]
    print("(1) pengacau, log-OR paparan sebenarnya -1.0:")
    print(f"    tanpa z: {tanpa:+.4f}, dengan z: {dengan:+.4f}")
    n = 500
    print("(2) multikolinearitas, n = 500, w = (1, 1):")
    for rho in (0.0, 0.95):
        a = rng.normal(size=n)
        b = rho * a + np.sqrt(1 - rho**2) * rng.normal(size=n)
        X = sm.add_constant(np.column_stack([a, b]))
        y = (rng.random(n) < expit(X @ np.array([0.0, 1.0, 1.0])))
        h = sm.Logit(y.astype(int), X).fit(disp=0)
        vif = 1 / (1 - np.corrcoef(a, b)[0, 1] ** 2)
        print(f"    rho = {rho:4.2f}, VIF {vif:4.1f}: "
              f"w1 = {h.params[1]:.3f} (SE {h.bse[1]:.3f})")
        print(f"                         w2 = {h.params[2]:.3f} "
              f"(SE {h.bse[2]:.3f})")
