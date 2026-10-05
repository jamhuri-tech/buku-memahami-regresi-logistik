"""Bab 11: tafsiran koefisien -- odds ratio, skala, tabel 2 x 2,
interaksi, efek marginal rata-rata, dan ketidakruntuhan.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians


def dari_tabel(sel):
    """Data per sel (peubah..., y, frekuensi) -> X, y, bobot."""
    sel = np.asarray(sel, dtype=float)
    return sel[:, :-2], sel[:, -2], sel[:, -1]


def logit_berbobot(X, y, f):
    return sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(),
                  freq_weights=f).fit()


if __name__ == "__main__":
    x, y = data_mini()
    m = mle_mini()
    se = np.sqrt(np.diag(kovarians(m, rancang(x))))
    z = norm.ppf(0.975)
    print("(1) odds ratio jam belajar (data mini):")
    for nama, k in [("per 1 jam", 1.0), ("per 2 jam", 2.0),
                    ("per SD x", np.std(x, ddof=1))]:
        print(f"    {nama:<10}: OR = {np.exp(k * m[1]):8.4f},"
              f" selang Wald ({np.exp(k * (m[1] - z * se[1])):.4f},"
              f" {np.exp(k * (m[1] + z * se[1])):.2f})")

    print("(2) tabel 2 x 2 (merokok, sakit):")
    X, yy, f = dari_tabel([[1, 1, 30], [1, 0, 20], [0, 1, 15], [0, 0, 35]])
    r = logit_berbobot(X, yy, f)
    print(f"    ad/bc = {30 * 35 / (20 * 15):.4f},"
          f" logistik e^w = {np.exp(r.params[1]):.4f}")
    print(f"    SE rumus = {np.sqrt(1/30 + 1/20 + 1/15 + 1/35):.4f},"
          f" SE logistik = {r.bse[1]:.4f}")
    print(f"    e^b = odds tidak merokok = {np.exp(r.params[0]):.4f}"
          f" (= 15/35)")

    print("(3) interaksi dua peubah biner (A, B):")
    sel = [[0, 0, 0, 1, 10], [0, 0, 0, 0, 40], [1, 0, 0, 1, 20],
           [1, 0, 0, 0, 20], [0, 1, 0, 1, 20], [0, 1, 0, 0, 20],
           [1, 1, 1, 1, 45], [1, 1, 1, 0, 5]]
    X, yy, f = dari_tabel(sel)
    r = logit_berbobot(X, yy, f)
    b0, bA, bB, bAB = r.params
    print(f"    b = {b0:.4f}, wA = {bA:.4f}, wB = {bB:.4f},"
          f" wAB = {bAB:.4f}")
    print(f"    OR A bila B=0: {np.exp(bA):.4f},"
          f" bila B=1: {np.exp(bA + bAB):.4f},"
          f" rasio: {np.exp(bAB):.4f}")

    p = expit(rancang(x) @ m)
    ame = np.mean(m[1] * p * (1 - p))
    print("(4) efek marginal jam belajar (data mini):")
    print(f"    AME = {ame:.4f}, efek di x = 3.5: {m[1] / 4:.4f}")
    rr = sm.Logit(y, rancang(x)).fit(disp=0)
    print(f"    statsmodels get_margeff: {rr.get_margeff().margeff[0]:.4f}")

    print("(5) ketidakruntuhan (dua strata sama besar):")
    sel = [[0, 0, 1, 20], [0, 0, 0, 80], [1, 0, 1, 50], [1, 0, 0, 50],
           [0, 1, 1, 50], [0, 1, 0, 50], [1, 1, 1, 80], [1, 1, 0, 20]]
    X, yy, f = dari_tabel(sel)
    rk = logit_berbobot(X, yy, f)
    rm = logit_berbobot(X[:, :1], yy, f)
    print(f"    OR x bersyarat strata = {np.exp(rk.params[1]):.4f}")
    print(f"    OR x marginal         = {np.exp(rm.params[1]):.4f}")
