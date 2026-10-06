"""Bab 11: tafsiran bobot -- odds ratio, skala, penyesuaian,
tabel 2 x 2, interaksi, efek marginal rata-rata, dan ketidakruntuhan.
"""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

from bab01_data import data_mini, mle_mini, rancang
from bab07_prediksi import kovarians
from bab10_inferensi import mle


def dari_tabel(sel):
    """Data per sel (fitur..., y, frekuensi) -> X, y, frekuensi."""
    sel = np.asarray(sel, dtype=float)
    return sel[:, :-2], sel[:, -2], sel[:, -1]


def logit_berbobot(X, y, f):
    return sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(),
                  freq_weights=f).fit()


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    se = np.sqrt(np.diag(kovarians(m, X)))
    z = norm.ppf(0.975)
    sd = np.std(Xf, axis=0, ddof=1)
    print("(1) odds ratio (data mini, fitur lain tetap):")
    for nama, j, k in [("x1 per 1 jam", 1, 1.0),
                       ("x1 per 2 jam", 1, 2.0),
                       ("x1 per SD", 1, sd[0]),
                       ("x2 per 1 absen", 2, 1.0),
                       ("x2 per SD", 2, sd[1])]:
        lo, hi = k * (m[j] - z * se[j]), k * (m[j] + z * se[j])
        print(f"    {nama:<15}: OR = {np.exp(k * m[j]):7.4f},"
              f" selang ({np.exp(lo):.4f}, {np.exp(hi):.2f})")
    print(f"    SD x1 = {sd[0]:.4f}, SD x2 = {sd[1]:.4f}")

    print("(2) penyesuaian: OR jam belajar tanpa dan dengan absen")
    w1 = mle(X[:, :2], y)
    print(f"    model x1 saja : w = {np.round(w1, 4)},"
          f" OR = {np.exp(w1[1]):.4f}")
    print(f"    model x1 + x2 : w = {np.round(m, 4)},"
          f" OR = {np.exp(m[1]):.4f}")
    print(f"    korelasi x1, x2 = {np.corrcoef(Xf.T)[0, 1]:.4f}")

    print("(3) tabel 2 x 2 (merokok, sakit):")
    T, yy, f = dari_tabel([[1, 1, 30], [1, 0, 20], [0, 1, 15],
                           [0, 0, 35]])
    r = logit_berbobot(T, yy, f)
    print(f"    ad/bc = {30 * 35 / (20 * 15):.4f},"
          f" logistik e^w1 = {np.exp(r.params[1]):.4f}")
    print(f"    SE rumus = {np.sqrt(1/30 + 1/20 + 1/15 + 1/35):.4f},"
          f" SE logistik = {r.bse[1]:.4f}")
    print(f"    e^w0 = odds tidak merokok = {np.exp(r.params[0]):.4f}"
          f" (= 15/35)")

    print("(4) interaksi dua fitur biner (x1, x2):")
    sel = [[0, 0, 0, 1, 10], [0, 0, 0, 0, 40], [1, 0, 0, 1, 20],
           [1, 0, 0, 0, 20], [0, 1, 0, 1, 20], [0, 1, 0, 0, 20],
           [1, 1, 1, 1, 45], [1, 1, 1, 0, 5]]
    T, yy, f = dari_tabel(sel)
    r = logit_berbobot(T, yy, f)
    print("    w = (" + ", ".join(f"{v:.4f}" for v in r.params) + ")")
    w = r.params
    print(f"    OR x1 bila x2=0: {np.exp(w[1]):.4f},"
          f" bila x2=1: {np.exp(w[1] + w[3]):.4f},"
          f" rasio: {np.exp(w[3]):.4f}")

    p = expit(X @ m)
    d = p * (1 - p)
    print("(5) efek marginal (data mini):")
    print(f"    jumlah d_i = {d.sum():.4f}, rata-rata = {d.mean():.4f}")
    print(f"    AME x1 = {m[1] * d.mean():.4f},"
          f" AME x2 = {m[2] * d.mean():.4f},"
          f" di p = 1/2: {m[1] / 4:.4f}")
    rr = sm.Logit(y, X).fit(disp=0)
    print("    statsmodels get_margeff:",
          np.round(rr.get_margeff().margeff, 4))

    print("(6) ketidakruntuhan (dua strata sama besar):")
    sel = [[0, 0, 1, 20], [0, 0, 0, 80], [1, 0, 1, 50], [1, 0, 0, 50],
           [0, 1, 1, 50], [0, 1, 0, 50], [1, 1, 1, 80], [1, 1, 0, 20]]
    T, yy, f = dari_tabel(sel)
    rk = logit_berbobot(T, yy, f)
    rm = logit_berbobot(T[:, :1], yy, f)
    print(f"    OR x1 bersyarat strata = {np.exp(rk.params[1]):.4f}")
    print(f"    OR x1 marginal         = {np.exp(rm.params[1]):.4f}")
