"""Bab 9: kalibrasi -- ECE, intersep dan kemiringan kalibrasi,
uji Hosmer-Lemeshow, dan Platt scaling.

Fungsi-fungsi di sini dipakai lagi di Bab 13 dan Bab 15.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from bab01_data import BENIH

W_BENAR = np.r_[1.0, -1.0, 0.5, np.zeros(17)]


def data_sintetis(m, benih):
    """m sampel, 20 fitur normal baku; hanya tiga yang berpengaruh,
    w0 = -1."""
    rng = np.random.default_rng(benih)
    X = rng.normal(size=(m, 20))
    y = (rng.random(m) < expit(-1.0 + X @ W_BENAR)).astype(int)
    return X, y


def ece(y, p, kelompok=10):
    """ECE dengan kelompok selebar sama di [0, 1]."""
    idx = np.minimum((p * kelompok).astype(int), kelompok - 1)
    total = 0.0
    for k in range(kelompok):
        s = idx == k
        if s.any():
            total += s.mean() * abs(p[s].mean() - y[s].mean())
    return total


def intersep_kemiringan(y, p):
    """(a, c): intersep kalibrasi (offset logit p) dan kemiringan."""
    s = logit(np.clip(p, 1e-15, 1 - 1e-15))
    a = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
               offset=s).fit().params[0]
    c = sm.Logit(y, sm.add_constant(s)).fit(disp=0).params[1]
    return a, c


def hosmer_lemeshow(y, p, G=10):
    """Statistik HL dengan G kelompok berukuran (hampir) sama."""
    urut = np.argsort(p, kind="stable")
    stat = 0.0
    for g in np.array_split(urut, G):
        O, E, ng = y[g].sum(), p[g].sum(), len(g)
        stat += (O - E) ** 2 / (E * (1 - E / ng))
    df = G - 2
    return stat, df, chi2.sf(stat, df) if df > 0 else np.nan


def platt(p_val, y_val):
    """Rekalibrasi: regresi logistik y pada logit p (data validasi)."""
    s = logit(np.clip(p_val, 1e-15, 1 - 1e-15))
    m = LogisticRegression(penalty=None, tol=1e-10).fit(s[:, None], y_val)
    return lambda p: m.predict_proba(
        logit(np.clip(p, 1e-15, 1 - 1e-15))[:, None])[:, 1]


if __name__ == "__main__":
    from bab08_performa import peluang_mini
    p, y = peluang_mini()
    print("(1) data mini (data latih):")
    print(f"    rata-rata p = {p.mean():.4f}, rata-rata y = {y.mean():.4f}")
    print(f"    ECE (2 kelompok) = {ece(y, p, 2):.4f},"
          f" ECE (3 kelompok) = {ece(y, p, 3):.4f}")
    a, c = intersep_kemiringan(y, p)
    print(f"    intersep = {a + 0:.4f}, kemiringan = {c:.4f}")
    hl, df, pv = hosmer_lemeshow(y, p, 3)
    print(f"    Hosmer-Lemeshow (G = 3) = {hl:.4f}, db = {df},"
          f" p = {pv:.4f}")
    hl, df, _ = hosmer_lemeshow(y, p, 2)
    print(f"    Hosmer-Lemeshow (G = 2) = {hl + 0:.4f}, db = {df}")

    Xl, yl = data_sintetis(200, BENIH)
    Xv, yv = data_sintetis(5000, BENIH + 1)
    Xu, yu = data_sintetis(20000, BENIH + 2)
    model = {
        "3 fitur benar, MLE": (LogisticRegression(
            penalty=None, solver="newton-cholesky", tol=1e-10),
            slice(0, 3)),
        "20 fitur, MLE": (LogisticRegression(
            penalty=None, solver="newton-cholesky", tol=1e-10),
            slice(0, 20)),
        "20 fitur, C = 0.01": (LogisticRegression(
            C=0.01, solver="newton-cholesky", tol=1e-10), slice(0, 20)),
    }
    print("(2) data sintetis, m latih 200, diukur pada 20 000 data uji:")
    print("    model                  ECE    intersep  kemiringan  HL p")
    hasil = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nama, (m, kol) in model.items():
            m.fit(Xl[:, kol], yl)
            pu = m.predict_proba(Xu[:, kol])[:, 1]
            hasil[nama] = (m, kol)
            a, c = intersep_kemiringan(yu, pu)
            _, _, pv = hosmer_lemeshow(yu, pu)
            teks = f"{pv:.1e}" if pv > 1e-10 else "< 1e-10"
            print(f"    {nama:<21} {ece(yu, pu):.4f}  {a:7.4f}"
                  f"   {c:7.4f}    {teks}")
    print("(3) Platt scaling, dilatih pada 5000 data validasi:")
    for nama in ["20 fitur, MLE", "20 fitur, C = 0.01"]:
        m, kol = hasil[nama]
        f = platt(m.predict_proba(Xv[:, kol])[:, 1], yv)
        pu = f(m.predict_proba(Xu[:, kol])[:, 1])
        a, c = intersep_kemiringan(yu, pu)
        print(f"    {nama:<21} ECE {ece(yu, pu):.4f},"
              f" kemiringan {c:.4f}")
