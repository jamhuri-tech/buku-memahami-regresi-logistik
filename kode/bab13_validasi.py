"""Bab 13: kestabilan dan validasi -- validasi silang berulang,
bootstrap, koreksi optimisme, stabilitas seleksi peubah, uji DeLong,
dan ukuran sampel. Fungsi-fungsi di sini dipakai lagi di Bab 15.
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab09_kalibrasi import data_sintetis


def model_mle():
    return LogisticRegression(penalty=None, solver="newton-cholesky",
                              tol=1e-10, max_iter=1000)


def komponen_delong(y, s):
    """V10 (per positif) dan V01 (per negatif)."""
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(1), psi.mean(0)


def delong(y, s1, s2=None):
    """AUC, SE, dan (bila s2 diberikan) uji selisih AUC."""
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    a10, a01 = komponen_delong(y, s1)
    auc1 = a10.mean()
    if s2 is None:
        var = np.var(a10, ddof=1) / n1 + np.var(a01, ddof=1) / n0
        return auc1, np.sqrt(var)
    b10, b01 = komponen_delong(y, s2)
    S10 = np.cov(np.vstack([a10, b10]))
    S01 = np.cov(np.vstack([a01, b01]))
    S = S10 / n1 + S01 / n0
    d = auc1 - b10.mean()
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return auc1, b10.mean(), d, se, 2 * norm.sf(abs(d / se))


def n_riley(p, r2cs, S=0.9):
    """Ukuran sampel agar susut yang diharapkan paling kecil S."""
    return p / ((S - 1) * np.log(1 - r2cs / S))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    X, y = data_sintetis(300, BENIH + 10)
    Xu, yu = data_sintetis(20000, BENIH + 2)
    m = model_mle().fit(X, y)
    p = m.predict_proba(X)[:, 1]
    pu = m.predict_proba(Xu)[:, 1]
    print("(1) model 20 peubah, n = 300:")
    print(f"    tampak (data latih): AUC = {roc_auc_score(y, p):.4f},"
          f" log-loss = {log_loss(y, p):.4f}")
    auc, ll = [], []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                 random_state=0)
    for a, b in cv.split(X, y):
        mm = model_mle().fit(X[a], y[a])
        q = mm.predict_proba(X[b])[:, 1]
        auc.append(roc_auc_score(y[b], q))
        ll.append(log_loss(y[b], q))
    print(f"    CV 5 lipatan x 10: AUC = {np.mean(auc):.4f},"
          f" log-loss = {np.mean(ll):.4f}")
    print(f"      SD antar lipatan: AUC {np.std(auc, ddof=1):.4f},"
          f" log-loss {np.std(ll, ddof=1):.4f}")
    print(f"    data uji 20 000:   AUC = {roc_auc_score(yu, pu):.4f},"
          f" log-loss = {log_loss(yu, pu):.4f}")

    rng = np.random.default_rng(BENIH)
    B = 200
    opt, w1 = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        mb = model_mle().fit(X[i], y[i])
        opt.append(roc_auc_score(y[i], mb.predict_proba(X[i])[:, 1])
                   - roc_auc_score(y, mb.predict_proba(X)[:, 1]))
        w1.append(mb.coef_[0, 0])
    tampak = roc_auc_score(y, p)
    print(f"(2) koreksi optimisme bootstrap (B = {B}):")
    print(f"    AUC tampak {tampak:.4f} - optimisme {np.mean(opt):.4f}"
          f" = {tampak - np.mean(opt):.4f}")
    import statsmodels.api as sm
    r = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    q = np.percentile(w1, [2.5, 97.5])
    print("(3) koefisien x1:")
    print(f"    MLE {r.params[1]:.4f}, SE Wald {r.bse[1]:.4f},"
          f" SE bootstrap {np.std(w1, ddof=1):.4f}")
    print(f"    selang persentil bootstrap ({q[0]:.4f}, {q[1]:.4f})")

    pilih = np.zeros(20)
    for _ in range(100):
        i = rng.integers(0, len(y), len(y))
        ml = LogisticRegression(penalty="l1", C=0.05, solver="liblinear",
                                tol=1e-8).fit(X[i], y[i])
        pilih += ml.coef_[0] != 0
    print("(4) frekuensi terpilih L1 (C = 0.05) dalam 100 bootstrap:")
    print("    x1, x2, x3:", (pilih[:3] / 100).round(2))
    print(f"    17 peubah noise: terbesar {pilih[3:].max() / 100:.2f},"
          f" rata-rata {pilih[3:].mean() / 100:.2f}")

    m3 = model_mle().fit(X[:, :3], y)
    a, b, d, se, pv = delong(yu, m3.predict_proba(Xu[:, :3])[:, 1], pu)
    print("(5) DeLong pada data uji, 3 peubah lawan 20 peubah:")
    print(f"    AUC {a:.4f} lawan {b:.4f}, selisih {d:.4f}")
    teks = f"{pv:.1e}" if pv > 1e-10 else "< 1e-10"
    print(f"    SE selisih {se:.4f}, z = {d / se:.1f}, p {teks}")
    x0, y0 = data_mini()
    pm = 1 / (1 + np.exp(-(rancang(x0) @ mle_mini())))
    a0, s0 = delong(y0, pm)
    print(f"    data mini: AUC = {a0:.4f}, SE DeLong = {s0:.4f}")
    print(f"(6) n Riley (p = 10, R2_CS = 0.2, S = 0.9):"
          f" {n_riley(10, 0.2):.1f}")
