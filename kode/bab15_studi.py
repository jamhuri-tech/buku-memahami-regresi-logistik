"""Bab 15: studi kasus riset utuh pada data Heart Disease Cleveland.

Protokol (ditetapkan sebelum melihat hasil): 17 parameter peubah
(Bab 15, tabel protokol); model tafsiran MLE dengan selang profil;
model prediksi MLE dan L2 (C dipilih dengan CV di dalam lipatan);
validasi dengan CV 5 lipatan x 10 dan koreksi optimisme bootstrap.
"""
import warnings

import numpy as np
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bab01_data import BENIH
from bab07_prediksi import kovarians
from bab09_kalibrasi import intersep_kemiringan
from bab10_inferensi import log_kem, mle, selang_profil
from bab12_diagnostik import cook, leverage, vif
from bab13_validasi import delong, n_riley
from bab15_data import baca_jantung, rancangan

warnings.simplefilter("ignore")


def model_mle():
    return make_pipeline(StandardScaler(), LogisticRegression(
        penalty=None, solver="newton-cholesky", tol=1e-10))


def model_l2():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 21), cv=5, scoring="neg_log_loss",
        solver="newton-cholesky", tol=1e-10))


if __name__ == "__main__":
    d = baca_jantung()
    X, y, nama = rancangan(d)
    n, k = X.shape[0], X.shape[1] - 1
    print("(1) data dan ukuran sampel:")
    print(f"    n = {n}, sakit = {y.sum()} ({y.mean():.3f}),"
          f" parameter peubah = {k}")
    print(f"    EPV = {min(y.sum(), n - y.sum()) / k:.2f}")

    th = mle(X, y)
    se = np.sqrt(np.diag(kovarians(th, X)))
    print("(2) model tafsiran: OR, selang Wald, selang profil")
    print("    peubah              OR    Wald            profil")
    for j in range(1, k + 1):
        a, b = selang_profil(X, y, j)
        print(f"    {nama[j]:<18} {np.exp(th[j]):5.2f}"
              f"  ({np.exp(th[j] - 1.96 * se[j]):5.2f},"
              f"{np.exp(th[j] + 1.96 * se[j]):6.2f})"
              f"  ({np.exp(a):5.2f},{np.exp(b):6.2f})")
    ll = log_kem(th, X, y)
    print("    uji rasio kemungkinan per kelompok:")
    for g, idx in [("nyeri", [3, 4, 5]), ("lereng", [13, 14]),
                   ("thal", [16, 17])]:
        sisa = [j for j in range(k + 1) if j not in idx]
        G = 2 * (ll - log_kem(mle(X[:, sisa], y), X[:, sisa], y))
        print(f"      {g:<7} G = {G:6.2f}, db {len(idx)},"
              f" p = {chi2.sf(G, len(idx)):.4f}")

    print("(3) diagnostik:")
    print(f"    VIF terbesar = {vif(X[:, 1:]).max():.2f}")
    for j, nm in [(1, "usia"), (6, "tensi"), (7, "kolesterol"),
                  (10, "nadi_maks"), (12, "depresi_st")]:
        X2 = np.c_[X, X[:, j] ** 2]
        G = 2 * (log_kem(mle(X2, y), X2, y) - ll)
        print(f"    suku kuadrat {nm:<11}: G = {G:5.2f},"
              f" p = {chi2.sf(G, 1):.4f}")
    p = expit(X @ th)
    h, _ = leverage(X, p)
    D = cook(y, p, h, k + 1)
    i = int(np.argmax(D))
    t2 = mle(np.delete(X, i, 0), np.delete(y, i))
    print(f"    Cook terbesar = {D[i]:.4f} (pasien {i});"
          f" tanpa pasien itu,")
    print(f"    perubahan |koefisien| terbesar = {np.abs(t2 - th).max():.4f}"
          f" ({nama[int(np.argmax(np.abs(t2 - th)))]})")

    print("(4) model prediksi, CV 5 lipatan x 10:")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    hasil = {"MLE": [], "L2": []}
    oof = {"MLE": np.zeros((10, n)), "L2": np.zeros((10, n))}
    Z = X[:, 1:]
    for f, (a, b) in enumerate(cv.split(Z, y)):
        for nm, mk in [("MLE", model_mle), ("L2", model_l2)]:
            m = mk().fit(Z[a], y[a])
            q = m.predict_proba(Z[b])[:, 1]
            oof[nm][f // 5, b] = q
            hasil[nm].append((roc_auc_score(y[b], q), log_loss(y[b], q),
                              brier_score_loss(y[b], q)))
    print("    model  AUC (SD)         log-loss  Brier   kemiringan")
    for nm in hasil:
        r = np.array(hasil[nm])
        km = np.mean([intersep_kemiringan(y, o)[1] for o in oof[nm]])
        print(f"    {nm:<5}  {r[:, 0].mean():.4f} ({r[:, 0].std(ddof=1):.4f})"
              f"  {r[:, 1].mean():.4f}    {r[:, 2].mean():.4f}  {km:.4f}")
    a10 = delong(y, oof["L2"][0], oof["MLE"][0])
    print("    DeLong L2 lawan MLE (pengulangan pertama):")
    print(f"      selisih AUC {a10[2]:.4f}, p = {a10[4]:.4f}")

    rng = np.random.default_rng(BENIH)
    m = model_mle().fit(Z, y)
    tampak = roc_auc_score(y, m.predict_proba(Z)[:, 1])
    opt, kem = [], []
    for _ in range(200):
        i = rng.integers(0, n, n)
        mb = model_mle().fit(Z[i], y[i])
        opt.append(roc_auc_score(y[i], mb.predict_proba(Z[i])[:, 1])
                   - roc_auc_score(y, mb.predict_proba(Z)[:, 1]))
        kem.append(intersep_kemiringan(y, mb.predict_proba(Z)[:, 1])[1])
    print("(5) koreksi optimisme bootstrap MLE (B = 200):")
    print(f"    AUC tampak {tampak:.4f}, optimisme {np.mean(opt):.4f},"
          f" terkoreksi {tampak - np.mean(opt):.4f}")
    print(f"    kemiringan kalibrasi (susut) = {np.mean(kem):.4f}")
    r2cs = 1 - np.exp(2 * (log_kem(np.r_[logit(y.mean()), np.zeros(k)],
                                   X, y) - ll) / n)
    print(f"    R2 Cox-Snell = {r2cs:.4f},"
          f" n Riley = {n_riley(k, r2cs):.0f}")
