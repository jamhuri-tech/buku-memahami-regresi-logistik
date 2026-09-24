"""Bab 11: memilih C, elastic net, dan skala peubah.

(1) LogisticRegressionCV dengan skor bawaan (akurasi) lawan
    scoring="neg_log_loss" pada data jarang.
(2) Dua peubah yang hampir sama (korelasi 0.99): L2, L1, dan elastic
    net membagi bobotnya dengan cara berbeda.
(3) Mengubah satuan satu peubah mengubah penaltinya; pembakuan
    menghapus ketergantungan itu.
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bab04_data import BENIH, dua_peubah
from bab11_data import data_jarang
from bab11_jalur import CS

warnings.simplefilter("ignore")

if __name__ == "__main__":
    X, y, Xu, yu = data_jarang()
    print("(1) LogisticRegressionCV, 5 lipatan, 21 nilai C:")
    print("    skor      pen  C        log-loss uji  akurasi uji")
    for skor in (None, "neg_log_loss"):
        for pen, solver in (("l2", "lbfgs"), ("l1", "saga")):
            m = LogisticRegressionCV(Cs=CS, cv=5, penalty=pen,
                                     solver=solver, scoring=skor,
                                     max_iter=20_000, tol=1e-6,
                                     random_state=0).fit(X, y)
            nama = "akurasi" if skor is None else "log-loss"
            ll = log_loss(yu, m.predict_proba(Xu))
            ak = accuracy_score(yu, m.predict(Xu))
            print(f"    {nama:9s} {pen}   {m.C_[0]:<8.3g} {ll:.3f}"
                  f"         {ak:.3f}")
    rng = np.random.default_rng(BENIH)
    n = 200
    z = rng.normal(size=n)
    X2 = np.column_stack([z + 0.1 * rng.normal(size=n),
                          z + 0.1 * rng.normal(size=n),
                          rng.normal(size=n)])
    y2 = (rng.random(n) < expit(1.5 * z)).astype(int)
    r = np.corrcoef(X2[:, 0], X2[:, 1])[0, 1]
    print(f"(2) korelasi x1, x2 = {r:.3f}; bobot sebenarnya lewat z")
    for pen, opsi in (("l2", {}), ("l1", {}),
                      ("elasticnet", {"l1_ratio": 0.5})):
        m = LogisticRegression(penalty=pen, C=0.05, solver="saga",
                               tol=1e-10, max_iter=100_000,
                               random_state=0, **opsi).fit(X2, y2)
        w = m.coef_[0] + 0.0
        print(f"    {pen:10s}: w = ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")
    X, y = dua_peubah()
    print("(3) penalti L2, C = 0.05, data dua peubah:")
    for nama, skala in (("x1 asli", 1.0), ("x1 x 100", 100.0)):
        Xs = X * np.array([skala, 1.0])
        m = LogisticRegression(C=0.05).fit(Xs, y)
        w = m.coef_[0] * np.array([skala, 1.0])
        p = make_pipeline(StandardScaler(), LogisticRegression(C=0.05))
        p.fit(Xs, y)
        prob = p.predict_proba(Xs[:3])[:, 1]
        print(f"    {nama:8s}: w1 (satuan asli) = {w[0]:.4f}, "
              f"w2 = {w[1]:.4f}")
        print(f"      dibakukan: p(3 titik pertama) = "
              f"{np.round(prob, 4)}")
