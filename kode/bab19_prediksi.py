"""Bab 19: model untuk prediksi, dengan scikit-learn.

Data dibagi berstrata menjadi 70 persen latih dan 30 persen uji. Tiga
model: tanpa penalti, L2, dan L1 (C dipilih validasi silang lima
lipatan berskor log-loss pada data latih). Peubah numerik dibakukan,
peubah kategori diberi one-hot encoding, semuanya di dalam pipeline.
"""
import warnings

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bab04_data import BENIH
from bab16_kalibrasi import ece
from bab19_data import BINER, KATEGORI, NUMERIK, baca_jantung

warnings.simplefilter("ignore")
CS = np.geomspace(1e-3, 1e2, 21)


def model(akhir):
    ubah = make_column_transformer(
        (StandardScaler(), NUMERIK + BINER),
        (OneHotEncoder(drop="first", sparse_output=False),
         list(KATEGORI)))
    return make_pipeline(ubah, akhir)


def bagi():
    d = baca_jantung()
    X, y = d.drop(columns="sakit"), d.sakit.to_numpy()
    return train_test_split(X, y, test_size=0.3, stratify=y,
                            random_state=BENIH % 2**31)


if __name__ == "__main__":
    Xl, Xu, yl, yu = bagi()
    print(f"latih {len(yl)} (sakit {yl.mean():.3f}), "
          f"uji {len(yu)} (sakit {yu.mean():.3f})")
    daftar = {
        "tanpa penalti": model(LogisticRegression(penalty=None,
                                                  tol=1e-10,
                                                  max_iter=10_000)),
        "L2, C dari CV": model(LogisticRegressionCV(
            Cs=CS, cv=5, scoring="neg_log_loss", max_iter=10_000)),
        "L1, C dari CV": model(LogisticRegressionCV(
            Cs=CS, cv=5, penalty="l1", solver="saga", tol=1e-6,
            scoring="neg_log_loss", max_iter=100_000, random_state=0)),
    }
    print("model           C       log-loss  Brier   AUC     ECE")
    q = np.full(len(yu), yl.mean())
    print(f"proporsi latih  -       {log_loss(yu, q):.4f}    "
          f"{brier_score_loss(yu, q):.4f}  0.5000  "
          f"{ece(yu, q, 5):.4f}")
    hasil = {}
    for nama, m in daftar.items():
        p = m.fit(Xl, yl).predict_proba(Xu)[:, 1]
        hasil[nama] = (m, p)
        C = getattr(m[-1], "C_", [np.nan])[0]
        teks_C = "-" if np.isnan(C) else f"{C:.3g}"
        print(f"{nama:15s} {teks_C:7s} {log_loss(yu, p):.4f}    "
              f"{brier_score_loss(yu, p):.4f}  {roc_auc_score(yu, p):.4f}"
              f"  {ece(yu, p, 5):.4f}")
    m, p = hasil["L1, C dari CV"]
    nol = int(np.sum(np.abs(m[-1].coef_) < 1e-10))
    print(f"L1: {nol} dari {m[-1].coef_.size} bobot tepat nol")
    m, p = hasil["L2, C dari CV"]
    for t in (0.5, 1 / 6):
        tn, fp, fn, tp = confusion_matrix(yu, p >= t).ravel()
        biaya = (5 * fn + fp) / len(yu)
        print(f"L2, ambang {t:.3f}: TP {tp:2d} FP {fp:2d} TN {tn:2d} "
              f"FN {fn:2d}, biaya {biaya:.3f}")
