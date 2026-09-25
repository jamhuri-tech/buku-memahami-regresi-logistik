"""Bab 18: one-vs-rest lawan softmax pada data tiga kelas.

Model dilatih pada 600 titik dan dinilai pada 20 ribu titik uji dari
model yang sama. Untuk one-vs-rest dicatat pula jumlah tiga peluang
biner sebelum dinormalkan.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.multiclass import OneVsRestClassifier

from bab18_data import tiga_kelas

warnings.simplefilter("ignore")

if __name__ == "__main__":
    X, y, _ = tiga_kelas()
    Xu, yu, Pu = tiga_kelas(20_000, benih=1)
    lr = dict(penalty=None, tol=1e-10, max_iter=10_000)
    soft = LogisticRegression(**lr).fit(X, y)
    ovr = OneVsRestClassifier(LogisticRegression(**lr)).fit(X, y)
    print("data uji (20000 titik)   log-loss  akurasi")
    for nama, p in (("peluang sebenarnya", Pu),
                    ("softmax", soft.predict_proba(Xu)),
                    ("one-vs-rest", ovr.predict_proba(Xu))):
        print(f"    {nama:18s}   {log_loss(yu, p):.4f}    "
              f"{accuracy_score(yu, p.argmax(axis=1)):.4f}")
    mentah = np.column_stack([e.predict_proba(Xu)[:, 1]
                              for e in ovr.estimators_])
    s = mentah.sum(axis=1)
    print(f"one-vs-rest, jumlah tiga peluang biner sebelum dinormalkan:")
    print(f"    terkecil {s.min():.3f}, terbesar {s.max():.3f}, "
          f"rata-rata {s.mean():.3f}")
    sama = np.mean(soft.predict(Xu) == ovr.predict(Xu))
    print(f"tebakan softmax dan one-vs-rest sama pada {sama:.4f} data uji")
