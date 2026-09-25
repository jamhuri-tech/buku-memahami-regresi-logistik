"""Bab 15: matriks kebingungan, ROC, AUC, dan precision-recall.

(1) Matriks kebingungan pada ambang 0.5 dan ukuran turunannya, dari
    nol dan dari sklearn.metrics.
(2) Kurva ROC dan AUC dari nol (Listing 15.1), AUC sebagai statistik
    Mann-Whitney, dan ketakpekaan AUC terhadap transformasi monoton.
(3) Average precision dan model tanpa usia.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_score, recall_score,
                             roc_auc_score)

from bab15_data import latih_pasien


def kurva_roc(y, s):
    urut = np.argsort(-s, kind="stable")
    y = y[urut]
    tp = np.r_[0, np.cumsum(y)]
    fp = np.r_[0, np.cumsum(1 - y)]
    return fp / fp[-1], tp / tp[-1]          # FPR, TPR


def auc_trapesium(fpr, tpr):
    return np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)


def auc_mann_whitney(y, s):
    r = rankdata(s)             # peringkat, seri dirata
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    _, y, p = latih_pasien()
    tebak = (p >= 0.5).astype(int)
    TP = np.sum((tebak == 1) & (y == 1))
    FP = np.sum((tebak == 1) & (y == 0))
    TN = np.sum((tebak == 0) & (y == 0))
    FN = np.sum((tebak == 0) & (y == 1))
    print(f"(1) ambang 0.5: TP {TP}, FP {FP}, TN {TN}, FN {FN}")
    tn, fp, fn, tp = confusion_matrix(y, tebak).ravel()
    print(f"    sklearn: TP {tp}, FP {fp}, TN {tn}, FN {fn}")
    presisi, recall = TP / (TP + FP), TP / (TP + FN)
    print(f"    akurasi {(TP + TN) / len(y):.4f} (sk {accuracy_score(y, tebak):.4f})")
    print(f"    precision {presisi:.4f} (sk {precision_score(y, tebak):.4f})")
    print(f"    recall {recall:.4f} (sk {recall_score(y, tebak):.4f})")
    print(f"    spesifisitas {TN / (TN + FP):.4f}")
    print(f"    F1 {2 * presisi * recall / (presisi + recall):.4f} "
          f"(sk {f1_score(y, tebak):.4f})")
    print(f"    selalu menebak 0: akurasi {np.mean(y == 0):.4f}")
    fpr, tpr = kurva_roc(y, p)
    print(f"(2) AUC trapesium {auc_trapesium(fpr, tpr):.6f}, "
          f"Mann-Whitney {auc_mann_whitney(y, p):.6f}")
    print(f"    sklearn roc_auc_score {roc_auc_score(y, p):.6f}")
    for nama, s in (("p^3", p ** 3), ("log-odds", np.log(p / (1 - p))),
                    ("p / 10", p / 10)):
        print(f"    AUC dari {nama:8s}: {roc_auc_score(y, s):.6f}")
    _, y2, p2 = latih_pasien(["imt", "perokok", "wilayah"])
    print(f"(3) average precision {average_precision_score(y, p):.4f}, "
          f"proporsi kelas 1 {y.mean():.4f}")
    print(f"    tanpa usia: AUC {roc_auc_score(y2, p2):.4f}, "
          f"AP {average_precision_score(y2, p2):.4f}")
