"""Bab 15: memilih ambang menurut biaya, dan aturan skor proper.

(1) Biaya salah: tebakan negatif yang salah (FN) berbiaya 5, positif
    yang salah (FP) berbiaya 1. Ambang teoretis c_FP / (c_FP + c_FN)
    dibandingkan dengan ambang terbaik pada data uji.
(2) Ambang lain yang lazim: F1 terbesar dan indeks Youden.
(3) Log-loss dan Brier score untuk model lengkap dan tanpa usia.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from bab15_data import latih_pasien

C_FN, C_FP = 5.0, 1.0


def biaya(y, p, t):
    tebak = p >= t
    return (C_FN * np.sum(~tebak & (y == 1))
            + C_FP * np.sum(tebak & (y == 0))) / len(y)


if __name__ == "__main__":
    _, y, p = latih_pasien()
    ambang = np.round(np.arange(0.01, 1.0, 0.01), 2)
    b = np.array([biaya(y, p, t) for t in ambang])
    t_teori = C_FP / (C_FP + C_FN)
    print(f"(1) biaya FN = {C_FN:g}, FP = {C_FP:g}; ambang teori "
          f"{t_teori:.4f}")
    print(f"    biaya per pasien: ambang 0.5 {biaya(y, p, 0.5):.4f}, "
          f"ambang teori {biaya(y, p, t_teori):.4f}")
    k = int(np.argmin(b))
    print(f"    ambang terbaik pada data uji {ambang[k]:.2f}, "
          f"biaya {b[k]:.4f}")
    print(f"    selalu negatif {biaya(y, p, 1.01):.4f}, "
          f"selalu positif {biaya(y, p, 0.0):.4f}")
    tp = np.array([np.sum((p >= t) & (y == 1)) for t in ambang])
    fp = np.array([np.sum((p >= t) & (y == 0)) for t in ambang])
    fn = y.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    youden = tp / y.sum() - fp / (len(y) - y.sum())
    print(f"(2) ambang F1 terbesar {ambang[np.argmax(f1)]:.2f} "
          f"(F1 {f1.max():.4f}); Youden {ambang[np.argmax(youden)]:.2f}")
    _, y2, p2 = latih_pasien(["imt", "perokok", "wilayah"])
    brier = np.mean((p - y) ** 2)
    print(f"(3) model lengkap : log-loss {log_loss(y, p):.4f}, Brier "
          f"{brier:.4f} (sk {brier_score_loss(y, p):.4f})")
    print(f"    tanpa usia    : log-loss {log_loss(y2, p2):.4f}, Brier "
          f"{brier_score_loss(y2, p2):.4f}")
    q = np.full_like(p, y.mean())
    print(f"    proporsi uji  : log-loss {log_loss(y, q):.4f}, Brier "
          f"{brier_score_loss(y, q):.4f}")
