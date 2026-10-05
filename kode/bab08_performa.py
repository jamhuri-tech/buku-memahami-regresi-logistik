"""Bab 8: ukuran performa, dihitung sendiri dan dicocokkan dengan
scikit-learn. Fungsi-fungsi di sini dipakai lagi di bab berikutnya.
"""
import numpy as np
from scipy.special import expit
from sklearn import metrics

from bab01_data import data_mini, mle_mini, rancang


def kebingungan(y, p, t=0.5):
    """(TP, FP, TN, FN) untuk tebakan p >= t."""
    yh = p >= t
    return (int(np.sum(yh & (y == 1))), int(np.sum(yh & (y == 0))),
            int(np.sum(~yh & (y == 0))), int(np.sum(~yh & (y == 1))))


def ukuran(tp, fp, tn, fn):
    akurasi = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    spesifisitas = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return akurasi, precision, recall, spesifisitas, f1


def auc_mw(y, s):
    """AUC = peluang skor positif > skor negatif (seri dihitung 1/2)."""
    pos, neg = s[y == 1], s[y == 0]
    beda = pos[:, None] - neg[None, :]
    return (np.sum(beda > 0) + 0.5 * np.sum(beda == 0)) / beda.size


def average_precision(y, s):
    urut = np.argsort(-s, kind="stable")
    yy = y[urut]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return np.sum(prec * yy) / yy.sum()


def brier(y, p):
    return np.mean((p - y) ** 2)


def pseudo_r2(y, p):
    """McFadden, Cox-Snell, Nagelkerke, dan Tjur."""
    n = len(y)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    yb = y.mean()
    ll0 = n * (yb * np.log(yb) + (1 - yb) * np.log(1 - yb))
    mcf = 1 - ll / ll0
    cs = 1 - np.exp(2 * (ll0 - ll) / n)
    nag = cs / (1 - np.exp(2 * ll0 / n))
    tjur = p[y == 1].mean() - p[y == 0].mean()
    return mcf, cs, nag, tjur


if __name__ == "__main__":
    x, y = data_mini()
    p = expit(rancang(x) @ mle_mini())
    print("(1) matriks kebingungan dan ukuran ambang:")
    print("     t    TP FP TN FN   akurasi  prec   recall  spes   F1")
    for t in [0.5, 0.3]:
        k = kebingungan(y, p, t)
        u = ukuran(*k)
        print(f"    {t:.1f}   {k[0]}  {k[1]}  {k[2]}  {k[3]}    "
              + "  ".join(f"{v:.4f}" for v in u))
    yh = (p >= 0.5).astype(int)
    print("    sklearn sama:",
          np.isclose(metrics.accuracy_score(y, yh), 4 / 6),
          np.isclose(metrics.f1_score(y, yh), 2 / 3))

    print("(2) ROC dan AUC:")
    fpr, tpr, ambang = metrics.roc_curve(y, p)
    for a, b, c in zip(fpr, tpr, ambang):
        if np.isfinite(c):
            print(f"    t = {c:.4f}: FPR = {a:.4f}, TPR = {b:.4f}")
    print(f"    AUC Mann-Whitney = {auc_mw(y, p):.4f},"
          f" sklearn = {metrics.roc_auc_score(y, p):.4f}")
    print(f"    AP sendiri = {average_precision(y, p):.4f},"
          f" sklearn = {metrics.average_precision_score(y, p):.4f}")

    print("(3) ukuran peluang:")
    print(f"    log-loss = {metrics.log_loss(y, p):.4f},"
          f" Brier = {brier(y, p):.4f},"
          f" sklearn = {metrics.brier_score_loss(y, p):.4f}")
    print(f"    Brier model nol = {brier(y, np.full(6, 0.5)):.4f},"
          f" skill = {1 - brier(y, p) / 0.25:.4f}")
    mcf, cs, nag, tjur = pseudo_r2(y, p)
    print(f"    McFadden = {mcf:.4f}, Cox-Snell = {cs:.4f}")
    print(f"    Nagelkerke = {nag:.4f}, Tjur = {tjur:.4f}")
