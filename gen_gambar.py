# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261005  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().





# ============================ Bab 1 ==================================

def bab01_sigmoid():
    from scipy.special import expit
    from bab01_data import data_mini
    x, y = data_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.1))
    z = np.linspace(-6, 6, 400)
    a.plot(z, expit(z), color=BIRU, lw=1.1, label=r"$\sigma(z)$")
    a.plot(z, expit(z) * (1 - expit(z)), color=JINGGA, lw=1.0,
           label=r"$\sigma'(z)$")
    a.plot([-2, 2], [0, 1], color=HIJAU, lw=0.7, ls="--",
           label="garis singgung di 0")
    a.axhline(0.5, color=ABU_GARIS, lw=0.5)
    a.set_ylim(-0.05, 1.05)
    a.set_xlabel("$z$")
    a.legend(loc="center right", fontsize=6)
    a.set_title("fungsi logistik dan turunannya")
    g = np.linspace(0, 7, 400)
    b.scatter(x, y, s=12, color=np.where(y == 1, BIRU, JINGGA), zorder=3)
    b.plot(g, expit(-2.8 + 0.8 * g), color=ABU, lw=0.9, ls="--",
           label=r"$\theta = (-2{,}8;\ 0{,}8)$")
    bb, ww = -4.2491, 1.2140
    b.plot(g, expit(bb + ww * g), color=BIRU, lw=1.1, label="MLE")
    b.plot([3.5], [0.5], "o", ms=3, color=MERAH, zorder=4)
    b.annotate("$-b/w = 3{,}5$", (3.5, 0.5), (4.4, 0.25), fontsize=6,
               color=MERAH, arrowprops=dict(arrowstyle="-", lw=0.4,
                                            color=MERAH))
    b.set_xlabel("jam belajar $x$")
    b.set_ylabel("peluang lulus")
    b.legend(loc="upper left", fontsize=6)
    b.set_title("data mini")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-sigmoid")


# ============================ Bab 2 ==================================

def bab02_loss():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    from bab02_loss import log_loss
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    z = np.linspace(-5, 5, 400)
    a.plot(z, np.logaddexp(0, -z), color=BIRU, lw=1.1,
           label=r"log-loss $\log(1 + e^{-z})$")
    a.plot(z, (1 - expit(z)) ** 2, color=JINGGA, lw=1.0,
           label=r"kuadrat $(1 - \sigma(z))^2$")
    a.set_xlabel("skor $z$ (label $y = 1$)")
    a.set_ylabel("loss satu titik")
    a.set_ylim(0, 4)
    a.legend(loc="upper right", fontsize=6)
    a.set_title("loss satu titik")
    x, y = data_mini()
    X = rancang(x)
    bb = np.linspace(-9, 1, 241)
    ww = np.linspace(-0.5, 2.6, 241)
    B, W = np.meshgrid(bb, ww)
    T = np.stack([B.ravel(), W.ravel()], 1)
    Z = np.logaddexp(0, T @ X.T) - (T @ X.T) * y
    Lg = Z.mean(1).reshape(B.shape)
    cs = b.contour(B, W, Lg, levels=[0.42, 0.45, 0.5, 0.6, 0.7, 0.9,
                                     1.2, 1.6], colors=BIRU,
                   linewidths=0.6)
    b.clabel(cs, fontsize=5, fmt=lambda v: f"{v:g}".replace(".", ","))
    m = mle_mini()
    b.plot(0, 0, "s", ms=3, color=ABU)
    b.plot(-2.8, 0.8, "^", ms=3.5, color=JINGGA)
    b.plot(*m, "o", ms=3.5, color=MERAH)
    b.annotate("MLE", m, (m[0] + 1.2, m[1] + 0.6), fontsize=6,
               color=MERAH, arrowprops=dict(arrowstyle="-", lw=0.4,
                                            color=MERAH))
    b.set_xlabel("$b$")
    b.set_ylabel("$w$")
    b.set_title("$L(b, w)$ pada data mini")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-loss")


# ============================ Bab 3 ==================================

def bab03_gradien():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    from bab03_turunan import gradien
    x, y = data_mini()
    X = rancang(x)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    bb = np.linspace(-9, 1, 241)
    ww = np.linspace(-0.5, 2.6, 241)
    B, W = np.meshgrid(bb, ww)
    T = np.stack([B.ravel(), W.ravel()], 1)
    Z = T @ X.T
    Lg = (np.logaddexp(0, Z) - Z * y).mean(1).reshape(B.shape)
    a.contour(B, W, Lg, levels=[0.42, 0.45, 0.5, 0.6, 0.7, 0.9, 1.2,
                                1.6], colors=ABU_GARIS, linewidths=0.6)
    for tb in np.linspace(-8, 0, 5):
        for tw in np.linspace(0, 2.4, 5):
            g = gradien(np.array([tb, tw]), X, y)
            a.arrow(tb, tw, -g[0] * 0.5, -g[1] * 0.5, color=BIRU, lw=0.5,
                    head_width=0.1, length_includes_head=True)
    m = mle_mini()
    a.plot(*m, "o", ms=3.5, color=MERAH, zorder=4)
    a.set_xlabel("$b$")
    a.set_ylabel("$w$")
    a.set_title(r"arah $-\nabla L$ (panjang $\times 0{,}5$)")
    a.set_ylim(-0.6, 2.7)
    for th, nama, gaya, wr in [
            (np.zeros(2), r"$\theta = 0$", "s", ABU),
            (np.array([-2.8, 0.8]), r"$(-2{,}8;\ 0{,}8)$", "^", JINGGA),
            (m, "MLE", "o", MERAH)]:
        p = expit(X @ th)
        b.plot(x, p * (1 - p), gaya + "-", ms=3, lw=0.8, label=nama,
               color=wr)
    b.set_xlabel("jam belajar $x$")
    b.set_ylabel("$d_i = p_i(1 - p_i)$")
    b.set_ylim(0, 0.27)
    b.legend(loc="lower center", fontsize=6)
    b.set_title("bobot Hessian per titik")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-gradien")


# ============================ Bab 4 ==================================

def bab04_gd():
    from bab01_data import data_mini, mle_mini, rancang
    from bab02_loss import log_loss
    from bab04_gd import gd, sgd
    x, y = data_mini()
    X = rancang(x)
    Xc = rancang(x - 3.5)
    m = mle_mini()
    Ls = log_loss(m, X, y)
    L = np.linalg.eigvalsh(X.T @ X / 24).max()
    Lc = 17.5 / 24
    jm = gd(X, y, 1 / L, 4000)
    jc = gd(Xc, y, 1 / Lc, 4000)
    js = sgd(Xc, y, 0.5, 4000)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    bb = np.linspace(-6, 1, 241)
    ww = np.linspace(-0.3, 1.9, 241)
    B, W = np.meshgrid(bb, ww)
    T = np.stack([B.ravel(), W.ravel()], 1)
    Z = T @ X.T
    Lg = (np.logaddexp(0, Z) - Z * y).mean(1).reshape(B.shape)
    a.contour(B, W, Lg, levels=[0.415, 0.43, 0.46, 0.5, 0.6, 0.7, 0.9],
              colors=ABU_GARIS, linewidths=0.6)
    a.plot(jm[:, 0], jm[:, 1], "-", color=BIRU, lw=0.9,
           label="x mentah")
    a.plot(jm[[0, 10, 100, 1000], 0], jm[[0, 10, 100, 1000], 1], "o",
           ms=2.5, color=BIRU)
    # jalur x dipusat, dipetakan ke (b, w) = (b_c - 3,5 w, w)
    a.plot(jc[:60, 0] - 3.5 * jc[:60, 1], jc[:60, 1], "-", color=JINGGA,
           lw=0.9, label="x dipusat")
    a.plot(*m, "*", ms=6, color=MERAH, zorder=4)
    a.set_xlabel("$b$")
    a.set_ylabel("$w$")
    a.legend(loc="upper right", fontsize=6)
    a.set_title(r"jalur GD, $\eta = 1/L$")
    k = np.arange(4001)
    for jj, A, nama, wr in [(jm, X, "GD, x mentah", BIRU),
                            (jc, Xc, "GD, x dipusat", JINGGA)]:
        sel = np.array([log_loss(t, A, y) for t in jj]) - Ls
        b.semilogy(k, np.maximum(sel, 1e-16), color=wr, lw=1.0,
                   label=nama)
    e = np.arange(len(js)) * 6
    sel = np.array([log_loss(t, Xc, y) for t in js]) - Ls
    b.semilogy(e, np.maximum(sel, 1e-16), color=HIJAU, lw=1.0,
               label="SGD (per epoch)")
    b.set_xlim(0, 4000)
    b.set_ylim(1e-14, 1)
    b.set_xlabel("langkah (SGD: pembaruan)")
    b.set_ylabel(r"$L(\theta_k) - L^*$")
    b.legend(loc="upper right", fontsize=6)
    b.set_title("kecepatan turun")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-gd")


# ============================ Bab 5 ==================================

def bab05_newton():
    from bab01_data import data_mini, mle_mini, rancang
    from bab02_loss import log_loss
    from bab04_gd import gd
    from bab05_newton import newton
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    bb = np.linspace(-8, 1, 241)
    ww = np.linspace(-0.3, 3.3, 241)
    B, W = np.meshgrid(bb, ww)
    T = np.stack([B.ravel(), W.ravel()], 1)
    Z = T @ X.T
    Lg = (np.logaddexp(0, Z) - Z * y).mean(1).reshape(B.shape)
    a.contour(B, W, Lg, levels=[0.415, 0.43, 0.46, 0.5, 0.6, 0.8, 1.1,
                                1.5], colors=ABU_GARIS, linewidths=0.6)
    jn, _ = newton(X, y, langkah=8)
    a.plot(jn[:, 0], jn[:, 1], "o-", ms=2.5, lw=0.9, color=BIRU,
           label=r"Newton dari $(0, 0)$")
    jd, _ = newton(X, y, np.array([-6.0, 3.0]), langkah=30, redam=True)
    a.plot(jd[:, 0], jd[:, 1], "s-", ms=2.5, lw=0.9, color=JINGGA,
           label=r"teredam dari $(-6, 3)$")
    a.plot(*m, "*", ms=6, color=MERAH, zorder=4)
    a.set_xlabel("$b$")
    a.set_ylabel("$w$")
    a.legend(loc="lower left", fontsize=6)
    a.set_title("jalur Newton")
    e = lambda j: np.maximum([np.abs(t - m).max() for t in j], 1e-16)
    L = np.linalg.eigvalsh(X.T @ X / 24).max()
    b.semilogy(e(jn), "o-", ms=2.5, lw=0.9, color=BIRU, label="Newton")
    b.semilogy(e(gd(X, y, 1 / L, 30)), "-", lw=0.9, color=ABU,
               label="GD, x mentah")
    Xc = rancang(x - 3.5)
    jc = gd(Xc, y, 24 / 17.5, 30)
    jc = np.c_[jc[:, 0] - 3.5 * jc[:, 1], jc[:, 1]]
    b.semilogy(e(jc), "-", lw=0.9, color=JINGGA, label="GD, x dipusat")
    b.set_xlim(0, 30)
    b.set_ylim(1e-16, 10)
    b.set_xlabel("langkah $k$")
    b.set_ylabel(r"$\max_j|\theta_{k,j} - \hat\theta_j|$")
    b.legend(loc="center right", fontsize=6)
    b.set_title("galat per langkah")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-newton")


# ============================ Bab 6 ==================================

def bab06_penalti():
    from bab01_data import data_mini, mle_mini, rancang
    from bab02_loss import log_loss
    from bab06_penalti import Y_PISAH, ista_l1, newton_l2
    x, y = data_mini()
    X = rancang(x)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    t = np.linspace(0, 12, 300)
    v = np.array([-3.5, 1.0])
    a.plot(t, [log_loss(s * v, X, Y_PISAH) for s in t], color=BIRU,
           lw=1.1, label="terpisah")
    a.plot(t, [log_loss(s * v, X, y) for s in t], color=JINGGA, lw=1.0,
           label="data mini")
    a.axhline(0, color=ABU_GARIS, lw=0.5)
    a.set_xlabel(r"$t$ pada $\theta = t\,(-3{,}5;\ 1)$")
    a.set_ylabel(r"$L(\theta)$")
    a.legend(loc="upper right", fontsize=6)
    a.set_title("sepanjang satu sinar")
    C = np.logspace(-2, 2, 41)
    w2 = [newton_l2(X, y, 1 / (6 * c))[1] for c in C]
    w2s = [newton_l2(X, Y_PISAH, 1 / (6 * c))[1] for c in C]
    w1 = [ista_l1(X, y, 1 / (6 * c), 0.25, 4000)[1] for c in C]
    b.semilogx(C, w2, color=BIRU, lw=1.1, label="L2, data mini")
    b.semilogx(C, w1, color=HIJAU, lw=1.1, label="L1, data mini")
    b.semilogx(C, w2s, color=JINGGA, lw=1.0, ls="--",
               label="L2, terpisah")
    b.axhline(mle_mini()[1], color=MERAH, lw=0.6, ls=":")
    b.axvline(1 / (6 * 0.58333), color=ABU, lw=0.5, ls=":")
    b.set_xlabel("$C$ scikit-learn")
    b.set_ylabel("$w$")
    b.legend(loc="upper left", fontsize=6)
    b.set_title("jalur regularisasi")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-penalti")


# ============================ Bab 7 ==================================

def bab07_prediksi():
    from scipy.special import logit
    from bab01_data import data_mini, mle_mini, rancang
    from bab07_prediksi import kovarians, prediksi_selang
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    C = kovarians(m, X)
    g = np.linspace(0, 8, 300)
    p, lo, hi, _ = prediksi_selang(m, C, rancang(g))
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.fill_between(g, lo, hi, color=BIRU_MUDA, lw=0,
                    label="selang 95% (metode delta)")
    ax.plot(g, p, color=BIRU, lw=1.1, label=r"$\hat p(x)$")
    ax.scatter(x, y, s=12, color=np.where(y == 1, BIRU, JINGGA), zorder=3)
    for t, wr in [(0.5, MERAH), (0.2, HIJAU)]:
        xt = (logit(t) - m[0]) / m[1]
        ax.plot([0, xt], [t, t], color=wr, lw=0.6, ls="--")
        ax.plot([xt, xt], [0, t], color=wr, lw=0.6, ls="--")
        ax.annotate(f"$t = {angka_mat(t, 1)}$", (0.1, t + 0.03),
                    fontsize=6, color=wr)
    ax.set_xlabel("jam belajar $x$")
    ax.set_ylabel("peluang lulus")
    ax.legend(loc="lower right", fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-prediksi")


# ============================ Bab 8 ==================================

def bab08_kurva():
    from scipy.special import expit
    from sklearn import metrics
    from bab01_data import data_mini, mle_mini, rancang
    x, y = data_mini()
    p = expit(rancang(x) @ mle_mini())
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    fpr, tpr, _ = metrics.roc_curve(y, p, drop_intermediate=False)
    a.fill_between(fpr, tpr, step=None, color=BIRU_MUDA, lw=0)
    a.plot(fpr, tpr, "o-", ms=3, color=BIRU, lw=1.0)
    a.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    a.annotate("AUC = 8/9", (0.45, 0.35), fontsize=7, color=BIRU)
    a.set_xlabel("FPR = 1 - spesifisitas")
    a.set_ylabel("TPR = recall")
    a.set_title("kurva ROC")
    a.set_aspect("equal")
    urut = np.argsort(-p)
    tp = np.cumsum(y[urut])
    prec = tp / np.arange(1, 7)
    rec = tp / 3
    b.step(np.r_[0, rec], np.r_[1, prec], where="pre", color=JINGGA,
           lw=1.0)
    b.plot(rec, prec, "o", ms=3, color=JINGGA)
    b.axhline(0.5, color=ABU_GARIS, lw=0.6, ls="--")
    b.annotate("AP = 11/12", (0.1, 0.62), fontsize=7, color=JINGGA)
    b.set_xlim(-0.02, 1.02)
    b.set_ylim(0, 1.05)
    b.set_xlabel("recall")
    b.set_ylabel("precision")
    b.set_title("kurva precision-recall")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-kurva")


# ============================ Bab 9 ==================================

def bab09_reliabilitas():
    import warnings
    from sklearn.linear_model import LogisticRegression
    from bab01_data import BENIH
    from bab09_kalibrasi import data_sintetis, platt
    Xl, yl = data_sintetis(200, BENIH)
    Xv, yv = data_sintetis(5000, BENIH + 1)
    Xu, yu = data_sintetis(20000, BENIH + 2)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.7, ls="--")
    tepi = np.linspace(0, 1, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nama, C, kol, wr, gaya in [
                ("3 peubah, MLE", None, slice(0, 3), HIJAU, "o"),
                ("20 peubah, MLE", None, slice(0, 20), MERAH, "s"),
                ("20 peubah, C = 0,01", 0.01, slice(0, 20), JINGGA, "^"),
                ("20 peubah, MLE + Platt", "platt", slice(0, 20), BIRU,
                 "D")]:
            if C == "platt":
                m = LogisticRegression(penalty=None, solver="newton-cholesky",
                                       tol=1e-10).fit(Xl[:, kol], yl)
                f = platt(m.predict_proba(Xv[:, kol])[:, 1], yv)
                pu = f(m.predict_proba(Xu[:, kol])[:, 1])
            else:
                m = (LogisticRegression(penalty=None) if C is None else
                     LogisticRegression(C=C))
                m.set_params(solver="newton-cholesky", tol=1e-10)
                m.fit(Xl[:, kol], yl)
                pu = m.predict_proba(Xu[:, kol])[:, 1]
            idx = np.minimum((pu * 10).astype(int), 9)
            mp = [pu[idx == k].mean() for k in range(10) if (idx == k).sum() > 20]
            my = [yu[idx == k].mean() for k in range(10) if (idx == k).sum() > 20]
            ax.plot(mp, my, gaya + "-", ms=2.8, lw=0.9, color=wr, label=nama)
    ax.set_xlabel("rata-rata peluang dalam kelompok")
    ax.set_ylabel("proporsi positif")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-reliabilitas")


# ============================ Bab 10 =================================

def bab10_profil():
    from bab01_data import data_mini, mle_mini, rancang
    from bab07_prediksi import kovarians
    from bab10_inferensi import log_kem, profil, selang_profil
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    se = np.sqrt(kovarians(m, X)[1, 1])
    lmax = log_kem(m, X, y)
    w = np.linspace(-1.5, 5.5, 141)
    dev = np.array([2 * (lmax - profil(X, y, 1, v)) for v in w])
    wald = ((w - m[1]) / se) ** 2
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(w, dev, color=BIRU, lw=1.1, label="profil (rasio kemungkinan)")
    ax.plot(w, wald, color=JINGGA, lw=1.0, ls="--",
            label="hampiran kuadratik (Wald)")
    ax.axhline(3.8415, color=ABU, lw=0.6, ls=":")
    a, b = selang_profil(X, y, 1)
    for v in (a, b):
        ax.plot([v, v], [0, 3.8415], color=BIRU, lw=0.6)
    for v in (m[1] - 1.96 * se, m[1] + 1.96 * se):
        ax.plot([v, v], [0, 3.8415], color=JINGGA, lw=0.6, ls="--")
    ax.plot([0], [3.3618], "o", ms=3, color=MERAH)
    ax.annotate("$w = 0$: $G = 3{,}36$", (0, 3.36), (0.4, 6.0),
                fontsize=6, color=MERAH,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=MERAH))
    ax.set_ylim(0, 9)
    ax.set_xlabel("$w$")
    ax.set_ylabel(r"$2(\hat\ell - \ell_{\mathrm{p}}(w))$")
    ax.legend(loc="upper right", fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-profil")


# ============================ Bab 11 =================================

def bab11_tafsiran():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    x, y = data_mini()
    m = mle_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    p0 = np.linspace(0.001, 0.95, 300)
    for OR, wr in [(2, HIJAU), (3, BIRU), (5, JINGGA)]:
        o = OR * p0 / (1 - p0)
        a.plot(p0, o / (1 + o) / p0, color=wr, lw=1.0,
               label=f"OR = {OR}")
    a.axhline(1, color=ABU_GARIS, lw=0.5)
    a.set_xlabel("peluang dasar $p_0$")
    a.set_ylabel("rasio risiko $p_1/p_0$")
    a.legend(loc="upper right", fontsize=6)
    a.set_title("OR bukan rasio risiko")
    g = np.linspace(0, 7, 300)
    pg = expit(m[0] + m[1] * g)
    b.plot(g, m[1] * pg * (1 - pg), color=BIRU, lw=1.1,
           label=r"$\hat w\,\hat p(1 - \hat p)$")
    p = expit(rancang(x) @ m)
    b.plot(x, m[1] * p * (1 - p), "o", ms=3, color=BIRU)
    ame = np.mean(m[1] * p * (1 - p))
    b.axhline(ame, color=MERAH, lw=0.8, ls="--",
              label=f"AME = {angka(ame, 4)}")
    b.set_xlabel("jam belajar $x$")
    b.set_ylabel(r"$\partial\hat p/\partial x$")
    b.legend(loc="upper left", fontsize=6)
    b.set_title("efek marginal")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-tafsiran")


# ============================ Bab 12 =================================

def bab12_diagnostik():
    import statsmodels.api as sm
    from scipy.special import expit
    from bab01_data import BENIH, data_mini, mle_mini, rancang
    from bab12_diagnostik import cook, leverage
    x, y = data_mini()
    X = rancang(x)
    p = expit(X @ mle_mini())
    h, _ = leverage(X, p)
    D = cook(y, p, h, 2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    a.bar(x - 0.18, h, width=0.36, color=BIRU, label="leverage $h_i$")
    a.bar(x + 0.18, D, width=0.36, color=MERAH, label="Cook $D_i$")
    a.set_xticks(x)
    a.set_xlabel("mahasiswa (jam belajar)")
    a.legend(loc="upper left", fontsize=6)
    a.set_title("data mini")
    a.set_ylim(0, 0.95)
    rng = np.random.default_rng(BENIH)
    n = 1000
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + np.sqrt(1 - 0.81) * rng.normal(size=n)
    x3 = rng.normal(size=n)
    eta = -0.5 + x1 + 0.5 * x2 + 0.8 * x3 - 0.6 * x3 ** 2
    yy = (rng.random(n) < expit(eta)).astype(int)
    for A, nama, wr, g in [(np.c_[x1, x2, x3], "linear dalam $x_3$", JINGGA,
                            "o"),
                           (np.c_[x1, x2, x3, x3 ** 2], "dengan $x_3^2$",
                            BIRU, "s")]:
        r = sm.Logit(yy, sm.add_constant(A)).fit(disp=0)
        pp = r.predict(sm.add_constant(A))
        urut = np.argsort(x3)
        kel = np.array_split(urut, 20)
        mx = [x3[k].mean() for k in kel]
        mr = [(yy[k] - pp[k]).mean() for k in kel]
        b.plot(mx, mr, g + "-", ms=2.5, lw=0.8, color=wr, label=nama)
    se = 2 * np.sqrt(0.25 / 50)
    b.axhspan(-se, se, color=ABU_GARIS, alpha=0.4, lw=0)
    b.axhline(0, color=ABU, lw=0.5)
    b.set_xlabel("$x_3$ (rata-rata kelompok)")
    b.set_ylabel(r"rata-rata $y - \hat p$")
    b.legend(loc="lower center", fontsize=6)
    b.set_title("residu berkelompok")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-diagnostik")


# ============================ Bab 13 =================================

def bab13_validasi():
    import warnings
    import statsmodels.api as sm
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from bab01_data import BENIH
    from bab09_kalibrasi import data_sintetis
    from bab13_validasi import model_mle
    warnings.simplefilter("ignore")
    X, y = data_sintetis(300, BENIH + 10)
    Xu, yu = data_sintetis(20000, BENIH + 2)
    m = model_mle().fit(X, y)
    tampak = roc_auc_score(y, m.predict_proba(X)[:, 1])
    uji = roc_auc_score(yu, m.predict_proba(Xu)[:, 1])
    auc = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    for a, b in cv.split(X, y):
        mm = model_mle().fit(X[a], y[a])
        auc.append(roc_auc_score(y[b], mm.predict_proba(X[b])[:, 1]))
    rng = np.random.default_rng(BENIH)
    w1 = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        mb = model_mle().fit(X[i], y[i])
        w1.append(mb.coef_[0, 0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    a.hist(auc, bins=12, color=BIRU_MUDA, edgecolor=BIRU, lw=0.5)
    for v, nama, wr, ls in [(tampak, "tampak", MERAH, "-"),
                            (np.mean(auc), "rata-rata CV", BIRU, "--"),
                            (uji, "uji", HIJAU, ":")]:
        a.axvline(v, color=wr, lw=1.0, ls=ls, label=nama)
    a.set_xlabel("AUC per lipatan")
    a.legend(loc="upper left", fontsize=6)
    a.set_title("50 lipatan CV")
    r = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    b.hist(w1, bins=20, density=True, color=JINGGA_MUDA, edgecolor=JINGGA,
           lw=0.5, label="bootstrap")
    g = np.linspace(0.4, 2.4, 200)
    b.plot(g, norm.pdf(g, r.params[1], r.bse[1]), color=BIRU, lw=1.0,
           label="normal Wald")
    b.axvline(1.0, color=ABU, lw=0.6, ls=":")
    b.set_xlabel("$\\hat w_1$")
    b.legend(loc="upper right", fontsize=6)
    b.set_title("200 sampel bootstrap")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-validasi")


# ============================ PENANDA ================================
# Fungsi gambar baru disisipkan DI ATAS penanda ini.


if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
