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
