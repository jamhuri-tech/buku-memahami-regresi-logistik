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
