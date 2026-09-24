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

BENIH = 20260924  # sama dengan seluruh kode/bab*.py
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
# ---------------------------------------------------------------------



# ============================ Bab 4 ==================================

def bab04_sigmoid():
    from bab04_sigmoid import sigmoid
    z = np.linspace(-7, 7, 701)
    s = sigmoid(z)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 1.9))
    a.axhline(0, color=ABU_GARIS, lw=0.5)
    a.axhline(1, color=ABU_GARIS, lw=0.5, ls="--")
    a.plot(z, s, color=BIRU, lw=1.2, label=r"$\sigma(z)$")
    zz = np.linspace(-2.6, 2.6, 2)
    a.plot(zz, 0.5 + zz / 4, color=JINGGA, lw=0.8, ls="--",
           label=r"$\frac{1}{2} + \frac{z}{4}$")
    a.plot([0], [0.5], "o", ms=3, color=BIRU)
    a.set_ylim(-0.12, 1.12)
    a.set_xlabel("$z$")
    a.set_title("fungsi logistik")
    a.legend(loc="upper left", fontsize=6)
    b.plot(z, s * (1 - s), color=HIJAU, lw=1.2)
    b.axhline(0.25, color=ABU_GARIS, lw=0.5, ls="--")
    b.text(3.0, 0.232, r"$\frac{1}{4}$", fontsize=7)
    b.set_ylim(0, 0.29)
    b.set_xlabel("$z$")
    b.set_title(r"turunannya $\sigma(z)\,(1-\sigma(z))$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-sigmoid")


def bab04_odds():
    p = np.linspace(0.002, 0.998, 999)
    odds = p / (1 - p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 1.9))
    a.plot(p, odds, color=JINGGA, lw=1.2)
    a.axhline(1, color=ABU_GARIS, lw=0.5, ls="--")
    a.axvline(0.5, color=ABU_GARIS, lw=0.5, ls="--")
    a.set_ylim(0, 12)
    a.set_xlabel("peluang $p$")
    a.set_title(r"odds $p/(1-p)$")
    b.plot(p, np.log(odds), color=BIRU, lw=1.2)
    b.axhline(0, color=ABU_GARIS, lw=0.5, ls="--")
    b.axvline(0.5, color=ABU_GARIS, lw=0.5, ls="--")
    for q in (0.1, 0.9):
        b.plot([q], [np.log(q / (1 - q))], "o", ms=2.5, color=BIRU)
    b.set_ylim(-6.5, 6.5)
    b.set_xlabel("peluang $p$")
    b.set_title(r"log-odds $\log\,\frac{p}{1-p}$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-odds")


def bab04_model():
    from bab04_data import jam_belajar
    from bab04_model import taksir
    from bab04_sigmoid import sigmoid
    x, y = jam_belajar()
    b0, w = taksir()
    xx = np.linspace(0, 12, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.1))
    # titik yang jamnya sama digeser sedikit supaya terlihat
    geser = np.zeros(len(x))
    for v in np.unique(x):
        i = np.flatnonzero(x == v)
        geser[i] = 0.035 * (np.arange(len(i)) - (len(i) - 1) / 2)
    a.scatter(x, y + geser, s=9, color=np.where(y == 1, BIRU, JINGGA),
              zorder=3)
    a.plot(xx, sigmoid(b0 + w * xx), color=BIRU, lw=1.1)
    x50 = -b0 / w
    zz = np.array([x50 - 2.2, x50 + 2.2])
    a.plot(zz, 0.5 + w / 4 * (zz - x50), color=HIJAU, lw=0.8, ls="--")
    a.plot([x50], [0.5], "o", ms=3, color=HIJAU)
    a.annotate(r"$-b/w$", (x50, 0.5), (x50 + 0.9, 0.3), fontsize=6,
               color=HIJAU, arrowprops=dict(arrowstyle="-", lw=0.4,
                                            color=HIJAU))
    for s in (1, 5, 10):
        p0, p1 = sigmoid(b0 + w * s), sigmoid(b0 + w * (s + 1))
        a.plot([s + 1, s + 1], [p0, p1], color=MERAH, lw=1.4)
        a.plot([s, s + 1], [p0, p0], color=MERAH, lw=0.5)
    a.set_xlabel("jam belajar per minggu")
    a.set_ylabel("peluang lulus")
    a.set_title("skala peluang")
    zlin = b0 + w * xx
    b.plot(xx, zlin, color=BIRU, lw=1.1)
    for s in (1, 5, 10):
        b.plot([s, s + 1, s + 1], [b0 + w * s] * 2 + [b0 + w * (s + 1)],
               color=MERAH, lw=0.8)
    b.axhline(0, color=ABU_GARIS, lw=0.5, ls="--")
    b.set_xlabel("jam belajar per minggu")
    b.set_ylabel("log-odds lulus")
    b.set_title("skala logit")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-model")


def bab04_kontur():
    from bab04_data import B_BENAR, W_BENAR, dua_peubah
    X, y = dua_peubah()
    g = np.linspace(-3.2, 3.2, 300)
    G1, G2 = np.meshgrid(g, g)
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.35))
    for ax, c in zip(axs, (1, 3)):
        Z = c * (W_BENAR[0] * G1 + W_BENAR[1] * G2 + B_BENAR)
        P = 1 / (1 + np.exp(-Z))
        ax.contourf(G1, G2, P, levels=[0, .1, .25, .5, .75, .9, 1],
                    colors=[JINGGA_MUDA, "#FEF6EE", "#FFFBF7",
                            "#F6F8FC", "#EEF2F9", BIRU_MUDA])
        cs = ax.contour(G1, G2, P, levels=[.1, .25, .75, .9],
                        colors=ABU, linewidths=0.4, linestyles="--")
        ax.clabel(cs, fmt=lambda v: angka(v, 2), fontsize=5)
        ax.contour(G1, G2, P, levels=[.5], colors=HIJAU, linewidths=1)
        ax.scatter(*X[y == 1].T, s=4, color=BIRU, lw=0)
        ax.scatter(*X[y == 0].T, s=5, facecolors="none",
                   edgecolors=JINGGA, lw=0.4)
        # vektor bobot, digambar dari titik di batas keputusan
        t0 = -B_BENAR * W_BENAR / (W_BENAR @ W_BENAR)
        u = W_BENAR / np.linalg.norm(W_BENAR)
        ax.annotate("", t0 + 1.8 * u, t0,
                    arrowprops=dict(arrowstyle="-|>", color=MERAH,
                                    lw=1.4, mutation_scale=8))
        ax.text(*(t0 + 1.8 * u + [0.1, -0.35]), r"$\mathbf{w}$",
                color=MERAH, fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("$x_1$")
        ax.set_title(r"$(\mathbf{w}, b)$" if c == 1
                     else r"$3\,(\mathbf{w}, b)$")
    axs[0].set_ylabel("$x_2$")
    fig.tight_layout()
    simpan(fig, "bab04-kontur")


def bab04_asal():
    from scipy.stats import norm
    from bab04_sigmoid import sigmoid
    x = np.linspace(-4, 6, 500)
    pi1 = 0.3
    f0 = (1 - pi1) * norm.pdf(x, 0, 1)
    f1 = pi1 * norm.pdf(x, 2, 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    a.fill_between(x, f0, color=JINGGA_MUDA)
    a.plot(x, f0, color=JINGGA, lw=0.9, label=r"$0{,}7\,\mathcal{N}(0,1)$")
    a.fill_between(x, f1, color=BIRU_MUDA)
    a.plot(x, f1, color=BIRU, lw=0.9, label=r"$0{,}3\,\mathcal{N}(2,1)$")
    a2 = a.twinx()
    a2.plot(x, f1 / (f0 + f1), color=HIJAU, lw=1.1)
    a2.plot(x, sigmoid(2 * x - 2 + np.log(pi1 / (1 - pi1))), color=MERAH,
            lw=0.6, ls=(0, (3, 3)))
    a2.set_ylim(0, 1.05)
    a2.tick_params(colors=HIJAU)
    a2.spines["top"].set_visible(False)
    a.set_xlabel("$x$")
    a.set_title(r"posterior $\mathbb{P}(y=1\mid x)$ (hijau)")
    a.legend(loc="upper left", fontsize=5.5)
    a.spines["top"].set_visible(False)
    z = np.linspace(-5, 5, 500)
    b.plot(z, norm.cdf(z), color=BIRU, lw=1.1, label=r"probit $\Phi(z)$")
    b.plot(z, sigmoid(z), color=ABU, lw=0.8, ls="--",
           label=r"$\sigma(z)$")
    b.plot(z, sigmoid(1.702 * z), color=MERAH, lw=0.7, ls=(0, (3, 3)),
           label=r"$\sigma(1{,}702\,z)$")
    b.set_xlabel("$z$")
    b.set_title("logistik lawan probit")
    b.legend(fontsize=5.5, loc="upper left")
    _rapikan(b)
    fig.tight_layout()
    simpan(fig, "bab04-asal")


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
