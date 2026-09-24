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



# ============================ Bab 5 ==================================

def bab05_lanskap():
    from scipy.special import expit
    from bab04_data import jam_belajar
    from bab04_model import taksir
    x, y = jam_belajar()
    bb = np.linspace(-9, 2, 400)
    ww = np.linspace(-0.3, 1.6, 400)
    B, W = np.meshgrid(bb, ww)
    P = expit(B[..., None] + W[..., None] * x)
    LL = np.sum(np.where(y == 1, np.log(P), np.log(1 - P)), axis=-1)
    b0, w0 = taksir()
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    tingkat = [-30, -20, -15, -12, -10, -9, -8.5, -8, -7.6]
    cs = ax.contour(B, W, LL, levels=tingkat, colors=BIRU,
                    linewidths=0.5)
    ax.clabel(cs, levels=[-30, -20, -12, -9, -8],
              fmt=lambda v: angka(v, 1).replace(",0", ""), fontsize=5)
    ax.contourf(B, W, LL, levels=[-7.6, -7.3], colors=[BIRU_MUDA])
    ax.plot([b0], [w0], "o", ms=3.5, color=MERAH)
    ax.annotate("MLE", (b0, w0), (b0 + 1.2, w0 + 0.25), fontsize=6,
                color=MERAH, arrowprops=dict(arrowstyle="-", lw=0.4,
                                             color=MERAH))
    for b, w in ((0, 0), (-2, 0.3), (-5, 1.0)):
        ax.plot([b], [w], "s", ms=2.8, color=JINGGA)
    ax.set_xlabel("intersep $b$")
    ax.set_ylabel("bobot $w$")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-lanskap")


def bab05_perbutir():
    from scipy.special import expit
    p = np.linspace(0.005, 0.995, 400)
    z = np.linspace(-6, 6, 600)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 1.95))
    a.plot(p, -np.log(p), color=BIRU, lw=1.1, label="$y = 1$: $-\\log p$")
    a.plot(p, -np.log(1 - p), color=JINGGA, lw=1.1,
           label="$y = 0$: $-\\log(1-p)$")
    a.set_ylim(0, 5)
    a.set_xlabel("peluang kelas 1, $p$")
    a.set_title("log-loss satu titik")
    a.legend(fontsize=5.5, loc="upper center")
    b.plot(z, np.logaddexp(0, -z), color=BIRU, lw=1.1, label="log-loss")
    b.plot(z, (expit(z) - 1) ** 2, color=MERAH, lw=1.0, ls="--",
           label="kuadrat galat")
    b.step(z, (z <= 0).astype(float), color=ABU, lw=0.8, where="post",
           label="galat 0-1")
    b.set_ylim(0, 4)
    b.set_xlabel("skor $z$ (label $y = 1$)")
    b.set_title("sebagai fungsi skor")
    b.legend(fontsize=5.5)
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-perbutir")


def bab05_kuadrat():
    from scipy.special import expit
    import bab05_kuadrat as k5
    z = np.linspace(-2, 12, 500)
    p = expit(z)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    a.semilogy(z, p, color=BIRU, lw=1.1, label="log-loss: $\\sigma$")
    a.semilogy(z, 2 * p * p * (1 - p), color=MERAH, lw=1.0, ls="--",
               label="kuadrat: $2\\sigma^2(1-\\sigma)$")
    a.set_xlabel("skor $z$ (label $y = 0$)")
    a.set_title("besar gradien terhadap $z$")
    a.legend(fontsize=5.5, loc="lower left")
    kunci_label(a, "y")
    for nama, f, g, w, gaya in (
            ("log-loss", k5.log_loss, k5.grad_log_loss, BIRU, "-"),
            ("kuadrat galat", k5.kuadrat, k5.grad_kuadrat, MERAH, "--")):
        t = np.array([10.0, -2.0])
        f_min = f(k5.gd(g, (0.0, 0.0))[100000])
        langkah, sisa = [], []
        for i in range(60001):
            if i % 50 == 0 and f(t) - f_min > 1e-12:
                langkah.append(i + 1)
                sisa.append(f(t) - f_min)
            t = t - 0.05 * g(t)
        b.loglog(langkah, sisa, color=w, lw=1.0, ls=gaya, label=nama)
    b.set_xlabel("langkah (mulai dari $(b, w) = (10, -2)$)")
    b.set_title("loss dikurangi minimumnya")
    b.legend(fontsize=5.5, loc="lower left")
    kunci_label(b, "x", "y")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-kuadrat")


def bab05_margin():
    m = np.linspace(-2.5, 3, 500)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.step(m, (m <= 0).astype(float), color=ABU, lw=0.9, where="post",
            label="galat 0-1")
    ax.plot(m, np.logaddexp(0, -m) / np.log(2), color=BIRU, lw=1.2,
            label="logistik $/ \\log 2$")
    ax.plot(m, np.maximum(0, 1 - m), color=HIJAU, lw=1.0, ls="--",
            label="hinge (SVM)")
    ax.plot(m, np.exp(-m), color=JINGGA, lw=1.0, ls=":",
            label="eksponensial (AdaBoost)")
    ax.set_ylim(0, 4)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.set_xlabel("margin $m = s\\,z$")
    ax.set_ylabel("loss")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-margin")


def bab05_jujur():
    q = np.linspace(0.002, 0.998, 500)
    ps = 0.3
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    kurva = (
        ("log-loss", -(ps * np.log(q) + (1 - ps) * np.log(1 - q)), BIRU,
         "-"),
        ("kuadrat galat", ps * (1 - q) ** 2 + (1 - ps) * q ** 2, MERAH,
         "--"),
        ("galat mutlak", ps * (1 - q) + (1 - ps) * q, JINGGA, ":"),
    )
    for nama, h, w, gaya in kurva:
        ax.plot(q, h / h.min(), color=w, lw=1.1, ls=gaya, label=nama)
        ax.plot([q[np.argmin(h)]], [1], "o", ms=3, color=w)
    ax.axvline(ps, color=ABU_GARIS, lw=0.5)
    ax.set_ylim(0.9, 2.6)
    ax.set_xlabel("peluang jawaban model $q$  ($p^* = 0{,}3$)")
    ax.set_ylabel("harapan loss / minimumnya")
    ax.legend(fontsize=5.5, loc="upper center")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-jujur")



# ============================ Bab 6 ==================================

def bab06_sisa():
    from scipy.special import expit
    from bab04_data import jam_belajar
    from bab04_model import taksir
    x, y = jam_belajar()
    geser = np.zeros(len(x))
    for v in np.unique(x):
        i = np.flatnonzero(x == v)
        geser[i] = 0.12 * (np.arange(len(i)) - (len(i) - 1) / 2)
    xx = np.linspace(0, 12, 300)
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0), sharey=True)
    for ax, (b, w), judul in zip(axs, ((0.0, 0.0), taksir()),
                                 (r"$\theta = 0$", "MLE")):
        p = expit(b + w * x)
        ax.plot(xx, expit(b + w * xx), color=ABU, lw=0.8)
        for xi, yi, pi, g in zip(x, y, p, geser):
            ax.annotate("", (xi + g, pi), (xi + g, yi),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                        mutation_scale=5,
                                        color=MERAH if yi == 0
                                        else BIRU))
        ax.scatter(x + geser, y, s=7, zorder=3,
                   color=np.where(y == 1, BIRU, JINGGA))
        ax.set_xlabel("jam belajar")
        ax.set_title(f"sisa $p_i - y_i$ di {judul}")
        _rapikan(ax)
    axs[0].set_ylabel("peluang lulus")
    fig.tight_layout()
    simpan(fig, "bab06-sisa")


def bab06_bobot():
    from scipy.special import expit
    from bab04_data import jam_belajar
    from bab04_model import taksir
    from bab06_turunan import hessian, rancang
    x, y = jam_belajar()
    b, w = taksir()
    xx = np.linspace(0, 12, 300)
    fig, (a, c) = plt.subplots(1, 2, figsize=(4.7, 1.95))
    pp = expit(b + w * xx)
    a.plot(xx, pp * (1 - pp), color=HIJAU, lw=1.0)
    p = expit(b + w * x)
    a.vlines(x, 0, p * (1 - p), color=HIJAU, lw=0.8)
    a.plot(x, p * (1 - p), "o", ms=2.5, color=HIJAU)
    a.axvline(-b / w, color=ABU_GARIS, lw=0.5, ls="--")
    a.set_xlabel("jam belajar")
    a.set_title(r"bobot $d_i = p_i(1 - p_i)$ di MLE")
    X = rancang(x)
    t = np.geomspace(0.5, 30, 60)
    e = np.array([np.linalg.eigvalsh(hessian(s * np.array([b, w]), X,
                                              y)) for s in t])
    c.loglog(t, e[:, 0], color=BIRU, lw=1.1, label="terkecil")
    c.loglog(t, e[:, 1], color=JINGGA, lw=1.0, ls="--",
             label="terbesar")
    c.set_xlabel(r"$t$ pada sinar $t\,\hat{\theta}$")
    c.set_title("nilai eigen Hessian")
    c.legend(fontsize=5.5, loc="lower left")
    kunci_label(c, "x", "y")
    for ax in (a, c):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-bobot")


def _loss_grid(X, y, B, W):
    Z = B[..., None] * X[:, 0] + W[..., None] * X[:, 1]
    return np.mean(np.logaddexp(0, Z) - y * Z, axis=-1)


def bab06_taylor():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import hessian, log_loss, rancang
    x, y = jam_belajar()
    X = rancang(x)
    t = sm.Logit(y, X).fit(disp=0).params
    H = hessian(t, X, y)
    B, W = np.meshgrid(np.linspace(-9, 2, 300),
                       np.linspace(-0.2, 1.5, 300))
    L = _loss_grid(X, y, B, W)
    dB, dW = B - t[0], W - t[1]
    Q = log_loss(t, X, y) + 0.5 * (H[0, 0] * dB ** 2
                                  + 2 * H[0, 1] * dB * dW
                                  + H[1, 1] * dW ** 2)
    tingkat = log_loss(t, X, y) + np.array([0.02, 0.08, 0.2, 0.5])
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.contour(B, W, L, levels=tingkat, colors=BIRU, linewidths=0.9)
    ax.contour(B, W, Q, levels=tingkat, colors=MERAH, linewidths=0.7,
               linestyles="--")
    ax.plot(*t, "o", ms=3, color=MERAH)
    ax.plot([], [], color=BIRU, lw=0.9, label="log-loss")
    ax.plot([], [], color=MERAH, lw=0.7, ls="--",
            label="hampiran kuadratik")
    ax.set_xlabel("intersep $b$")
    ax.set_ylabel("bobot $w$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-taylor")


def bab06_pusat():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import log_loss, rancang
    x, y = jam_belajar()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.2))
    for ax, u, judul in zip(axs, (x, x - x.mean()),
                            ("$x$ mentah", "$x$ dipusatkan")):
        X = rancang(u)
        t = sm.Logit(y, X).fit(disp=0).params
        B, W = np.meshgrid(np.linspace(t[0] - 3, t[0] + 3, 300),
                           np.linspace(t[1] - 0.75, t[1] + 0.75, 300))
        L = _loss_grid(X, y, B, W)
        f0 = log_loss(t, X, y)
        ax.contour(B, W, L, levels=f0 + np.array([.01, .03, .08, .2,
                                                  .4, .8]),
                   colors=BIRU, linewidths=0.6)
        ax.plot(*t, "o", ms=3, color=MERAH)
        ax.set_xlabel("intersep")
        ax.set_title(judul)
        _rapikan(ax)
    axs[0].set_ylabel("bobot $w$")
    fig.tight_layout()
    simpan(fig, "bab06-pusat")


def bab06_kolinear():
    from bab04_data import jam_belajar
    from bab04_model import taksir
    x, y = jam_belajar()
    b, w = taksir()
    W1, W2 = np.meshgrid(np.linspace(-0.6, 1.6, 300),
                         np.linspace(-0.6, 0.9, 300))
    Z = b + (W1[..., None] + 2 * W2[..., None]) * x
    L = np.mean(np.logaddexp(0, Z) - y * Z, axis=-1)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.contour(W1, W2, L, levels=L.min() + np.array([.01, .05, .15,
                                                    .4, 1]),
               colors=BIRU, linewidths=0.6)
    ww = np.linspace(-0.6, 1.6, 2)
    ax.plot(ww, (w - ww) / 2, color=MERAH, lw=1.1,
            label=r"$w_1 + 2w_2 = \hat w$")
    for t in ((w, 0), (0, w / 2), (0.1279, 0.2557)):
        ax.plot(*t, "o", ms=2.8, color=MERAH)
    ax.set_xlabel("$w_1$ (bobot $x$)")
    ax.set_ylabel("$w_2$ (bobot $2x$)")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-kolinear")


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
