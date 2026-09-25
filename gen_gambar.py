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

def bab01_linear():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from bab04_data import jam_belajar
    x, y = jam_belajar()
    x2 = np.r_[x, 25.0, 30.0, 35.0]
    y2 = np.r_[y, 1, 1, 1]
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.1))
    for ax, (xx, yy), judul, ujung in zip(
            axs, [(x, y), (x2, y2)],
            ["16 mahasiswa", "ditambah 3 yang belajar lama"], [14, 37]):
        lin = LinearRegression().fit(xx[:, None], yy)
        log = LogisticRegression(penalty=None, tol=1e-10).fit(
            xx[:, None], yy)
        g = np.linspace(0, ujung, 400)
        geser = np.zeros(len(xx))
        for v in np.unique(xx):
            i = np.flatnonzero(xx == v)
            geser[i] = 0.035 * (np.arange(len(i)) - (len(i) - 1) / 2)
        ax.scatter(xx, yy + geser, s=9,
                   color=np.where(yy == 1, BIRU, JINGGA), zorder=3)
        ax.plot(g, lin.predict(g[:, None]), color=MERAH, lw=1.0,
                label="garis lurus")
        ax.plot(g, log.predict_proba(g[:, None])[:, 1], color=BIRU,
                lw=1.1, label="logistik")
        ax.axhspan(-0.6, 0, color=MERAH_MUDA, lw=0, zorder=0)
        ax.axhspan(1, 1.6, color=MERAH_MUDA, lw=0, zorder=0)
        t_lin = (0.5 - lin.intercept_) / lin.coef_[0]
        t_log = -log.intercept_[0] / log.coef_[0, 0]
        ax.plot([t_lin], [0.5], "o", ms=3, color=MERAH, zorder=4)
        ax.plot([t_log], [0.5], "o", ms=3, color=BIRU, zorder=4)
        ax.set_ylim(-0.35, 1.35)
        ax.set_xlim(0, ujung)
        ax.set_xlabel("jam belajar per minggu")
        ax.set_title(judul)
        _rapikan(ax)
    axs[0].set_ylabel("lulus / peluang lulus")
    axs[0].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab01-linear")


def bab02_lanskap():
    from bab02_turunan import f, hess_f
    u = np.linspace(-2, 1.5, 300)
    v = np.linspace(-1.5, 2, 300)
    U, V = np.meshgrid(u, v)
    F = f((U, V))
    G = U ** 4 - 2 * U ** 2 + V ** 2
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    a.contour(U, V, F, levels=np.linspace(1.6, 9, 12), colors=BIRU,
              linewidths=0.6)
    from scipy.optimize import minimize
    m = minimize(f, [0.0, 0.0]).x
    a.plot(*m, "o", ms=3, color=MERAH)
    a.set_title("$f$: Hessian definit positif")
    uu = np.linspace(-1.6, 1.6, 300)
    U, V = np.meshgrid(uu, np.linspace(-1.3, 1.3, 300))
    G = U ** 4 - 2 * U ** 2 + V ** 2
    b.contour(U, V, G, levels=np.linspace(-0.9, 2.5, 14), colors=BIRU,
              linewidths=0.6)
    b.axvspan(-1 / np.sqrt(3), 1 / np.sqrt(3), color=MERAH_MUDA, lw=0,
              zorder=0)
    b.plot([-1, 1], [0, 0], "o", ms=3, color=MERAH)
    b.plot([0], [0], "x", ms=4, color=MERAH)
    b.set_title("$g$: pelana di pusat")
    for ax in (a, b):
        ax.set_xlabel("$u$")
        ax.set_aspect("equal")
        _rapikan(ax)
    a.set_ylabel("$v$")
    fig.tight_layout()
    simpan(fig, "bab02-lanskap")


def bab02_optim():
    from bab02_optimasi import T_BINTANG, gd, newton
    z0 = 1.0
    zz = np.linspace(-3, 4, 300)
    s0 = np.log1p(np.exp(z0))
    s1 = 1 / (1 + np.exp(-z0))
    s2 = s1 * (1 - s1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.1))
    a.plot(zz, np.log1p(np.exp(zz)), color=BIRU, lw=1.1,
           label="$\\log(1 + e^z)$")
    a.plot(zz, s0 + s1 * (zz - z0), color=ABU, lw=0.7, ls="--",
           label="orde satu")
    a.plot(zz, s0 + s1 * (zz - z0) + s2 * (zz - z0) ** 2 / 2,
           color=MERAH, lw=0.9, label="orde dua")
    a.plot([z0], [s0], "o", ms=3, color=MERAH)
    a.set_ylim(-1, 4.5)
    a.set_xlabel("$z$")
    a.set_title("hampiran Taylor di $z_0 = 1$")
    a.legend(loc="upper left", fontsize=6)
    k = np.arange(13)
    ga = np.abs(gd(0.0, 4.0, 12) - T_BINTANG)
    gb = np.abs(newton(0.0, 12) - T_BINTANG)
    gb = np.maximum(gb, 1e-17)
    b.semilogy(k, ga, "o-", ms=2.5, lw=0.8, color=BIRU,
               label="gradient descent")
    b.semilogy(k[:5], gb[:5], "s-", ms=2.5, lw=0.8, color=MERAH,
               label="Newton")
    b.set_ylim(1e-16, 3)
    b.set_xlabel("langkah $k$")
    b.set_ylabel("$|t_k - t^*|$")
    b.set_title("galat per langkah")
    b.legend(loc="lower left", fontsize=6)
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-optim")


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



# ============================ Bab 7 ==================================

def bab07_terpisah():
    from scipy.special import expit
    from bab06_turunan import log_loss, rancang
    from bab07_data import jam_terpisah
    x, y = jam_terpisah()
    xx = np.linspace(0, 12, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 1.95))
    a.scatter(x, y, s=8, zorder=3, color=np.where(y == 1, BIRU, JINGGA))
    for t, w in zip((1, 2, 5, 20), (ABU_GARIS, ABU, HIJAU, MERAH)):
        a.plot(xx, expit(t * (xx - 5.5)), color=w, lw=1.0,
               label=f"$t = {t}$")
    a.set_xlabel("jam belajar")
    a.set_ylabel("peluang lulus")
    a.set_title(r"$\sigma(t\,(x - 5{,}5))$")
    a.legend(fontsize=5.5, loc="lower right")
    X = rancang(x)
    t = np.linspace(0.2, 40, 300)
    L = [log_loss(s * np.array([-5.5, 1.0]), X, y) for s in t]
    b.semilogy(t, L, color=BIRU, lw=1.1)
    b.set_xlabel("$t$")
    b.set_title("log-loss sepanjang sinar")
    kunci_label(b, "y")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-terpisah")


def bab07_arah():
    from bab07_arah import jalankan_gd, margin_maksimum
    from bab07_data import gumpalan_terpisah
    X, y = gumpalan_terpisah()
    svm = margin_maksimum(X, y)
    catat = jalankan_gd(X, y)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(*X[y == 1].T, s=7, color=BIRU, zorder=3)
    ax.scatter(*X[y == 0].T, s=8, facecolors="none", edgecolors=JINGGA,
               lw=0.6, zorder=3)
    g = np.array([-3.5, 3.5])
    for c, w, gaya in ((0, HIJAU, "-"), (1, HIJAU, ":"), (-1, HIJAU, ":")):
        ax.plot(g, (c - svm[0] - svm[1] * g) / svm[2], color=w, lw=1.1,
                ls=gaya)
    for k, w in ((10, ABU_GARIS), (100, ABU), (10**4, MERAH)):
        th = catat[k]
        ax.plot(g, (-th[0] - th[1] * g) / th[2], color=w, lw=0.8,
                ls="--", label=f"GD, $t = {k:g}$")
    ax.plot([], [], color=HIJAU, lw=1.1, label="margin maksimum")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=5.2, loc="lower right", framealpha=0.9,
              frameon=True, edgecolor="none")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-arah")


def bab07_laju():
    from bab07_arah import LANGKAH, jalankan_gd, margin_maksimum
    from bab07_data import gumpalan_terpisah
    X, y = gumpalan_terpisah()
    svm = margin_maksimum(X, y)
    u = svm / np.linalg.norm(svm)
    catat = jalankan_gd(X, y)
    k = np.array(sorted(catat))
    nr = np.array([np.linalg.norm(catat[i]) for i in k])
    sudut = np.degrees(np.arccos(np.minimum(1, [catat[i] @ u / r
                                                for i, r in zip(k, nr)])))
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 1.95))
    a.semilogx(k, nr, "o-", ms=2, lw=0.9, color=BIRU,
               label=r"$\|\theta(t)\|$")
    tt = np.geomspace(1e3, 1e6, 50)
    a.semilogx(tt, nr[-1] + np.linalg.norm(svm) * np.log(tt / k[-1]),
               color=MERAH, lw=0.8, ls="--",
               label=r"kemiringan $\|\theta_{\mathrm{svm}}\|$")
    a.set_xlabel("langkah $t$")
    a.set_title("norma parameter")
    a.legend(fontsize=5.5, loc="upper left")
    kunci_label(a, "x")
    b.semilogx(k, sudut, "o-", ms=2, lw=0.9, color=HIJAU)
    b.set_xlabel("langkah $t$")
    b.set_title("sudut ke arah margin maksimum (derajat)")
    kunci_label(b, "x")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-laju")


def bab07_cover():
    from bab07_deteksi import peluang_cover, percobaan_cover
    n = 60
    d = np.arange(1, 60)
    teori = [peluang_cover(n, k + 1) for k in d]
    empiris = percobaan_cover(n)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot((d + 1) / n, teori, color=BIRU, lw=1.1, label="rumus Cover")
    ax.plot([(k + 1) / n for k in empiris], list(empiris.values()), "o",
            ms=3, color=MERAH, label="program linear, 200 ulangan")
    ax.axvline(0.5, color=ABU_GARIS, lw=0.5, ls="--")
    ax.set_xlabel("$(d + 1)/n$")
    ax.set_ylabel("peluang terpisah")
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-cover")



# ============================ Bab 8 ==================================

def _jalur_gd(X, y, eta, langkah, momentum=False):
    from bab06_turunan import gradien
    theta = v = np.zeros(X.shape[1])
    jalur = [theta]
    for k in range(1, langkah + 1):
        baru = v - eta * gradien(v, X, y)
        v = baru + ((k - 1) / (k + 2) * (baru - theta) if momentum
                    else 0)
        theta = baru
        jalur.append(theta)
    return np.array(jalur)


def bab08_lintasan():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    x, y = jam_belajar()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for ax, u, judul, langkah in zip(
            axs, (x, x - x.mean()), ("$x$ mentah, 400 langkah",
                                     "$x$ dipusatkan, 40 langkah"),
            (400, 40)):
        X = rancang(u)
        t = sm.Logit(y, X).fit(disp=0).params
        L = np.linalg.eigvalsh(X.T @ X / len(y)).max() / 4
        B, W = np.meshgrid(np.linspace(min(0, t[0]) - 1,
                                       max(0, t[0]) + 1, 300),
                           np.linspace(-0.15, 1.0, 300))
        L_ = _loss_grid(X, y, B, W)
        ax.contour(B, W, L_, levels=L_.min() + np.array(
            [.01, .03, .08, .2, .4, .8]), colors=ABU_GARIS,
            linewidths=0.5)
        jn = _jalur_gd(X, y, 1 / L, langkah, momentum=True)
        ax.plot(jn[:, 0], jn[:, 1], "--", color=JINGGA, lw=0.7)
        j = _jalur_gd(X, y, 1 / L, langkah)
        ax.plot(j[:, 0], j[:, 1], "-", color=BIRU, lw=0.9, zorder=3)
        ax.plot(j[::10, 0], j[::10, 1], "o", ms=1.6, color=BIRU,
                zorder=3)
        ax.plot(*t, "*", ms=6, color=MERAH)
        ax.set_title(judul)
        ax.set_xlabel("intersep")
        _rapikan(ax)
    axs[0].set_ylabel("bobot $w$")
    axs[0].plot([], [], color=BIRU, lw=0.8, label="GD")
    axs[0].plot([], [], color=JINGGA, lw=0.7, ls="--", label="Nesterov")
    axs[0].legend(fontsize=5.5, loc="upper right")
    fig.tight_layout()
    simpan(fig, "bab08-lintasan")


def bab08_laju():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    x, y = jam_belajar()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0), sharey=True)
    for ax, u, judul, langkah in zip(axs, (x, x - x.mean()),
                                     ("$x$ mentah", "$x$ dipusatkan"),
                                     (10000, 400)):
        X = rancang(u)
        t = sm.Logit(y, X).fit(disp=0).params
        L = np.linalg.eigvalsh(X.T @ X / len(y)).max() / 4
        for c, w, gaya in ((1, BIRU, "-"), (2, HIJAU, "-"),
                           (4, MERAH, ":")):
            j = _jalur_gd(X, y, c / L, langkah)
            ax.semilogy(np.linalg.norm(j - t, axis=1), color=w, lw=0.9,
                        ls=gaya, label=f"GD, $\\eta = {c}/L$")
        j = _jalur_gd(X, y, 1 / L, langkah, momentum=True)
        ax.semilogy(np.linalg.norm(j - t, axis=1), color=JINGGA, lw=0.5,
                    alpha=0.8, label="Nesterov")
        ax.set_ylim(1e-7, 20)
        ax.set_xlabel("langkah")
        ax.set_title(judul)
        kunci_label(ax, "y")
        _rapikan(ax)
    axs[0].set_ylabel(r"$\|\theta_k - \hat\theta\|$")
    axs[1].legend(fontsize=5.2, loc="upper right")
    fig.tight_layout()
    simpan(fig, "bab08-laju")


def bab08_epoch():
    import statsmodels.api as sm
    from bab08_sgd import data_besar, loss, semua_metode
    X, y = data_besar()
    ref = sm.Logit(y, X).fit(disp=0).params
    L_opt = loss(ref, X, y)
    hasil, _, _ = semua_metode(X, y)
    warna = [BIRU, ABU, ABU_GARIS, HIJAU, MERAH]
    gaya = ["-", "--", ":", "-.", "-"]
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    for (nama, catat), w, g in zip(hasil.items(), warna, gaya):
        sel = [loss(c, X, y) - L_opt for c in catat]
        ax.semilogy(range(1, len(sel) + 1), sel, color=w, ls=g, lw=1.0,
                    marker="o", ms=1.8,
                    label=nama.replace("eta", "$\\eta$")
                    .replace("L_maks", "$L_{\\mathrm{maks}}$"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("log-loss dikurangi minimum")
    ax.legend(fontsize=5.2, loc="lower left")
    kunci_label(ax, "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-epoch")



# ============================ Bab 9 ==================================

def bab09_kuadratik():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    x, y = jam_belajar()
    u = x - x.mean()
    X = rancang(u)
    b0 = sm.Logit(y, X).fit(disp=0).params[0]
    w = np.linspace(-0.4, 1.6, 400)
    Z = b0 + w[:, None] * u
    f = np.mean(np.logaddexp(0, Z) - y * Z, axis=1)

    def turunan(wk):
        from scipy.special import expit
        p = expit(b0 + wk * u)
        return (np.mean((p - y) * u), np.mean(p * (1 - p) * u ** 2),
                np.mean(np.logaddexp(0, b0 + wk * u) - y * (b0 + wk * u)))
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.plot(w, f, color=BIRU, lw=1.2, label="log-loss")
    wk = 0.0
    for k, warna in zip(range(3), (JINGGA, HIJAU, MERAH)):
        g, h, fk = turunan(wk)
        q = fk + g * (w - wk) + 0.5 * h * (w - wk) ** 2
        ax.plot(w, q, color=warna, lw=0.8, ls="--",
                label=f"hampiran di $w_{k}$")
        ax.plot([wk], [fk], "o", ms=3, color=warna)
        baru = wk - g / h
        ax.annotate("", (baru, fk + g * (baru - wk)
                         + 0.5 * h * (baru - wk) ** 2), (wk, fk),
                    arrowprops=dict(arrowstyle="->", lw=0.6,
                                    color=warna))
        wk = baru
    ax.set_ylim(0.44, 0.75)
    ax.set_xlabel("bobot $w$ (intersep tetap di MLE)")
    ax.set_ylabel("log-loss")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-kuadratik")


def bab09_lintasan():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    from bab09_newton import newton, newton_teredam
    x, y = jam_belajar()
    X = rancang(x)
    t = sm.Logit(y, X).fit(disp=0).params
    B, W = np.meshgrid(np.linspace(-6, 12, 300),
                       np.linspace(-2.3, 1.3, 300))
    L_ = _loss_grid(X, y, B, W)
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.contour(B, W, L_, levels=np.geomspace(0.47, 8, 9), colors=ABU_GARIS,
               linewidths=0.5)
    L = np.linalg.eigvalsh(X.T @ X / len(y)).max() / 4
    j = _jalur_gd(X, y, 1 / L, 30)
    ax.plot(j[:, 0], j[:, 1], "-o", ms=1.5, lw=0.7, color=ABU,
            label="GD, 30 langkah")
    jn = np.array(newton(X, y, np.zeros(2), 6))
    ax.plot(jn[:, 0], jn[:, 1], "-o", ms=2.5, lw=0.9, color=BIRU,
            label="Newton dari 0")
    theta, jalur = np.array([10.0, -2.0]), [np.array([10.0, -2.0])]
    for _ in range(8):
        theta, _ = newton_teredam(X, y, theta, maks=1)
        jalur.append(theta)
    jd = np.array(jalur)
    ax.plot(jd[:, 0], jd[:, 1], "-s", ms=2.5, lw=0.9, color=JINGGA,
            label="Newton teredam dari $(10, -2)$")
    ax.plot(*t, "*", ms=7, color=MERAH, zorder=4)
    ax.set_xlabel("intersep $b$")
    ax.set_ylabel("bobot $w$")
    ax.legend(fontsize=5.3, loc="lower left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-lintasan")


def bab09_galat():
    import statsmodels.api as sm
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    from bab09_newton import newton
    x, y = jam_belajar()
    X = rancang(x)
    t = sm.Logit(y, X).fit(disp=0).params
    L = np.linalg.eigvalsh(X.T @ X / len(y)).max() / 4
    jn = np.array(newton(X, y, np.zeros(2), 6))
    jg = _jalur_gd(X, y, 1 / L, 30)
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    en = np.maximum(np.linalg.norm(jn - t, axis=1), 1e-17)
    ax.semilogy(range(len(en)), en, "-o", ms=3, color=BIRU, label="Newton")
    ax.semilogy(range(len(jg)), np.linalg.norm(jg - t, axis=1), "-o",
                ms=1.8, lw=0.8, color=ABU, label="GD, $\\eta = 1/L$")
    ax.axhline(2.2e-16 * np.linalg.norm(t), color=ABU_GARIS, lw=0.5,
               ls="--")
    ax.set_xlabel("iterasi")
    ax.set_ylabel(r"$\|\theta_k - \hat\theta\|$")
    ax.legend(fontsize=5.5, loc="center right")
    kunci_label(ax, "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-galat")



# ============================ Bab 10 =================================

def bab10_jalur():
    import warnings
    from sklearn.linear_model import LogisticRegression
    from bab04_data import dua_peubah
    X, y = dua_peubah()
    Cs = np.geomspace(1e-3, 1e3, 40)
    hasil = {"lbfgs": [], "liblinear": []}
    for C in Cs:
        for s in hasil:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(C=C, solver=s, tol=1e-10,
                                       max_iter=10_000).fit(X, y)
            hasil[s].append(np.r_[m.intercept_, m.coef_[0]])
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0), sharex=True)
    for ax, j, judul in zip(axs, (1, 0), ("bobot $w_1$", "intersep $b$")):
        for s, w, g in (("lbfgs", BIRU, "-"), ("liblinear", MERAH, "--")):
            ax.semilogx(Cs, np.array(hasil[s])[:, j], color=w, ls=g,
                        lw=1.0, label=s)
        ax.set_xlabel("$C$")
        ax.set_title(judul)
        kunci_label(ax, "x")
        _rapikan(ax)
    axs[1].legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    simpan(fig, "bab10-jalur")


def bab10_ukuran():
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from bab04_data import W_BENAR, dua_peubah
    ns = [25, 50, 100, 200, 400, 800, 1600, 3200]
    X, y = dua_peubah(n=max(ns))
    w_c1, w_c1n, w_mle = [], [], []
    for n in ns:
        Xn, yn = X[:n], y[:n]
        w_c1.append(LogisticRegression(C=1.0).fit(Xn, yn).coef_[0, 0])
        w_c1n.append(LogisticRegression(C=200.0 / n)
                     .fit(Xn, yn).coef_[0, 0])
        w_mle.append(sm.Logit(yn, sm.add_constant(Xn)).fit(disp=0)
                     .params[1])
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.semilogx(ns, w_mle, "-o", ms=2.5, color=ABU, lw=0.9, label="MLE")
    ax.semilogx(ns, w_c1, "-o", ms=2.5, color=BIRU, lw=1.0,
                label="$C = 1$ tetap")
    ax.semilogx(ns, w_c1n, "-s", ms=2.5, color=MERAH, lw=1.0,
                label="$C = 200/n$")
    ax.axhline(W_BENAR[0], color=HIJAU, lw=0.6, ls="--")
    ax.text(ns[0], W_BENAR[0] + 0.05, "nilai sebenarnya", fontsize=5.5,
            color=HIJAU)
    ax.set_xlabel("banyaknya titik $n$")
    ax.set_ylabel("taksiran $w_1$")
    ax.legend(fontsize=5.5, loc="lower right")
    kunci_label(ax, "x")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-ukuran")


def bab10_newton():
    from bab04_data import dua_peubah
    from bab10_kita import RegresiLogistikKita
    X, y = dua_peubah()
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    for C, w in ((0.01, HIJAU), (1.0, BIRU), (100.0, JINGGA),
                 (None, MERAH)):
        m = RegresiLogistikKita(C=C if C else 1.0,
                                penalti="l2" if C else None)
        Xt = m._rancang(X)
        s = np.ones(len(y))
        theta, norma = np.zeros(3), []
        for _ in range(8):
            f, g, H = m._bagian(theta, Xt, y.astype(float), s)
            norma.append(max(np.max(np.abs(g)), 1e-17))
            theta = theta - np.linalg.solve(H, g)
        ax.semilogy(range(8), norma, "-o", ms=2.5, lw=0.9, color=w,
                    label="tanpa penalti" if C is None
                    else "$C = " + f"{C:g}".replace(".", "{,}") + "$")
    ax.axhline(1e-10, color=ABU_GARIS, lw=0.5, ls="--")
    ax.set_xlabel("iterasi Newton")
    ax.set_ylabel(r"$\max_j |g_j|$")
    ax.legend(fontsize=5.5)
    kunci_label(ax, "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-newton")



# ============================ Bab 11 =================================

_JALUR11 = {}


def _jalur11():
    if not _JALUR11:
        from bab11_data import data_jarang
        from bab11_jalur import jalur
        _JALUR11.update(jalur(*data_jarang()))
    return _JALUR11


def bab11_kurva():
    from scipy.special import expit
    from sklearn.metrics import log_loss
    from bab11_data import W_JARANG, data_jarang
    hasil = _jalur11()
    _, _, Xu, yu = data_jarang()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0), sharey=True)
    for ax, pen, judul in zip(axs, ("l2", "l1"), ("penalti L2",
                                                  "penalti L1")):
        c = hasil[pen]
        Cs = [r["C"] for r in c]
        ax.semilogx(Cs, [r["latih"] for r in c], color=ABU, lw=1.0,
                    label="latih")
        ax.semilogx(Cs, [r["uji"] for r in c], color=BIRU, lw=1.2,
                    label="uji")
        k = int(np.argmin([r["uji"] for r in c]))
        ax.plot([Cs[k]], [c[k]["uji"]], "o", ms=3.5, color=MERAH)
        ax.axhline(log_loss(yu, expit(Xu @ W_JARANG)), color=HIJAU,
                   lw=0.6, ls="--")
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("$C$")
        ax.set_title(judul)
        kunci_label(ax, "x")
        _rapikan(ax)
    axs[0].set_ylabel("log-loss")
    axs[0].legend(fontsize=5.5, loc="upper left")
    fig.tight_layout()
    simpan(fig, "bab11-kurva")


def bab11_jalur():
    hasil = _jalur11()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.2), sharey=True)
    warna5 = [BIRU, MERAH, HIJAU, JINGGA, "#6B4C9A"]
    for ax, pen, judul in zip(axs, ("l2", "l1"), ("L2", "L1")):
        c = hasil[pen]
        Cs = [r["C"] for r in c]
        W = np.array([r["coef"] for r in c])
        for j in range(5, W.shape[1]):
            ax.semilogx(Cs, W[:, j], color=ABU_GARIS, lw=0.5)
        for j in range(5):
            ax.semilogx(Cs, W[:, j], color=warna5[j], lw=1.1)
        k = int(np.argmin([r["uji"] for r in c]))
        ax.axvline(Cs[k], color=ABU, lw=0.5, ls="--")
        ax.set_ylim(-4, 4)
        ax.set_xlabel("$C$")
        ax.set_title(f"jalur bobot, penalti {judul}")
        kunci_label(ax, "x")
        _rapikan(ax)
    axs[0].set_ylabel("bobot")
    fig.tight_layout()
    simpan(fig, "bab11-jalur")


def bab11_geometri():
    import warnings
    from sklearn.linear_model import LogisticRegression
    from bab04_data import dua_peubah
    X, y = dua_peubah()
    Cs = np.geomspace(1e-3, 10, 80)
    jalur = {"l2": [], "l1": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in Cs:
            for pen in jalur:
                m = LogisticRegression(penalty=pen, C=C, solver="saga",
                                       tol=1e-8, max_iter=100_000,
                                       random_state=0).fit(X, y)
                jalur[pen].append(m.coef_[0].copy())
        b0 = LogisticRegression(penalty=None).fit(X, y).intercept_[0]
    W1, W2 = np.meshgrid(np.linspace(-0.3, 2.2, 300),
                         np.linspace(-1.5, 0.3, 300))
    Z = b0 + W1[..., None] * X[:, 0] + W2[..., None] * X[:, 1]
    L = np.mean(np.logaddexp(0, Z) - y * Z, axis=-1)
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.contour(W1, W2, L, levels=L.min() + np.array([.005, .02, .05, .1,
                                                    .2, .35]),
               colors=ABU_GARIS, linewidths=0.5)
    for r in (0.4, 0.8):
        ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], color=MERAH, lw=0.5,
                ls=":")
        s = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(s), r * np.sin(s), color=BIRU, lw=0.5, ls=":")
    for pen, w, nama in (("l2", BIRU, "jalur L2"), ("l1", MERAH,
                                                   "jalur L1")):
        J = np.array(jalur[pen])
        ax.plot(J[:, 0], J[:, 1], color=w, lw=1.3, label=nama)
    ax.plot([0], [0], "o", ms=2.5, color=ABU)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 2.2)
    ax.set_ylim(-1.5, 0.3)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-geometri")


def bab11_dini():
    from scipy.special import expit
    from sklearn.metrics import log_loss
    from bab06_turunan import rancang
    from bab11_data import data_jarang
    from bab11_dini import lintasan_gd
    X, y, Xu, yu = data_jarang()
    jalur = lintasan_gd(X, y, 3000)
    uji = [log_loss(yu, expit(rancang(Xu) @ t)) for t in jalur]
    latih = [log_loss(y, expit(rancang(X) @ t)) for t in jalur]
    hasil = _jalur11()
    terbaik_l2 = min(r["uji"] for r in hasil["l2"])
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    k = np.arange(1, len(jalur) + 1)
    ax.semilogx(k, latih, color=ABU, lw=1.0, label="latih")
    ax.semilogx(k, uji, color=BIRU, lw=1.2, label="uji")
    ax.axhline(terbaik_l2, color=MERAH, lw=0.7, ls="--",
               label="L2 terbaik (uji)")
    ax.set_ylim(0, 2)
    ax.set_xlabel("langkah gradient descent")
    ax.set_ylabel("log-loss")
    ax.legend(fontsize=5.5, loc="upper left")
    kunci_label(ax, "x")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-dini")



# ============================ Bab 12 =================================

def bab12_profil():
    from bab04_data import jam_belajar
    from bab06_turunan import rancang
    from bab12_inferensi import (KRITIS, galat_baku, log_kem, mle,
                                 profil, selang_profil)
    x, y = jam_belajar()
    X, y = rancang(x), y.astype(float)
    theta, H = mle(X, y)
    se = galat_baku(H)[1]
    l_maks = log_kem(theta, X, y)
    ws = np.linspace(-0.3, 2.0, 120)
    dev = [2 * (l_maks - profil(X, y, 1, w, theta)) for w in ws]
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.plot(ws, dev, color=BIRU, lw=1.2, label="profil")
    ax.plot(ws, ((ws - theta[1]) / se) ** 2, color=MERAH, lw=0.9,
            ls="--", label="hampiran kuadratik (Wald)")
    ax.axhline(KRITIS, color=ABU, lw=0.6, ls=":")
    ax.text(-0.28, KRITIS + 0.25, r"$\chi^2_{1;\,0{,}95} = 3{,}84$",
            fontsize=5.5, color=ABU)
    lo, hi = selang_profil(X, y, 1, theta, se)
    for v, w, g in ((lo, BIRU, "-"), (hi, BIRU, "-"),
                    (theta[1] - 1.96 * se, MERAH, "--"),
                    (theta[1] + 1.96 * se, MERAH, "--")):
        ax.plot([v, v], [0, KRITIS], color=w, lw=0.6, ls=g)
    ax.set_ylim(0, 9)
    ax.set_xlabel("bobot $w$ jam belajar")
    ax.set_ylabel(r"$2(\ell_{\max} - \ell_{\mathrm{profil}}(w))$")
    ax.legend(fontsize=5.5, loc="upper center")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-profil")


def bab12_hauck():
    from scipy.stats import chi2
    from bab12_hauck import statistik
    ws = np.linspace(0.05, 12, 400)
    S = np.array([statistik(w) for w in ws])
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    ax.plot(ws, S[:, 0] ** 2, color=MERAH, lw=1.1, label="Wald $z^2$")
    ax.plot(ws, S[:, 1], color=BIRU, lw=1.1,
            label="rasio kemungkinan $G$")
    ax.axhline(chi2.ppf(0.95, 1), color=ABU, lw=0.6, ls=":")
    ax.set_ylim(0, 55)
    ax.set_xlabel("$w$ sebenarnya (data harapan, $n = 40$)")
    ax.set_ylabel("statistik uji $w = 0$")
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-hauck")


def bab12_sebaran():
    from bab04_data import BENIH, jam_belajar
    from bab06_turunan import rancang
    from bab12_simulasi import skenario
    rng = np.random.default_rng(BENIH)
    x, _ = jam_belajar()
    daftar = [("jam belajar, $n = 16$", rancang(x),
               np.array([-3.5903, 0.6393]), (0, 3)),
              ("garis, $n = 40$", rancang(np.linspace(-2, 2, 40)),
               np.array([0.0, 3.0]), (0, 12))]
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0))
    for ax, (nama, X, benar, rentang) in zip(axs, daftar):
        c = skenario(X, benar, 1, 1000, rng)
        tepi = np.linspace(*rentang, 40)
        ax.hist(np.clip(c["mle"], *rentang), bins=tepi, color=BIRU_MUDA,
                edgecolor=BIRU, lw=0.4, label="MLE")
        ax.hist(np.clip(c["firth"], *rentang), bins=tepi,
                histtype="step", color=MERAH, lw=0.9, label="Firth")
        ax.axvline(benar[1], color=HIJAU, lw=0.9)
        ax.set_xlabel("taksiran $w$")
        ax.set_title(nama)
        _rapikan(ax)
    axs[0].legend(fontsize=5.5, loc="upper right")
    fig.tight_layout()
    simpan(fig, "bab12-sebaran")



# ============================ Bab 13 =================================

def bab13_interaksi():
    import warnings
    import statsmodels.formula.api as smf
    from scipy.special import logit
    from bab13_data import data_pasien
    warnings.simplefilter("ignore")
    d = data_pasien()
    h = smf.logit("penyakit ~ I(usia - 55) * perokok + "
                  "C(wilayah, Treatment('kota')) + imt", d).fit(disp=0)
    usia = np.linspace(30, 80, 200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.0))
    for rokok, w, nama in ((0, BIRU, "bukan perokok"),
                           (1, MERAH, "perokok")):
        baru = {"usia": usia, "perokok": rokok, "wilayah": "kota",
                "imt": 26.0}
        import pandas as pd
        p = h.predict(pd.DataFrame(baru))
        a.plot(usia, p, color=w, lw=1.1, label=nama)
        b.plot(usia, logit(p), color=w, lw=1.1)
    a.set_xlabel("usia (tahun)")
    a.set_title("peluang penyakit")
    a.legend(fontsize=5.5, loc="upper left")
    b.set_xlabel("usia (tahun)")
    b.set_title("log-odds penyakit")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-interaksi")


def bab13_marginal():
    import warnings
    import statsmodels.formula.api as smf
    from bab13_data import data_pasien
    warnings.simplefilter("ignore")
    d = data_pasien()
    h = smf.logit("penyakit ~ usia + perokok + "
                  "C(wilayah, Treatment('kota')) + imt", d).fit(disp=0)
    p = h.predict(d)
    me = h.params["usia"] * p * (1 - p)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.hist(me, bins=40, color=BIRU_MUDA, edgecolor=BIRU, lw=0.4)
    ax.axvline(me.mean(), color=MERAH, lw=1.0, label="rata-rata (AME)")
    ax.axvline(h.params["usia"] / 4, color=HIJAU, lw=1.0, ls="--",
               label="batas $w/4$")
    ax.set_xlabel("perubahan peluang per tahun usia, per pasien")
    ax.set_ylabel("banyak pasien")
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-marginal")


def bab13_orrr():
    p0 = np.linspace(0.005, 0.95, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    for OR, w in ((1.5, HIJAU), (2.7, MERAH), (5.0, BIRU)):
        odds1 = OR * p0 / (1 - p0)
        p1 = odds1 / (1 + odds1)
        ax.plot(p0, p1 / p0, color=w, lw=1.1,
                label="OR $= " + f"{OR:g}".replace(".", "{,}") + "$")
        ax.axhline(OR, color=w, lw=0.4, ls=":")
    ax.axvline(0.17, color=ABU, lw=0.5, ls="--")
    ax.set_xlabel("peluang dasar $p_0$")
    ax.set_ylabel("rasio risiko $p_1/p_0$")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-orrr")


def bab13_simpson():
    from scipy.special import expit
    from bab04_data import BENIH
    rng = np.random.default_rng(BENIH)
    n = 20_000
    z = rng.normal(size=n)
    x = (rng.random(n) < expit(2 * z)).astype(int)
    y = (rng.random(n) < expit(-1 - 1.0 * x + 2 * z)).astype(int)
    tepi = np.quantile(z, np.linspace(0, 1, 6))
    kel = np.clip(np.digitize(z, tepi[1:-1]), 0, 4)
    fig, ax = plt.subplots(figsize=(4.0, 2.1))
    lebar = 0.35
    for i in range(5):
        for xx, w, g in ((0, BIRU, -1), (1, MERAH, 1)):
            m = (kel == i) & (x == xx)
            ax.bar(i + g * lebar / 2, y[m].mean(), lebar, color=w,
                   alpha=0.85)
    for xx, w in ((0, BIRU), (1, MERAH)):
        ax.axhline(y[x == xx].mean(), color=w, lw=0.9, ls="--")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=BIRU, label="$x = 0$"),
                       Patch(color=MERAH, label="$x = 1$"),
                       Line2D([], [], color=ABU, ls="--", lw=0.9,
                              label="seluruh data (tanpa $z$)")],
              fontsize=5.3, loc="upper left")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["$z$ terendah", "2", "3", "4", "$z$ tertinggi"],
                       fontsize=6)
    kunci_label(ax, "x")
    ax.set_ylabel("proporsi $y = 1$")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab13-simpson")



# ============================ Bab 14 =================================

def _kontur_model(ax, model, X, y, rentang, peluang=None):
    g = np.linspace(*rentang, 250)
    G1, G2 = np.meshgrid(g, g)
    P = model(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    ax.contourf(G1, G2, P, levels=[0, .25, .5, .75, 1],
                colors=[JINGGA_MUDA, "#FEF6EE", "#F6F8FC", BIRU_MUDA])
    ax.contour(G1, G2, P, levels=[.5], colors=HIJAU, linewidths=1.0)
    ax.scatter(*X[y == 1].T, s=3, color=BIRU, lw=0)
    ax.scatter(*X[y == 0].T, s=4, facecolors="none", edgecolors=JINGGA,
               lw=0.35)
    ax.set_aspect("equal")
    ax.set_xlim(*rentang)
    ax.set_ylim(*rentang)
    ax.set_xticks([])
    ax.set_yticks([])


def bab14_batas():
    import warnings
    warnings.simplefilter("ignore")
    from bab14_data import bulan, cincin
    from bab14_fitur import polinomial
    fig, axs = plt.subplots(1, 4, figsize=(4.8, 1.45))
    X, y, _ = cincin(300)
    for ax, d, judul in ((axs[0], 1, "cincin, linear"),
                         (axs[1], 2, "cincin, derajat 2")):
        m = polinomial(d, False).fit(X, y)
        _kontur_model(ax, lambda Z: m.predict_proba(Z)[:, 1], X, y,
                      (-2, 2))
        ax.set_title(judul, fontsize=6)
    X, y = bulan(200)
    for ax, pen, judul in ((axs[2], False, "bulan, derajat 9"),
                           (axs[3], True, "derajat 9, penalti CV")):
        m = polinomial(9, pen).fit(X, y)
        _kontur_model(ax, lambda Z: m.predict_proba(Z)[:, 1], X, y,
                      (-2, 3))
        ax.set_title(judul, fontsize=6)
    fig.tight_layout(pad=0.3)
    simpan(fig, "bab14-batas")


def bab14_derajat():
    import warnings
    warnings.simplefilter("ignore")
    from sklearn.metrics import log_loss
    from bab14_data import bulan
    from bab14_fitur import polinomial
    X, y = bulan(200)
    Xu, yu = bulan(5000, benih=1)
    ds = list(range(1, 10))
    hasil = {False: [], True: []}
    for d in ds:
        for pen in hasil:
            m = polinomial(d, pen).fit(X, y)
            hasil[pen].append(log_loss(yu, m.predict_proba(Xu)))
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot(ds, hasil[False], "-o", ms=2.5, color=MERAH, lw=1.0,
            label="penalti sangat lemah, $C = 10^4$")
    ax.plot(ds, hasil[True], "-o", ms=2.5, color=BIRU, lw=1.0,
            label="penalti L2, $C$ dari validasi silang")
    ax.set_ylim(0.2, 1.3)
    ax.set_xlabel("derajat polinomial")
    ax.set_ylabel("log-loss uji")
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-derajat")


def bab14_spline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer
    from scipy.special import expit
    from bab14_data import bentuk_u
    x, y, _ = bentuk_u(300)
    xx = np.linspace(0, 10, 300)[:, None]
    lin = LogisticRegression(penalty=None).fit(x, y)
    spl = make_pipeline(SplineTransformer(n_knots=5, degree=3),
                        LogisticRegression(penalty=None,
                                           max_iter=10_000)).fit(x, y)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.scatter(x[:, 0], y + np.random.default_rng(0).uniform(-.03, .03,
                                                            len(y)),
               s=3, color=ABU, lw=0)
    ax.plot(xx, expit(-1 + 0.25 * (xx - 5) ** 2), color=HIJAU, lw=1.0,
            ls="--", label="sebenarnya")
    ax.plot(xx, lin.predict_proba(xx)[:, 1], color=MERAH, lw=1.0,
            label="linear")
    ax.plot(xx, spl.predict_proba(xx)[:, 1], color=BIRU, lw=1.2,
            label="spline kubik, 5 simpul")
    ax.set_xlabel("$x$")
    ax.set_ylabel("peluang kelas 1")
    ax.legend(fontsize=5.5, loc="center", bbox_to_anchor=(0.5, 0.62))
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-spline")


def bab14_svm():
    import warnings
    warnings.simplefilter("ignore")
    from scipy.special import expit
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVC
    from bab14_data import bulan
    from bab14_kernel import GAMMA, C, logistik_kernel
    X, y = bulan(200)
    a, b = logistik_kernel(rbf_kernel(X, X, gamma=GAMMA), y, C)
    svm = SVC(kernel="rbf", gamma=GAMMA, C=C).fit(X, y)
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.3))
    _kontur_model(axs[0], lambda Z: expit(rbf_kernel(Z, X, gamma=GAMMA)
                                          @ a + b), X, y, (-2, 3))
    axs[0].set_title("regresi logistik kernel", fontsize=7)
    _kontur_model(axs[1], lambda Z: (svm.decision_function(Z) > 0)
                  .astype(float), X, y, (-2, 3))
    sv = X[svm.support_]
    axs[1].scatter(*sv.T, s=14, facecolors="none", edgecolors=MERAH,
                   lw=0.5)
    axs[1].set_title("SVC; lingkaran merah: vektor pendukung", fontsize=7)
    fig.tight_layout()
    simpan(fig, "bab14-svm")



# ============================ Bab 15 =================================

def bab15_sebaran():
    from bab15_data import latih_pasien
    _, y, p = latih_pasien()
    tepi = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.hist(p[y == 0], bins=tepi, color=JINGGA_MUDA, edgecolor=JINGGA,
            lw=0.4, label="sehat ($y = 0$)")
    ax.hist(p[y == 1], bins=tepi, histtype="step", color=BIRU, lw=1.0,
            label="sakit ($y = 1$)")
    for t, nama in ((0.5, "0,5"), (1 / 6, "1/6")):
        ax.axvline(t, color=MERAH, lw=0.7, ls="--")
        ax.text(t + 0.01, ax.get_ylim()[1] * 0.85, nama, fontsize=5.5,
                color=MERAH)
    ax.set_xlabel("peluang sakit menurut model")
    ax.set_ylabel("banyak pasien uji")
    ax.legend(fontsize=5.5, loc="upper right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-sebaran")


def bab15_roc():
    from sklearn.metrics import precision_recall_curve
    from bab15_data import latih_pasien
    from bab15_nilai import kurva_roc
    _, y, p = latih_pasien()
    _, y2, p2 = latih_pasien(["imt", "perokok", "wilayah"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.2))
    for yy, pp, w, nama in ((y, p, BIRU, "lengkap"),
                            (y2, p2, JINGGA, "tanpa usia")):
        fpr, tpr = kurva_roc(yy, pp)
        a.plot(fpr, tpr, color=w, lw=1.1, label=nama)
        pr, rc, _ = precision_recall_curve(yy, pp)
        b.plot(rc, pr, color=w, lw=1.1, label=nama)
    k = np.argmin(np.abs(np.sort(p)[::-1] - 0.5))
    a.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    a.set_xlabel("FPR = 1 - spesifisitas")
    a.set_ylabel("TPR = recall")
    a.set_title("kurva ROC")
    a.set_aspect("equal")
    b.axhline(y.mean(), color=ABU_GARIS, lw=0.6, ls="--")
    b.set_xlabel("recall")
    b.set_ylabel("precision")
    b.set_title("kurva precision-recall")
    b.set_ylim(0, 1.02)
    a.legend(fontsize=5.5, loc="lower right")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-roc")


def bab15_biaya():
    from bab15_ambang import biaya
    from bab15_data import latih_pasien
    _, y, p = latih_pasien()
    ts = np.linspace(0.01, 0.99, 197)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot(ts, [biaya(y, p, t) for t in ts], color=BIRU, lw=1.1)
    ax.axvline(1 / 6, color=MERAH, lw=0.7, ls="--")
    ax.axvline(0.5, color=ABU, lw=0.6, ls=":")
    ax.text(1 / 6 + 0.01, 1.0, "$c_{FP}/(c_{FP} + c_{FN})$",
            fontsize=5.5, color=MERAH)
    ax.set_xlabel("ambang")
    ax.set_ylabel("biaya rata-rata per pasien")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-biaya")



# ============================ Bab 16 =================================

def _reliabilitas(ax, y, p, warna, nama, k=10):
    kotak = np.minimum((p * k).astype(int), k - 1)
    xs, ys, ns = [], [], []
    for j in range(k):
        m = kotak == j
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    ax.plot(xs, ys, "-o", color=warna, lw=1.0, ms=2.5, label=nama)


def bab16_reliabilitas():
    import warnings
    warnings.simplefilter("ignore")
    from bab15_data import bagi_pasien, model_pasien
    from bab16_kalibrasi import MODEL, ece
    latih, uji = bagi_pasien()
    y = uji.penyakit.to_numpy()
    fig, axs = plt.subplots(1, 4, figsize=(4.8, 1.55), sharey=True)
    for ax, (nama, opsi, peubah) in zip(axs, MODEL):
        m = model_pasien(peubah, **opsi).fit(latih[peubah],
                                             latih.penyakit)
        p = m.predict_proba(uji[peubah])[:, 1]
        ax.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
        _reliabilitas(ax, y, p, BIRU, nama)
        ax.set_title(nama.replace("0.001", "0{,}001") if False else
                     nama.replace(".", ","), fontsize=6)
        ax.text(0.04, 0.88, "ECE " + angka(ece(y, p), 3), fontsize=5.5,
                color=MERAH)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("peluang model", fontsize=5.5)
        ax.tick_params(labelsize=5)
        _rapikan(ax)
    axs[0].set_ylabel("proporsi sakit", fontsize=5.5)
    fig.tight_layout(pad=0.3)
    simpan(fig, "bab16-reliabilitas")


def bab16_peta():
    import warnings
    warnings.simplefilter("ignore")
    from sklearn.calibration import CalibratedClassifierCV
    from bab15_data import bagi_pasien, model_pasien
    from bab16_kalibrasi import MODEL, ece
    latih, uji = bagi_pasien()
    y = uji.penyakit.to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for ax, (nama, opsi, peubah) in zip(axs, (MODEL[2], MODEL[1])):
        m = model_pasien(peubah, **opsi).fit(latih[peubah],
                                             latih.penyakit)
        p = m.predict_proba(uji[peubah])[:, 1]
        ax.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
        _reliabilitas(ax, y, p, ABU, "asli")
        for metode, w in (("sigmoid", BIRU), ("isotonic", HIJAU)):
            c = CalibratedClassifierCV(model_pasien(peubah, **opsi),
                                       method=metode, cv=5)
            q = c.fit(latih[peubah], latih.penyakit) \
                .predict_proba(uji[peubah])[:, 1]
            _reliabilitas(ax, y, q, w, metode)
        ax.set_title(nama.replace(".", ","), fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("peluang model")
        _rapikan(ax)
    axs[0].set_ylabel("proporsi sakit")
    axs[0].legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    simpan(fig, "bab16-peta")



# ============================ Bab 17 =================================

def bab17_geser():
    from scipy.special import expit, logit
    from bab17_bobot import biaya, latih
    from bab17_data import populasi
    X, y, _ = populasi()
    Xu, yu, _ = populasi(benih=1)
    a, b = latih(X, y), latih(X, y, class_weight="balanced")
    pa, pb = a.predict_proba(Xu)[:, 1], b.predict_proba(Xu)[:, 1]
    geser = np.log((len(y) - y.sum()) / y.sum())
    fig, (k, c) = plt.subplots(1, 2, figsize=(4.7, 2.1))
    ambil = np.random.default_rng(0).choice(len(pa), 3000, replace=False)
    k.scatter(pa[ambil], pb[ambil], s=2, color=BIRU, lw=0, alpha=0.5)
    q = np.geomspace(1e-4, 0.999, 300)
    k.plot(q, expit(logit(q) + geser), color=MERAH, lw=0.8,
           label=r"$\sigma(\mathrm{logit}\,p + \log\frac{n_0}{n_1})$")
    k.axhline(0.5, color=ABU_GARIS, lw=0.5, ls=":")
    k.axvline(expit(-geser), color=ABU_GARIS, lw=0.5, ls=":")
    k.set_xscale("log")
    k.set_xlim(1e-4, 1)
    k.set_xlabel("peluang tanpa bobot")
    k.set_ylabel("peluang balanced")
    k.legend(fontsize=5.3, loc="upper left")
    kunci_label(k, "x")
    ts = np.geomspace(0.002, 0.8, 120)
    c.semilogx(ts, [biaya(yu, pa, t) for t in ts], color=BIRU, lw=1.1)
    c.axvline(1 / 51, color=MERAH, lw=0.7, ls="--")
    c.axvline(expit(-geser), color=ABU, lw=0.6, ls=":")
    c.set_xlabel("ambang pada peluang tanpa bobot")
    c.set_ylabel("biaya per titik")
    kunci_label(c, "x")
    for ax in (k, c):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-geser")


def bab17_efisiensi():
    import statsmodels.api as sm
    from bab17_data import populasi
    X, y, _ = populasi()
    kasus, kontrol = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    penuh = sm.Logit(y, sm.add_constant(X)).fit(disp=0).bse[1]
    rng = np.random.default_rng(2)
    ks = [1, 2, 3, 5, 8, 12, 20, 30]
    se = []
    for k in ks:
        i = np.r_[kasus, rng.choice(kontrol, k * len(kasus),
                                    replace=False)]
        se.append(sm.Logit(y[i], sm.add_constant(X[i])).fit(disp=0).bse[1])
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot(ks, np.array(se) / penuh, "-o", ms=2.8, color=BIRU, lw=1.0)
    ax.axhline(1, color=ABU_GARIS, lw=0.6, ls="--")
    ax.set_xlabel("banyak kontrol per kasus $k$")
    ax.set_ylabel("SE($w_1$) / SE populasi")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-efisiensi")


def bab17_kasus():
    import statsmodels.api as sm
    from scipy.special import expit
    from bab17_data import populasi
    X, y, _ = populasi()
    Xu, yu, _ = populasi(benih=1)
    rng = np.random.default_rng(1)
    kasus, kontrol = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ambil = rng.choice(kontrol, size=len(kasus), replace=False)
    i = np.r_[kasus, ambil]
    h = sm.Logit(y[i], sm.add_constant(X[i])).fit(disp=0)
    koreksi = np.log(len(kontrol) / len(ambil))
    z = sm.add_constant(Xu) @ h.params
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for p, w, nama in ((expit(z), MERAH, "tanpa koreksi"),
                       (expit(z - koreksi), BIRU, "intersep dikoreksi")):
        tepi = np.quantile(p, np.linspace(0, 1, 16))
        kel = np.clip(np.digitize(p, tepi[1:-1]), 0, 14)
        xs = [p[kel == j].mean() for j in range(15)]
        ys = [yu[kel == j].mean() for j in range(15)]
        ax.loglog(xs, ys, "-o", ms=2.3, color=w, lw=1.0, label=nama)
    g = np.array([1e-4, 1])
    ax.loglog(g, g, color=ABU_GARIS, lw=0.6, ls="--")
    ax.set_xlabel("peluang model (populasi uji)")
    ax.set_ylabel("proporsi kelas 1")
    ax.legend(fontsize=5.5, loc="upper left")
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-kasus")



# ============================ Bab 18 =================================

def _model18():
    import warnings
    warnings.simplefilter("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from bab18_data import tiga_kelas
    X, y, _ = tiga_kelas()
    lr = dict(penalty=None, tol=1e-10, max_iter=10_000)
    return (X, y, LogisticRegression(**lr).fit(X, y),
            OneVsRestClassifier(LogisticRegression(**lr)).fit(X, y))


def bab18_daerah():
    from scipy.special import softmax
    from matplotlib.colors import ListedColormap
    from bab18_data import B_BENAR, W_BENAR
    X, y, soft, ovr = _model18()
    g = np.linspace(-3, 3, 300)
    G1, G2 = np.meshgrid(g, g)
    Z = np.column_stack([G1.ravel(), G2.ravel()])
    warna = [JINGGA, BIRU, HIJAU]
    muda = ListedColormap([JINGGA_MUDA, BIRU_MUDA, HIJAU_MUDA])
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.4))
    benar = softmax(Z @ W_BENAR.T + B_BENAR, axis=1).argmax(1)
    for ax, m, judul in ((axs[0], soft, "softmax"),
                         (axs[1], ovr, "one-vs-rest")):
        k = m.predict(Z).reshape(G1.shape)
        ax.contourf(G1, G2, k, levels=[-.5, .5, 1.5, 2.5], cmap=muda)
        ax.contour(G1, G2, benar.reshape(G1.shape), levels=[.5, 1.5],
                   colors=ABU, linewidths=0.6, linestyles="--")
        for j in range(3):
            ax.scatter(*X[y == j].T, s=3, color=warna[j], lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(judul, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    simpan(fig, "bab18-daerah")


def bab18_irisan():
    from scipy.special import softmax
    from bab18_data import B_BENAR, W_BENAR
    X, y, soft, ovr = _model18()
    x1 = np.linspace(-3, 3, 300)
    Z = np.column_stack([x1, np.zeros_like(x1)])
    benar = softmax(Z @ W_BENAR.T + B_BENAR, axis=1)
    ps, po = soft.predict_proba(Z), ovr.predict_proba(Z)
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    warna = [JINGGA, BIRU, HIJAU]
    for j in range(3):
        ax.plot(x1, benar[:, j], color=warna[j], lw=0.6, ls=":")
        ax.plot(x1, ps[:, j], color=warna[j], lw=1.1)
        ax.plot(x1, po[:, j], color=warna[j], lw=0.8, ls="--")
    ax.plot([], [], color=ABU, lw=0.6, ls=":", label="sebenarnya")
    ax.plot([], [], color=ABU, lw=1.1, label="softmax")
    ax.plot([], [], color=ABU, lw=0.8, ls="--", label="one-vs-rest")
    ax.set_xlabel("$x_1$ (dengan $x_2 = 0$)")
    ax.set_ylabel("peluang kelas")
    ax.legend(fontsize=5.5, loc="center left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-irisan")


def bab18_jumlah():
    from bab18_data import tiga_kelas
    _, _, _, ovr = _model18()
    Xu, _, _ = tiga_kelas(20_000, benih=1)
    s = np.column_stack([e.predict_proba(Xu)[:, 1]
                         for e in ovr.estimators_]).sum(axis=1)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    ax.hist(s, bins=60, color=BIRU_MUDA, edgecolor=BIRU, lw=0.4)
    ax.axvline(1, color=MERAH, lw=0.8, ls="--")
    ax.set_xlabel("jumlah tiga peluang biner one-vs-rest")
    ax.set_ylabel("banyak titik uji")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-jumlah")



# ============================ Bab 19 =================================

def bab19_eksplor():
    from bab19_data import baca_jantung
    d = baca_jantung()
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0), sharey=True)
    for ax, k in zip(axs, ("nyeri", "thal")):
        t = d.groupby(k).sakit.agg(["count", "mean"])
        ax.bar(range(len(t)), t["mean"], color=BIRU_MUDA, edgecolor=BIRU,
               lw=0.6)
        for i, (n, m) in enumerate(zip(t["count"], t["mean"])):
            ax.text(i, m + 0.02, f"n = {n}", ha="center", fontsize=5.5)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels(t.index, fontsize=5.5, rotation=15)
        kunci_label(ax, "x")
        ax.axhline(d.sakit.mean(), color=ABU, lw=0.5, ls="--")
        ax.set_title(k, fontsize=7)
        _rapikan(ax)
    axs[0].set_ylabel("proporsi sakit")
    fig.tight_layout()
    simpan(fig, "bab19-eksplor")


def bab19_or():
    import statsmodels.formula.api as smf
    from bab19_data import baca_jantung
    from bab19_inferensi import RUMUS, nama_pendek
    h = smf.logit(RUMUS, baca_jantung()).fit(disp=0)
    ci = h.conf_int()
    nama = [k for k in h.params.index[1:]]
    urut = np.argsort([h.params[k] for k in nama])
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    for j, i in enumerate(urut):
        k = nama[i]
        lo, hi = np.exp(ci.loc[k, 0]), np.exp(ci.loc[k, 1])
        ax.plot([max(lo, 0.02), min(hi, 50)], [j, j], color=ABU, lw=0.9)
        ax.plot(np.exp(h.params[k]), j, "o", ms=3,
                color=MERAH if (lo > 1 or hi < 1) else BIRU)
    ax.axvline(1, color=ABU_GARIS, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 50)
    ax.set_yticks(range(len(nama)))
    ax.set_yticklabels([nama_pendek(nama[i]) for i in urut], fontsize=5.5)
    ax.set_xlabel("odds ratio (selang Wald 95 persen, sumbu logaritmik)")
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab19-or")


def bab19_uji():
    from sklearn.metrics import roc_curve
    from bab19_prediksi import CS, bagi, model
    from sklearn.linear_model import LogisticRegressionCV
    Xl, Xu, yl, yu = bagi()
    m = model(LogisticRegressionCV(Cs=CS, cv=5, scoring="neg_log_loss",
                                   max_iter=10_000)).fit(Xl, yl)
    p = m.predict_proba(Xu)[:, 1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    fpr, tpr, _ = roc_curve(yu, p)
    a.plot(fpr, tpr, color=BIRU, lw=1.1)
    a.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    a.set_aspect("equal")
    a.set_xlabel("FPR")
    a.set_ylabel("TPR")
    a.set_title("ROC, data uji", fontsize=7)
    kotak = np.minimum((p * 5).astype(int), 4)
    xs = [p[kotak == j].mean() for j in range(5) if np.any(kotak == j)]
    ys = [yu[kotak == j].mean() for j in range(5) if np.any(kotak == j)]
    ns = [np.sum(kotak == j) for j in range(5) if np.any(kotak == j)]
    b.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    b.plot(xs, ys, "-o", color=BIRU, lw=1.0, ms=2.5)
    for x, yv, n in zip(xs, ys, ns):
        b.text(x + 0.02, yv - 0.07, f"{n}", fontsize=5.5, color=ABU)
    b.set_aspect("equal")
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.set_xlabel("peluang model")
    b.set_ylabel("proporsi sakit")
    b.set_title("reliabilitas, data uji", fontsize=7)
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab19-uji")



# ============================ Bab 20 =================================

def bab20_tautan():
    from scipy.special import expit
    from scipy.stats import norm
    z = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot(z, expit(z), color=BIRU, lw=1.2, label="logit: $\\sigma(z)$")
    ax.plot(z, norm.cdf(z), color=HIJAU, lw=1.0, ls="--",
            label="probit: $\\Phi(z)$")
    ax.plot(z, 1 - np.exp(-np.exp(z)), color=MERAH, lw=1.0, ls="-.",
            label="cloglog: $1 - e^{-e^{z}}$")
    ax.axhline(0.5, color=ABU_GARIS, lw=0.5, ls=":")
    ax.set_xlabel("skor $z$")
    ax.set_ylabel("peluang")
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab20-tautan")


def bab20_laplace():
    from scipy.special import expit
    from bab04_data import BENIH, jam_belajar
    from bab20_lanjut import laplace
    x, y = jam_belajar()
    theta, S = laplace(x[:, None], y, C=10.0)
    rng = np.random.default_rng(BENIH)
    sampel = rng.multivariate_normal(theta, S, size=20_000)
    xx = np.linspace(-4, 24, 300)
    V = np.column_stack([np.ones_like(xx), xx])
    P = expit(sampel @ V.T)
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.fill_between(xx, np.quantile(P, 0.05, axis=0),
                    np.quantile(P, 0.95, axis=0), color=BIRU_MUDA,
                    label="selang posterior 90 persen")
    ax.plot(xx, expit(V @ theta), color=MERAH, lw=1.0, ls="--",
            label="MAP")
    ax.plot(xx, P.mean(axis=0), color=BIRU, lw=1.2, label="prediktif")
    ax.scatter(x, y, s=8, color=np.where(y == 1, BIRU, JINGGA), zorder=3)
    ax.axvspan(1, 11, color=ABU_GARIS, alpha=0.15, lw=0)
    ax.set_xlabel("jam belajar per minggu (daerah abu-abu: data)")
    ax.set_ylabel("peluang lulus")
    ax.legend(fontsize=5.5, loc="center right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab20-laplace")


def bab20_neuron():
    from matplotlib.patches import Circle, FancyArrowPatch
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0))

    def simpul(ax, x, y, teks, warna=BIRU_MUDA, r=0.28):
        ax.add_patch(Circle((x, y), r, facecolor=warna, edgecolor=BIRU,
                            lw=0.7))
        ax.text(x, y, teks, ha="center", va="center", fontsize=6)

    def panah(ax, a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=5, lw=0.5,
                                     color=ABU))
    a = axs[0]
    for i, yy in enumerate((2.2, 1.2, 0.2)):
        simpul(a, 0.4, yy, f"$x_{i + 1}$")
        panah(a, (0.68, yy), (1.72, 1.2))
    simpul(a, 2.0, 1.2, "$z$", JINGGA_MUDA)
    panah(a, (2.28, 1.2), (2.92, 1.2))
    simpul(a, 3.2, 1.2, "$\\sigma$", HIJAU_MUDA)
    panah(a, (3.48, 1.2), (3.9, 1.2))
    a.text(4.1, 1.2, "$p$", fontsize=7, va="center")
    a.set_title("regresi logistik: satu neuron", fontsize=7)
    b = axs[1]
    for i, yy in enumerate((2.2, 1.2, 0.2)):
        simpul(b, 0.3, yy, f"$x_{i + 1}$")
    for j, yy in enumerate((2.4, 1.6, 0.8, 0.0)):
        simpul(b, 1.5, yy, "", "#F1F1F1", 0.2)
        for yi in (2.2, 1.2, 0.2):
            panah(b, (0.58, yi), (1.3, yy))
    for k, yy in enumerate((1.9, 1.2, 0.5)):
        simpul(b, 2.9, yy, "$z_{" + str(k + 1) + "}$", JINGGA_MUDA, 0.24)
        for yj in (2.4, 1.6, 0.8, 0.0):
            panah(b, (1.7, yj), (2.66, yy))
    b.text(3.3, 1.2, "softmax\n$\\to p_k$", fontsize=6, va="center")
    b.text(1.5, -0.55, "fitur hasil belajar", fontsize=5.5, ha="center",
           color=ABU)
    b.set_title("jaringan saraf: regresi softmax di lapisan akhir",
                fontsize=7)
    for ax in axs:
        ax.set_xlim(-0.1, 4.6)
        ax.set_ylim(-0.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    simpan(fig, "bab20-neuron")


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
