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

BENIH = 20261006  # sama dengan seluruh kode/bab*.py
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

def bab01_arsitektur():
    from matplotlib.patches import Circle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def simpul(x, y, teks, r=0.38, wr=BIRU, isi=BIRU_MUDA):
        ax.add_patch(Circle((x, y), r, facecolor=isi, edgecolor=wr,
                            lw=0.8))
        ax.text(x, y, teks, ha="center", va="center", fontsize=7)

    def panah(a, b, wr=ABU, teks=None, pos=0.5, dy=0.15, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=7, color=wr, lw=0.7,
                                     linestyle=ls))
        if teks:
            x = a[0] + pos * (b[0] - a[0])
            y = a[1] + pos * (b[1] - a[1]) + dy
            ax.text(x, y, teks, ha="center", fontsize=6, color=wr)

    masuk = [(0.8, 4.3, "$1$"), (0.8, 3.2, "$x_{i1}$"),
             (0.8, 1.6, "$x_{in}$")]
    ax.text(0.8, 2.45, r"$\vdots$", ha="center", fontsize=8)
    for x, yy, t in masuk:
        simpul(x, yy, t)
    simpul(3.6, 2.9, r"$\Sigma$")
    for (x, yy, t), wt in zip(masuk, ["$w_0$", "$w_1$", "$w_n$"]):
        panah((x + 0.4, yy), (3.2, 2.9), teks=wt, pos=0.45, dy=0.12)
    simpul(5.5, 2.9, r"$\sigma$")
    panah((4.0, 2.9), (5.1, 2.9), teks="$z_i$")
    simpul(7.5, 2.9, r"$\ell_i$", wr=MERAH, isi=MERAH_MUDA)
    panah((5.9, 2.9), (7.1, 2.9), teks="$p_i$")
    ax.text(7.5, 4.3, "$y_i$", ha="center", fontsize=7)
    panah((7.5, 4.1), (7.5, 3.3))
    ax.text(9.2, 2.2, r"$\hat y_i = [p_i \geq t]$", ha="center",
            fontsize=6, color=HIJAU)
    panah((5.7, 2.55), (8.5, 2.25), wr=HIJAU, ls=":")
    yb = 1.0
    ax.text(7.5, yb + 0.05, r"$\dfrac{\partial\ell_i}{\partial p_i}$",
            ha="center", fontsize=6.5, color=MERAH)
    ax.text(5.5, yb, r"$\dfrac{\partial p_i}{\partial z_i} = p_i(1-p_i)$",
            ha="center", fontsize=6.5, color=MERAH)
    ax.text(3.0, yb, r"$\dfrac{\partial z_i}{\partial w_j} = x_{ij}$",
            ha="center", fontsize=6.5, color=MERAH)
    panah((7.0, 0.55), (3.0, 0.55), wr=MERAH)
    ax.text(5.0, 0.0,
            r"$\dfrac{\partial\ell_i}{\partial w_j} = (p_i - y_i)\,x_{ij}$",
            ha="center", fontsize=7, color=MERAH)
    for x, t in [(0.8, "masukan"), (3.6, "skor"), (5.5, "peluang"),
                 (7.5, "loss")]:
        ax.text(x, 5.0, t, ha="center", fontsize=6.5, color=ABU)
    fig.tight_layout()
    simpan(fig, "bab01-arsitektur")


def bab01_data():
    from scipy.special import expit
    from bab01_data import data_mini
    X, y = data_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    g1 = np.linspace(0, 7, 100)
    a.fill_between(g1, g1 - 2, 4, color=JINGGA_MUDA, lw=0)
    a.fill_between(g1, -1, g1 - 2, color=BIRU_MUDA, lw=0)
    a.plot(g1, g1 - 2, color=MERAH, lw=1.0)
    for c in (-1, 1):
        a.plot(g1, g1 - 2 + c, color=ABU, lw=0.6, ls="--")
    for k, (wr, mk, nama) in enumerate([(JINGGA, "s", "tidak lulus"),
                                        (BIRU, "o", "lulus")]):
        s = y == k
        a.scatter(X[s, 0], X[s, 1], color=wr, marker=mk, s=18, zorder=3,
                  label=nama)
    for i in range(6):
        a.annotate(str(i + 1), (X[i, 0] + 0.12, X[i, 1] + 0.12),
                   fontsize=6, color=ABU)
    a.set_xlim(0, 7)
    a.set_ylim(-0.6, 3.8)
    a.set_xlabel("$x_1$ (jam belajar)")
    a.set_ylabel("$x_2$ (absen)")
    a.legend(loc="upper left", fontsize=5.5)
    a.set_title(r"batas $x_2 = x_1 - 2$")
    z = np.linspace(-4, 4, 300)
    b.plot(z, expit(z), color=BIRU, lw=1.1)
    k = np.array([-1, -1, 1, 0, 0, 1]) * np.log(3)
    geser = np.array([0, 0.04, 0, 0, 0.04, -0.04])
    b.scatter(k, y + geser, c=np.where(y == 1, BIRU, JINGGA), s=14,
              zorder=3)
    b.scatter(k, expit(k), color=MERAH, s=10, zorder=4, marker="x")
    b.set_xlabel(r"skor MLE $z_i$")
    b.set_ylabel("peluang dan label")
    b.set_title(r"$p_i \in \{1/4,\ 1/2,\ 3/4\}$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab01-data")


# ============================ Bab 2 ==================================

def _profil_w0(X, y, W1, W2):
    """L minimum atas w0 untuk setiap (w1, w2) di kisi (Newton 1-D)."""
    from scipy.special import expit
    s = W1.ravel()[:, None] * X[None, :, 1] + W2.ravel()[:, None] * X[None, :, 2]
    w0 = -s.mean(1)
    for _ in range(60):
        p = expit(w0[:, None] + s)
        g = (p - y).sum(1)
        h = (p * (1 - p)).sum(1) + 1e-12
        w0 = w0 - np.clip(g / h, -1.0, 1.0)
    z = w0[:, None] + s
    return (np.logaddexp(0, z) - y * z).mean(1).reshape(W1.shape)


def bab02_loss():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    Xf, y = data_mini()
    X = rancang(Xf)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    z = np.linspace(-5, 5, 400)
    a.plot(z, np.logaddexp(0, -z), color=BIRU, lw=1.1,
           label=r"log-loss $\log(1 + e^{-z})$")
    a.plot(z, (1 - expit(z)) ** 2, color=JINGGA, lw=1.0,
           label=r"kuadrat $(1 - \sigma(z))^2$")
    a.set_xlabel("skor $z$ (label $y = 1$)")
    a.set_ylabel("loss satu sampel")
    a.set_ylim(0, 4)
    a.legend(loc="upper right", fontsize=6)
    a.set_title("loss satu sampel")
    W1, W2 = np.meshgrid(np.linspace(-0.5, 2.6, 161),
                         np.linspace(-2.8, 0.8, 161))
    Lg = _profil_w0(X, y, W1, W2)
    cs = b.contour(W1, W2, Lg, levels=[0.607, 0.62, 0.64, 0.66, 0.68,
                                       0.70, 0.75, 0.85],
                   colors=BIRU, linewidths=0.6)
    b.clabel(cs, fontsize=5, fmt=lambda v: f"{v:g}".replace(".", ","))
    m = mle_mini()
    b.plot(0, 0, "s", ms=3, color=ABU)
    b.plot(1, -1, "^", ms=3.5, color=JINGGA)
    b.plot(m[1], m[2], "o", ms=3.5, color=MERAH)
    b.set_xlabel("$w_1$")
    b.set_ylabel("$w_2$")
    b.set_title(r"$L$ minimum atas $w_0$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-loss")


# ============================ Bab 3 ==================================

def bab03_gradien():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    W1, W2 = np.meshgrid(np.linspace(-0.5, 2.6, 121),
                         np.linspace(-2.8, 0.8, 121))
    Lg = _profil_w0(X, y, W1, W2)
    a.contour(W1, W2, Lg, levels=[0.607, 0.62, 0.64, 0.66, 0.68, 0.70,
                                  0.75, 0.85], colors=ABU_GARIS,
              linewidths=0.6)
    for t1 in np.linspace(0, 2.2, 5):
        for t2 in np.linspace(-2.4, 0.4, 5):
            # w0 terbaik untuk (t1, t2), lalu gradien penuh
            s = X[:, 1] * t1 + X[:, 2] * t2
            w0 = -s.mean()
            for _ in range(60):
                p = expit(w0 + s)
                w0 -= np.clip((p - y).sum() / (p * (1 - p)).sum(), -1, 1)
            p = expit(w0 + s)
            g = X.T @ (p - y) / 6
            a.arrow(t1, t2, -g[1] * 1.5, -g[2] * 1.5, color=BIRU, lw=0.5,
                    head_width=0.06, length_includes_head=True)
    a.plot(m[1], m[2], "o", ms=3.5, color=MERAH, zorder=4)
    a.set_xlabel("$w_1$")
    a.set_ylabel("$w_2$")
    a.set_title(r"komponen $(w_1, w_2)$ dari $-\nabla L$")
    i = np.arange(1, 7)
    for w, nama, gaya, wr, dx in [
            (np.zeros(3), r"$\mathbf{w} = 0$", "s", ABU, -0.22),
            (np.array([-2.0, 1.0, -1.0]), r"$(-2, 1, -1)$", "^", JINGGA,
             0.0),
            (m, "MLE", "o", MERAH, 0.22)]:
        p = expit(X @ w)
        b.bar(i + dx, p * (1 - p), width=0.22, color=wr, label=nama)
    b.set_xticks(i)
    b.set_xlabel("mahasiswa $i$")
    b.set_ylabel("$d_i = p_i(1 - p_i)$")
    b.set_ylim(0, 0.34)
    b.legend(loc="upper center", fontsize=5.5, ncol=3)
    b.set_title("bobot Hessian per sampel")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab03-gradien")


# ============================ Bab 4 ==================================

def bab04_gd():
    from bab01_data import data_mini, mle_mini, rancang
    from bab02_loss import log_loss
    from bab04_gd import gd, sgd
    Xf, y = data_mini()
    X = rancang(Xf)
    mu, sd = Xf.mean(0), Xf.std(0)
    Xs = rancang((Xf - mu) / sd)
    m = mle_mini()
    Ls = log_loss(m, X, y)
    L = np.linalg.eigvalsh(X.T @ X / 24).max()
    Lsd = np.linalg.eigvalsh(Xs.T @ Xs / 24).max()
    jm = gd(X, y, 1 / L, 4000)
    js = gd(Xs, y, 1 / Lsd, 4000)
    jsgd = sgd(Xs, y, 1.0, 700)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    W1, W2 = np.meshgrid(np.linspace(-0.3, 1.6, 121),
                         np.linspace(-1.8, 0.5, 121))
    a.contour(W1, W2, _profil_w0(X, y, W1, W2),
              levels=[0.607, 0.62, 0.64, 0.66, 0.68, 0.70],
              colors=ABU_GARIS, linewidths=0.6)
    a.plot(jm[:, 1], jm[:, 2], "-", color=BIRU, lw=0.9, label="x mentah")
    a.plot(jm[[10, 100, 1000], 1], jm[[10, 100, 1000], 2], "o", ms=2.5,
           color=BIRU)
    w12 = js[:300, 1:] / sd
    a.plot(w12[:, 0], w12[:, 1], "-", color=JINGGA, lw=0.9,
           label="x dibakukan")
    a.plot(m[1], m[2], "*", ms=6, color=MERAH, zorder=4)
    a.set_xlabel("$w_1$")
    a.set_ylabel("$w_2$")
    a.legend(loc="lower left", fontsize=6)
    a.set_title(r"jalur GD, $\eta = 1/L$")
    k = np.arange(4001)
    for jj, A, nama, wr in [(jm, X, "GD, x mentah", BIRU),
                            (js, Xs, "GD, x dibakukan", JINGGA)]:
        sel = np.array([log_loss(t, A, y) for t in jj]) - Ls
        b.semilogy(k, np.maximum(sel, 1e-16), color=wr, lw=1.0,
                   label=nama)
    e = np.arange(len(jsgd)) * 6
    sel = np.array([log_loss(t, Xs, y) for t in jsgd]) - Ls
    b.semilogy(e, np.maximum(sel, 1e-16), color=HIJAU, lw=1.0,
               label="SGD (per epoch)")
    b.set_xlim(0, 4000)
    b.set_ylim(1e-14, 1)
    b.set_xlabel("langkah (SGD: pembaruan)")
    b.set_ylabel(r"$L(\mathbf{w}_k) - L^*$")
    b.legend(loc="upper right", fontsize=6)
    b.set_title("kecepatan turun")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-gd")


# ============================ Bab 5 ==================================

def bab05_newton():
    from bab01_data import data_mini, mle_mini, rancang
    from bab04_gd import gd
    from bab05_newton import newton
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    W1, W2 = np.meshgrid(np.linspace(-0.6, 2.6, 121),
                         np.linspace(-2.6, 0.6, 121))
    a.contour(W1, W2, _profil_w0(X, y, W1, W2),
              levels=[0.607, 0.62, 0.64, 0.66, 0.68, 0.70, 0.75, 0.85],
              colors=ABU_GARIS, linewidths=0.6)
    jn, _ = newton(X, y, langkah=8)
    a.plot(jn[:, 1], jn[:, 2], "o-", ms=2.5, lw=0.9, color=BIRU,
           label="Newton dari 0")
    jd, _ = newton(X, y, np.array([-6.0, 2.0, -2.0]), langkah=30,
                   redam=True)
    a.plot(jd[:, 1], jd[:, 2], "s-", ms=2.5, lw=0.9, color=JINGGA,
           label=r"teredam dari $(-6, 2, -2)$")
    a.plot(m[1], m[2], "*", ms=6, color=MERAH, zorder=4)
    a.set_xlabel("$w_1$")
    a.set_ylabel("$w_2$")
    a.legend(loc="lower left", fontsize=5.5)
    a.set_title(r"jalur Newton di bidang $(w_1, w_2)$")
    e = lambda j: np.maximum([np.abs(t - m).max() for t in j], 1e-16)
    L = np.linalg.eigvalsh(X.T @ X / 24).max()
    b.semilogy(e(jn), "o-", ms=2.5, lw=0.9, color=BIRU, label="Newton")
    b.semilogy(e(gd(X, y, 1 / L, 30)), "-", lw=0.9, color=ABU,
               label="GD, x mentah")
    mu, sd = Xf.mean(0), Xf.std(0)
    Xs = rancang((Xf - mu) / sd)
    Ls = np.linalg.eigvalsh(Xs.T @ Xs / 24).max()
    js = gd(Xs, y, 1 / Ls, 30)
    w12 = js[:, 1:] / sd
    asli = np.c_[js[:, 0] - w12 @ mu, w12]
    b.semilogy(e(asli), "-", lw=0.9, color=JINGGA, label="GD, x dibakukan")
    b.set_xlim(0, 30)
    b.set_ylim(1e-16, 10)
    b.set_xlabel("langkah $k$")
    b.set_ylabel(r"$\max_j|w_{k,j} - \hat w_j|$")
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
    Xf, y = data_mini()
    X = rancang(Xf)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    t = np.linspace(0, 12, 300)
    v = np.array([-3.5, 1.0, 0.0])
    a.plot(t, [log_loss(s * v, X, Y_PISAH) for s in t], color=BIRU,
           lw=1.1, label="terpisah")
    a.plot(t, [log_loss(s * v, X, y) for s in t], color=JINGGA, lw=1.0,
           label="data mini")
    a.axhline(0, color=ABU_GARIS, lw=0.5)
    a.set_xlabel(r"$t$ pada $\mathbf{w} = t\,(-3{,}5;\ 1;\ 0)$")
    a.set_ylabel(r"$L(\mathbf{w})$")
    a.legend(loc="upper right", fontsize=6)
    a.set_title("sepanjang satu sinar")
    C = np.logspace(-2, 2, 41)
    w2 = np.array([newton_l2(X, y, 1 / (6 * c)) for c in C])
    w1 = np.array([ista_l1(X, y, 1 / (6 * c), 0.2, 6000) for c in C])
    m = mle_mini()
    b.semilogx(C, w2[:, 1], color=BIRU, lw=1.1, label="$w_1$, L2")
    b.semilogx(C, w2[:, 2], color=BIRU, lw=1.0, ls="--", label="$w_2$, L2")
    b.semilogx(C, w1[:, 1], color=HIJAU, lw=1.1, label="$w_1$, L1")
    b.semilogx(C, w1[:, 2], color=HIJAU, lw=1.0, ls="--",
               label="$w_2$, L1")
    for k in (1, 2):
        b.axhline(m[k], color=MERAH, lw=0.5, ls=":")
    b.axvline(1 / 1.5, color=ABU, lw=0.5, ls=":")
    b.axhline(0, color=ABU_GARIS, lw=0.5)
    b.set_xlabel("$C$ scikit-learn")
    b.set_ylabel("bobot")
    b.legend(loc="lower left", fontsize=5.5, ncol=2)
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
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    C = kovarians(m, X)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    g = np.linspace(0, 8, 300)
    G = rancang(np.c_[g, np.ones_like(g)])
    p, lo, hi, _ = prediksi_selang(m, C, G)
    a.fill_between(g, lo, hi, color=BIRU_MUDA, lw=0, label="selang 95%")
    a.plot(g, p, color=BIRU, lw=1.1, label=r"$\hat p$")
    a.axvline(6, color=ABU_GARIS, lw=0.5, ls=":")
    a.set_xlabel("$x_1$ (jam belajar), $x_2 = 1$")
    a.set_ylabel("peluang lulus")
    a.legend(loc="upper left", fontsize=6)
    a.set_title("metode delta")
    g1 = np.linspace(0, 7, 100)
    for t, wr in [(0.5, MERAH), (0.25, HIJAU), (0.2, JINGGA)]:
        c = 2 + logit(t) / np.log(3)
        b.plot(g1, g1 - c, color=wr, lw=0.9,
               label=f"$t = {angka_mat(t, 2)}$")
    for k, (wr, mk) in enumerate([(JINGGA, "s"), (BIRU, "o")]):
        s = y == k
        b.scatter(Xf[s, 0], Xf[s, 1], color=wr, marker=mk, s=16, zorder=3)
    b.set_xlim(0, 7)
    b.set_ylim(-0.6, 3.8)
    b.set_xlabel("$x_1$")
    b.set_ylabel("$x_2$")
    b.legend(loc="upper left", fontsize=6)
    b.set_title(r"batas tebakan $\hat p = t$")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-prediksi")


# ============================ Bab 8 ==================================

def bab08_kurva():
    from sklearn import metrics
    from bab01_data import data_mini
    from bab08_performa import peluang_mini
    p, y = peluang_mini()
    Xf, _ = data_mini()
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    fpr, tpr, _ = metrics.roc_curve(y, p, drop_intermediate=False)
    a.fill_between(fpr, tpr, color=BIRU_MUDA, lw=0)
    a.plot(fpr, tpr, "o-", ms=3, color=BIRU, lw=1.0,
           label="$\\hat p$: AUC = 13/18")
    f1, t1, _ = metrics.roc_curve(y, Xf[:, 0], drop_intermediate=False)
    a.plot(f1, t1, "s--", ms=2.5, color=JINGGA, lw=0.8,
           label="$x_1$ saja: AUC = 2/3")
    a.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls=":")
    a.set_xlabel("FPR = 1 - spesifisitas")
    a.set_ylabel("TPR = recall")
    a.set_title("kurva ROC")
    a.legend(loc="lower right", fontsize=5.5)
    a.set_aspect("equal")
    rec, prec = [], []
    for t in np.unique(p)[::-1]:
        tp = np.sum((p >= t) & (y == 1))
        rec.append(tp / 3)
        prec.append(tp / np.sum(p >= t))
    b.step(np.r_[0, rec], np.r_[prec[0], prec], where="pre",
           color=JINGGA, lw=1.0)
    b.plot(rec, prec, "o", ms=3, color=JINGGA)
    b.axhline(0.5, color=ABU_GARIS, lw=0.6, ls="--")
    b.annotate("AP = 2/3", (0.45, 0.82), fontsize=7, color=JINGGA)
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
                ("3 fitur, MLE", None, slice(0, 3), HIJAU, "o"),
                ("20 fitur, MLE", None, slice(0, 20), MERAH, "s"),
                ("20 fitur, C = 0,01", 0.01, slice(0, 20), JINGGA, "^"),
                ("20 fitur, MLE + Platt", "platt", slice(0, 20), BIRU,
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
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    C = kovarians(m, X)
    lmax = log_kem(m, X, y)
    fig, sumbu = plt.subplots(1, 2, figsize=(4.7, 2.3), sharey=True)
    for ax, j, rentang in [(sumbu[0], 1, (-3, 7)), (sumbu[1], 2, (-8, 5))]:
        se = np.sqrt(C[j, j])
        w = np.linspace(*rentang, 141)
        dev = np.array([2 * (lmax - profil(X, y, j, v)) for v in w])
        ax.plot(w, dev, color=BIRU, lw=1.1, label="profil")
        ax.plot(w, ((w - m[j]) / se) ** 2, color=JINGGA, lw=1.0,
                ls="--", label="Wald")
        ax.axhline(3.8415, color=ABU, lw=0.6, ls=":")
        for v in selang_profil(X, y, j):
            ax.plot([v, v], [0, 3.8415], color=BIRU, lw=0.6)
        for v in (m[j] - 1.96 * se, m[j] + 1.96 * se):
            ax.plot([v, v], [0, 3.8415], color=JINGGA, lw=0.6, ls="--")
        g0 = 2 * (lmax - profil(X, y, j, 0.0))
        ax.plot([0], [g0], "o", ms=3, color=MERAH)
        ax.set_ylim(0, 9)
        ax.set_xlabel(f"$w_{j}$")
        ax.set_title(f"profil $w_{j}$")
        _rapikan(ax)
    sumbu[0].set_ylabel(r"$2(\hat\ell - \ell_{\mathrm{p}})$")
    sumbu[0].legend(loc="upper center", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab10-profil")


# ============================ Bab 11 =================================

def bab11_tafsiran():
    from scipy.special import expit
    from bab01_data import data_mini, mle_mini, rancang
    Xf, y = data_mini()
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
    g = np.linspace(-4, 4, 300)
    pg = expit(g)
    b.plot(g, m[1] * pg * (1 - pg), color=BIRU, lw=1.1,
           label=r"$\hat w_1\,\hat p(1 - \hat p)$")
    z = rancang(Xf) @ m
    p = expit(z)
    b.plot(z, m[1] * p * (1 - p), "o", ms=3, color=BIRU)
    ame = np.mean(m[1] * p * (1 - p))
    b.axhline(ame, color=MERAH, lw=0.8, ls="--",
              label=f"AME = {angka(ame, 4)}")
    b.set_xlabel(r"skor $z = \mathbf{x}^{\top}\hat{\mathbf{w}}$")
    b.set_ylabel(r"$\partial\hat p/\partial x_1$")
    b.set_ylim(0, 0.38)
    b.legend(loc="upper left", fontsize=6)
    b.set_title("efek marginal jam belajar")
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
    Xf, y = data_mini()
    X = rancang(Xf)
    p = expit(X @ mle_mini())
    h, _ = leverage(X, p)
    D = cook(y, p, h, 3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    g = np.linspace(0, 7, 50)
    a.plot(g, g - 2, color=ABU_GARIS, lw=0.6, ls="--")
    for k, (wr, mk) in enumerate([(JINGGA, "s"), (BIRU, "o")]):
        s = y == k
        a.scatter(Xf[s, 0], Xf[s, 1], s=12 + 60 * D[s], color=wr,
                  marker=mk, alpha=0.75, zorder=3)
        a.plot([], [], mk, ms=3, color=wr,
               label="lulus" if k else "tidak lulus")
    geser = [(-14, 6), (-14, 6), (12, -3), (8, -3), (-20, 6), (10, -10)]
    for i in range(6):
        a.annotate(angka(D[i], 2), (Xf[i, 0], Xf[i, 1]), geser[i],
                   textcoords="offset points", fontsize=5.5, color=ABU)
    a.set_xlim(0, 7.8)
    a.set_ylim(-1.2, 4.2)
    a.set_xlabel("$x_1$")
    a.set_ylabel("$x_2$")
    a.legend(loc="upper left", fontsize=5.5)
    a.set_title("jarak Cook (luas penanda)")
    rng = np.random.default_rng(BENIH)
    n = 1000
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + np.sqrt(1 - 0.81) * rng.normal(size=n)
    x3 = rng.normal(size=n)
    z = -0.5 + x1 + 0.5 * x2 + 0.8 * x3 - 0.6 * x3 ** 2
    yy = (rng.random(n) < expit(z)).astype(int)
    for A, nama, wr, gy in [(np.c_[x1, x2, x3], "linear dalam $x_3$",
                             JINGGA, "o"),
                            (np.c_[x1, x2, x3, x3 ** 2], "dengan $x_3^2$",
                             BIRU, "s")]:
        r = sm.Logit(yy, sm.add_constant(A)).fit(disp=0)
        pp = r.predict(sm.add_constant(A))
        kel = np.array_split(np.argsort(x3), 20)
        mx = [x3[k].mean() for k in kel]
        mr = [(yy[k] - pp[k]).mean() for k in kel]
        b.plot(mx, mr, gy + "-", ms=2.5, lw=0.8, color=wr, label=nama)
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
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                 random_state=0)
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


# ============================ Bab 14 =================================

def bab14_perluasan():
    from scipy.optimize import minimize
    from scipy.special import softmax
    from bab01_data import BENIH
    from bab14_perluasan import data_jarang, model, softmax_loss
    X, y = data_jarang(20000, BENIH)
    Xu, yu = data_jarang(20000, BENIH + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    for m, nama, wr, g in [(model().fit(X, y), "tanpa bobot", BIRU, "o"),
                           (model(class_weight="balanced").fit(X, y),
                            "balanced", MERAH, "s")]:
        p = m.predict_proba(Xu)[:, 1]
        kel = np.array_split(np.argsort(p), 10)
        a.plot([p[k].mean() for k in kel], [yu[k].mean() for k in kel],
               g + "-", ms=2.8, lw=0.9, color=wr, label=nama)
    a.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    a.set_xlim(0, 1)
    a.set_ylim(0, 0.4)
    a.set_xlabel("rata-rata peluang (10 kelompok)")
    a.set_ylabel("proporsi positif")
    a.legend(loc="upper left", fontsize=6)
    a.set_title("prevalensi 5%")
    rng = np.random.default_rng(BENIH + 7)
    mm = 3000
    Xm = rng.normal(size=(mm, 2))
    Wb = np.array([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0], [0.0, -0.5, 1.5]])
    A = np.c_[np.ones(mm), Xm]
    P = softmax(A @ Wb, axis=1)
    ym = np.array([rng.choice(3, p=q) for q in P])
    h = minimize(softmax_loss, np.zeros(9), args=(A, np.eye(3)[ym]),
                 jac=True, method="BFGS", options={"gtol": 1e-10})
    W = h.x.reshape(3, 3)
    g = np.linspace(-3, 3, 200)
    G1, G2 = np.meshgrid(g, g)
    Ag = np.c_[np.ones(G1.size), G1.ravel(), G2.ravel()]
    kelas = np.argmax(Ag @ W, axis=1).reshape(G1.shape)
    b.contourf(G1, G2, kelas, levels=[-0.5, 0.5, 1.5, 2.5],
               colors=[BIRU_MUDA, JINGGA_MUDA, HIJAU_MUDA])
    sub = rng.choice(mm, 300, replace=False)
    for k, wr in enumerate([BIRU, JINGGA, HIJAU]):
        s = sub[ym[sub] == k]
        b.scatter(Xm[s, 0], Xm[s, 1], s=3, color=wr, label=f"kelas {k}")
    b.set_xlabel("$x_1$")
    b.set_ylabel("$x_2$")
    b.set_aspect("equal")
    b.legend(loc="lower left", fontsize=5.5, markerscale=2, frameon=True,
             framealpha=0.85, edgecolor="none")
    b.set_title("softmax tiga kelas")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-perluasan")


# ============================ Bab 15 =================================

def bab15_studi():
    import warnings
    from sklearn.model_selection import RepeatedStratifiedKFold
    from bab10_inferensi import mle, selang_profil
    from bab15_data import baca_jantung, rancangan
    from bab15_studi import model_l2, model_mle
    warnings.simplefilter("ignore")
    d = baca_jantung()
    X, y, nama = rancangan(d)
    w = mle(X, y)
    n = X.shape[1] - 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.9, 3.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    for j in range(1, n + 1):
        lo, hi = selang_profil(X, y, j)
        yy = n - j
        a.plot([np.exp(lo), np.exp(hi)], [yy, yy], color=BIRU, lw=0.9)
        a.plot([np.exp(w[j])], [yy], "s", ms=2.8, color=BIRU)
    a.axvline(1, color=ABU, lw=0.6, ls="--")
    a.set_xscale("log")
    a.set_yticks(range(n))
    a.set_yticklabels(nama[1:][::-1], fontsize=5.5)
    kunci_label(a, "y")
    a.set_xlabel("odds ratio (selang profil 95%)")
    a.set_title("model tafsiran")
    Z = X[:, 1:]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                 random_state=0)
    for mk, nm, wr, g in [(model_mle, "MLE", MERAH, "o"),
                          (model_l2, "L2 (C dari CV)", BIRU, "s")]:
        q = np.zeros(len(y))
        for tr, te in cv.split(Z, y):
            q[te] = mk().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        kel = np.array_split(np.argsort(q), 8)
        b.plot([q[i].mean() for i in kel], [y[i].mean() for i in kel],
               g + "-", ms=2.8, lw=0.9, color=wr, label=nm)
    b.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.6, ls="--")
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.set_aspect("equal")
    b.set_xlabel("peluang (luar lipatan)")
    b.set_ylabel("proporsi sakit")
    b.legend(loc="upper left", fontsize=6)
    b.set_title("kalibrasi CV")
    for ax in (a, b):
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-studi")


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
