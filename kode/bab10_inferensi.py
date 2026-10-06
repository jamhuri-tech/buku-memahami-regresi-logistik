"""Bab 10: galat baku, uji Wald, rasio kemungkinan, skor, selang
profil, dan AIC/BIC. Fungsi-fungsi di sini dipakai lagi di Bab 11-15.
"""
import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq, linprog
from scipy.special import expit
from scipy.stats import chi2, norm

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab05_newton import newton
from bab07_prediksi import kovarians


def log_kem(w, X, y):
    z = X @ w
    return np.sum(y * z - np.logaddexp(0, z))


def mle(X, y):
    j, _ = newton(X, y, langkah=50, redam=True)
    return j[-1]


def terpisah(X, y):
    """True bila data terpisah (penuh atau kuasi): ada w dengan
    (2y_i - 1) x_i^T w >= 0 untuk semua i dan > 0 untuk sebagian."""
    A = (2 * y - 1)[:, None] * X
    r = linprog(-A.sum(axis=0), A_ub=-A, b_ub=np.zeros(len(y)),
                bounds=(-1, 1))
    return -r.fun > 1e-9


def profil(X, y, j, nilai):
    """Log-kemungkinan maksimum dengan w_j ditetapkan = nilai.

    Bobot lain ditaksir dengan Newton teredam (offset x_ij * nilai).
    """
    lain = [k for k in range(X.shape[1]) if k != j]
    offset = X[:, j] * nilai
    A = X[:, lain]
    t = np.zeros(len(lain))
    f = lambda t: log_kem(np.insert(t, j, nilai), X, y)
    for _ in range(100):
        p = expit(A @ t + offset)
        g = A.T @ (y - p)
        H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-12 * np.eye(len(t))
        s = np.linalg.solve(H, g)
        u = 1.0
        while f(t + u * s) < f(t) - 1e-12 and u > 1e-8:
            u /= 2
        t = t + u * s
        if np.abs(g).max() < 1e-10:
            break
    return f(t)


def selang_profil(X, y, j, tingkat=0.95):
    w = mle(X, y)
    lmax = log_kem(w, X, y)
    batas = chi2.ppf(tingkat, 1) / 2
    se = np.sqrt(kovarians(w, X)[j, j])
    f = lambda v: lmax - profil(X, y, j, v) - batas
    k = 1.0
    while f(w[j] - k * se) < 0:
        k *= 2
    bawah = brentq(f, w[j] - k * se, w[j], xtol=1e-10)
    k = 1.0
    while f(w[j] + k * se) < 0:
        k *= 2
    atas = brentq(f, w[j], w[j] + k * se, xtol=1e-10)
    return bawah, atas


if __name__ == "__main__":
    Xf, y = data_mini()
    X = rancang(Xf)
    m = mle_mini()
    C = kovarians(m, X)
    se = np.sqrt(np.diag(C))
    z975 = norm.ppf(0.975)
    print("(1) uji Wald per bobot:")
    print("    j     w_j      SE      z       p     selang 95%")
    for j in range(3):
        z = m[j] / se[j]
        lo, hi = m[j] - z975 * se[j], m[j] + z975 * se[j]
        print(f"    {j}  {m[j]:7.4f}  {se[j]:.4f}  {z:7.4f}"
              f"  {2 * norm.sf(abs(z)):.4f}  ({lo:.4f}, {hi:.4f})")
    b = m[1:]
    W = b @ np.linalg.solve(C[1:, 1:], b)
    print(f"    Wald gabungan w1 = w2 = 0: W = {W:.4f},"
          f" p = {chi2.sf(W, 2):.4f}")

    l1 = log_kem(m, X, y)
    l0 = log_kem(np.zeros(3), X, y)
    print("(2) rasio kemungkinan dan skor:")
    G = 2 * (l1 - l0)
    print(f"    l = {l1:.4f}, l0 = {l0:.4f}")
    print(f"    H0 w1 = w2 = 0: G = {G:.4f}, p = {chi2.sf(G, 2):.4f}")
    lk = {}
    for nama, kol in [("x1", [0, 1]), ("x2", [0, 2])]:
        lk[nama] = log_kem(mle(X[:, kol], y), X[:, kol], y)
    for h0, nama in [("w2 = 0", "x1"), ("w1 = 0", "x2")]:
        Gj = 2 * (l1 - lk[nama])
        print(f"    H0 {h0}: l(model {nama}) = {lk[nama]:.4f},"
              f" G = {Gj:.4f}, p = {chi2.sf(Gj, 1):.4f}")
    U = X.T @ (y - 0.5)
    S = U @ np.linalg.solve(X.T @ X / 4, U)
    print(f"    skor U = {U}, S = {S:.4f}, p = {chi2.sf(S, 2):.4f}")
    r = sm.Logit(y, X).fit(disp=0)
    print(f"    statsmodels: llr = {r.llr:.4f}, llr_pvalue ="
          f" {r.llr_pvalue:.4f}")

    print("(3) selang profil 95%:")
    for j in (1, 2):
        a, b = selang_profil(X, y, j)
        print(f"    w{j}: ({a:.4f}, {b:.4f}),"
              f" OR: ({np.exp(a):.4f}, {np.exp(b):.2f})")

    print("(4) AIC dan BIC (k = banyaknya bobot):")
    for nama, ll, k in [("nol", l0, 1), ("x1", lk["x1"], 2),
                        ("x2", lk["x2"], 2), ("x1 + x2", l1, 3)]:
        print(f"    {nama:<8} k = {k}: AIC = {-2 * ll + 2 * k:.4f},"
              f" BIC = {-2 * ll + k * np.log(6):.4f}")
    print(f"    statsmodels AIC = {r.aic:.4f}, BIC = {r.bic:.4f}")

    rng = np.random.default_rng(BENIH)
    wb = np.array([-1.0, 1.0, -0.5])
    hitung = {"Wald": [0, 0, 0], "profil": [0, 0, 0], "MLE": []}
    pisah, R, mm = 0, 1000, 30
    for _ in range(R):
        Xs = rancang(rng.normal(size=(mm, 2)))
        ys = (rng.random(mm) < expit(Xs @ wb)).astype(int)
        if terpisah(Xs, ys):
            pisah += 1
            continue
        w = mle(Xs, ys)
        hitung["MLE"].append(w[1])
        s = np.sqrt(kovarians(w, Xs)[1, 1])
        for nama, (lo, hi) in [("Wald", (w[1] - z975 * s,
                                          w[1] + z975 * s)),
                               ("profil", selang_profil(Xs, ys, 1))]:
            hitung[nama][0 if 1.0 < lo else (2 if 1.0 > hi else 1)] += 1
    ok = R - pisah
    print(f"(5) simulasi m = 30, w1 = 1, {R} ulangan"
          f" ({pisah} terpisah):")
    print("    selang 95%   cakupan  meleset bawah  meleset atas")
    for k in ["Wald", "profil"]:
        b, c, a = hitung[k]
        print(f"    {k:<11} {c / ok:.3f}      {b / ok:.3f}"
              f"          {a / ok:.3f}")
    print(f"    rata-rata MLE w1 = {np.mean(hitung['MLE']):.3f}")
