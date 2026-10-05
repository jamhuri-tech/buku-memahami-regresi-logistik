"""Bab 10: galat baku, uji Wald, rasio kemungkinan, skor, selang
profil, dan AIC/BIC. Fungsi-fungsi di sini dipakai lagi di Bab 11-15.
"""
import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2, norm

from bab01_data import BENIH, data_mini, mle_mini, rancang
from bab05_newton import newton
from bab07_prediksi import kovarians


def log_kem(theta, X, y):
    z = X @ theta
    return np.sum(y * z - np.logaddexp(0, z))


def mle(X, y):
    j, _ = newton(X, y, langkah=50, redam=True)
    return j[-1]


def profil(X, y, j, nilai):
    """Log-kemungkinan maksimum dengan theta_j ditetapkan = nilai.

    Parameter lain ditaksir dengan Newton teredam (offset X_j * nilai).
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
    th = mle(X, y)
    lmax = log_kem(th, X, y)
    batas = chi2.ppf(tingkat, 1) / 2
    se = np.sqrt(kovarians(th, X)[j, j])
    f = lambda v: lmax - profil(X, y, j, v) - batas
    k = 1.0
    while f(th[j] - k * se) < 0:
        k *= 2
    bawah = brentq(f, th[j] - k * se, th[j], xtol=1e-10)
    k = 1.0
    while f(th[j] + k * se) < 0:
        k *= 2
    atas = brentq(f, th[j], th[j] + k * se, xtol=1e-10)
    return bawah, atas


if __name__ == "__main__":
    x, y = data_mini()
    X = rancang(x)
    m = mle_mini()
    C = kovarians(m, X)
    se = np.sqrt(np.diag(C))
    l1 = log_kem(m, X, y)
    l0 = log_kem(np.zeros(2), X, y)
    print("(1) galat baku dan uji Wald untuk w:")
    zw = m[1] / se[1]
    print(f"    SE(b) = {se[0]:.4f}, SE(w) = {se[1]:.4f}")
    print(f"    z = {zw:.4f}, z^2 = {zw ** 2:.4f},"
          f" p = {2 * norm.sf(abs(zw)):.4f}")
    lo, hi = m[1] - norm.ppf(0.975) * se[1], m[1] + norm.ppf(0.975) * se[1]
    print(f"    selang Wald w: ({lo:.4f}, {hi:.4f}),"
          f" OR: ({np.exp(lo):.4f}, {np.exp(hi):.2f})")

    print("(2) uji rasio kemungkinan dan uji skor (H0: w = 0):")
    G = 2 * (l1 - l0)
    print(f"    l = {l1:.4f}, l0 = {l0:.4f}, G = {G:.4f},"
          f" p = {chi2.sf(G, 1):.4f}")
    print(f"    deviance = {-2 * l1:.4f}, deviance nol = {-2 * l0:.4f}")
    U = X.T @ (y - 0.5)
    S = U @ np.linalg.solve(X.T @ X / 4, U)
    print(f"    skor U = {U}, S = {S:.4f}, p = {chi2.sf(S, 1):.4f}")
    r = sm.Logit(y, X).fit(disp=0)
    print(f"    statsmodels: llr = {r.llr:.4f}, llr_pvalue ="
          f" {r.llr_pvalue:.4f}, z = {r.tvalues[1]:.4f}")

    a, b = selang_profil(X, y, 1)
    print(f"(3) selang profil w: ({a:.4f}, {b:.4f}),"
          f" OR: ({np.exp(a):.4f}, {np.exp(b):.2f})")

    print("(4) AIC dan BIC:")
    for nama, ll, k in [("model nol", l0, 1), ("model x", l1, 2)]:
        print(f"    {nama:<9}: AIC = {-2 * ll + 2 * k:.4f},"
              f" BIC = {-2 * ll + k * np.log(6):.4f}")
    print(f"    statsmodels AIC = {r.aic:.4f}, BIC = {r.bic:.4f}")

    rng = np.random.default_rng(BENIH)
    tb = np.array([-1.0, 1.0])
    hitung = {"Wald": [0, 0, 0], "profil": [0, 0, 0], "MLE": []}
    terpisah, R = 0, 1000
    for _ in range(R):
        xs = rng.normal(size=30)
        Xs = rancang(xs)
        ys = (rng.random(30) < expit(Xs @ tb)).astype(int)
        if ys.min() == ys.max() or (xs[ys == 1].min() > xs[ys == 0].max()
                                    or xs[ys == 1].max() < xs[ys == 0].min()):
            terpisah += 1
            continue
        th = mle(Xs, ys)
        hitung["MLE"].append(th[1])
        s = np.sqrt(kovarians(th, Xs)[1, 1])
        for nama, (lo, hi) in [("Wald", (th[1] - norm.ppf(0.975) * s,
                                          th[1] + norm.ppf(0.975) * s)),
                               ("profil", selang_profil(Xs, ys, 1))]:
            hitung[nama][0 if 1.0 < lo else (2 if 1.0 > hi else 1)] += 1
    ok = R - terpisah
    print(f"(5) simulasi n = 30, w = 1, {R} ulangan ({terpisah} terpisah):")
    print("    selang 95%   cakupan  meleset bawah  meleset atas")
    for k in ["Wald", "profil"]:
        b, c, a = hitung[k]
        print(f"    {k:<11} {c / ok:.3f}      {b / ok:.3f}          {a / ok:.3f}")
    print(f"    rata-rata MLE w = {np.mean(hitung['MLE']):.3f}")
