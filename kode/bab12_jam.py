"""Bab 12: inferensi pada data jam belajar, dicocokkan dengan statsmodels.

Galat baku, statistik z dan nilai-p Wald, selang Wald dan selang
profil, uji rasio kemungkinan, dan deviance.
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2, norm

from bab04_data import jam_belajar
from bab06_turunan import rancang
from bab07_data import jam_terpisah
from bab12_inferensi import firth, galat_baku, log_kem, mle, selang_profil

if __name__ == "__main__":
    x, y = jam_belajar()
    X = rancang(x)
    theta, H = mle(X, y.astype(float))
    se = galat_baku(H)
    sm_hasil = sm.Logit(y, X).fit(disp=0)
    print(f"(1) galat baku kita  : b {se[0]:.4f}, w {se[1]:.4f}")
    print(f"    galat baku sm    : b {sm_hasil.bse[0]:.4f}, "
          f"w {sm_hasil.bse[1]:.4f}")
    kov = np.linalg.inv(H)
    print(f"    korelasi taksiran b dan w: "
          f"{kov[0, 1] / (se[0] * se[1]):.3f}")
    z = theta / se
    p = 2 * norm.sf(np.abs(z))
    print(f"(2) Wald w: z = {z[1]:.4f}, nilai-p = {p[1]:.4f}; "
          f"sm: {sm_hasil.tvalues[1]:.4f}, {sm_hasil.pvalues[1]:.4f}")
    l1 = log_kem(theta, X, y)
    l0 = log_kem(np.array([np.log(y.mean() / (1 - y.mean())), 0.0]),
                 X, y)
    G = 2 * (l1 - l0)
    print(f"(3) rasio kemungkinan: G = {G:.4f}, nilai-p = "
          f"{chi2.sf(G, 1):.4f}")
    print(f"    sm: llr = {sm_hasil.llr:.4f}, nilai-p = "
          f"{sm_hasil.llr_pvalue:.4f}")
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    print(f"    deviance -2 l = {-2 * l1:.4f}; sm GLM {glm.deviance:.4f}")
    print(f"    deviance null -2 l0 = {-2 * l0:.4f}; sm GLM "
          f"{glm.null_deviance:.4f}")
    lo, hi = theta[1] - 1.96 * se[1], theta[1] + 1.96 * se[1]
    ci = sm_hasil.conf_int()[1]
    print(f"(4) selang Wald w   : [{lo:.4f}, {hi:.4f}]; "
          f"sm [{ci[0]:.4f}, {ci[1]:.4f}]")
    pl, ph = selang_profil(X, y.astype(float), 1, theta, se[1])
    print(f"    selang profil w : [{pl:.4f}, {ph:.4f}]")
    print(f"    odds ratio per jam {np.exp(theta[1]):.3f}:")
    print(f"      Wald [{np.exp(lo):.3f}, {np.exp(hi):.3f}], profil "
          f"[{np.exp(pl):.3f}, {np.exp(ph):.3f}]")
    xs, ys = jam_terpisah()
    tf = firth(rancang(xs), ys.astype(float))
    print(f"(5) Firth, jam terpisah: b = {tf[0]:.4f}, w = {tf[1]:.4f}")
    print(f"    titik setengah {-tf[0] / tf[1]:.3f}")
    tf = firth(X, y.astype(float))
    print(f"    Firth, jam belajar : b = {tf[0]:.4f}, w = {tf[1]:.4f}")
