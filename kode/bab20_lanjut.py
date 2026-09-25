"""Bab 20: tiga arah lanjutan.

(1) GLM biner dengan tautan logit, probit, dan cloglog pada data jam
    belajar (statsmodels GLM).
(2) Regresi logistik Bayes dengan hampiran Laplace: posterior kira-kira
    normal di MAP dengan kovarians invers Hessian. Peluang prediktif
    dengan hampiran probit, dibandingkan dengan Monte Carlo.
(3) Gradient boosting berloss log-loss pada data bulan, dibandingkan
    dengan regresi logistik polinomial Bab 14.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from bab04_data import BENIH, jam_belajar
from bab06_turunan import rancang
from bab10_kita import RegresiLogistikKita
from bab14_data import bulan
from bab14_fitur import polinomial

warnings.simplefilter("ignore")


def laplace(X, y, C):
    """MAP dan kovarians posterior; prior N(0, C) untuk bobot.

    Fungsi tujuan RegresiLogistikKita adalah C kali -log posterior,
    sehingga kovarians posterior adalah C kali invers Hessian-nya.
    """
    m = RegresiLogistikKita(C=C).fit(X, y)
    theta = np.r_[m.intercept_, m.coef_[0]]
    _, _, H = m._bagian(theta, rancang(X), y.astype(float),
                        np.ones(len(y)))
    return theta, C * np.linalg.inv(H)


if __name__ == "__main__":
    x, y = jam_belajar()
    X = sm.add_constant(x)
    print("(1) jam belajar, tautan:   log-kemungkinan   titik setengah")
    for nama, tautan in (("logit", sm.families.links.Logit()),
                         ("probit", sm.families.links.Probit()),
                         ("cloglog", sm.families.links.CLogLog())):
        h = sm.GLM(y, X, family=sm.families.Binomial(link=tautan)).fit()
        b, w = h.params
        xs = np.linspace(0, 12, 120001)
        p = h.predict(sm.add_constant(xs))
        setengah = xs[np.argmin(np.abs(p - 0.5))]
        print(f"    {nama:8s}              {h.llf:8.4f}         "
              f"{setengah:.3f}")
    theta, S = laplace(x[:, None], y, C=10.0)
    rng = np.random.default_rng(BENIH)
    sampel = rng.multivariate_normal(theta, S, size=200_000)
    print("(2) Laplace, prior N(0, 10) untuk w:")
    print(f"    MAP b = {theta[0]:.4f}, w = {theta[1]:.4f}; "
          f"SE w = {np.sqrt(S[1, 1]):.4f}")
    print("     x   p(MAP)   prediktif probit   Monte Carlo")
    for xx in (0.0, 5.6, 12.0, 20.0):
        v = np.array([1.0, xx])
        mu, s2 = v @ theta, v @ S @ v
        p_probit = expit(mu / np.sqrt(1 + np.pi * s2 / 8))
        p_mc = expit(sampel @ v).mean()
        print(f"    {xx:4.1f}   {expit(mu):.4f}   {p_probit:.4f}"
              f"             {p_mc:.4f}")
    Xb, yb = bulan(200)
    Xu, yu = bulan(5000, benih=1)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                        max_depth=3,
                                        random_state=0).fit(Xb, yb)
    lr = polinomial(4, True).fit(Xb, yb)
    print("(3) data bulan, log-loss uji:")
    print(f"    regresi logistik polinomial 4 + CV   "
          f"{log_loss(yu, lr.predict_proba(Xu)):.3f}")
    print(f"    gradient boosting (log-loss)         "
          f"{log_loss(yu, gb.predict_proba(Xu)):.3f}")
