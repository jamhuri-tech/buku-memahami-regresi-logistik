"""Bab 7: log-loss pada data terpisah, dan tanggapan pustaka.

(1) Sepanjang sinar t * (-5.5, 1), log-loss data jam_terpisah turun
    ke nol tanpa pernah mencapainya.
(2) statsmodels: peringatan, bobot, dan galat baku.
(3) scikit-learn tanpa penalti: bobot ditentukan oleh tol.
(4) Pemisahan kuasi: log-kemungkinan menuju 2 log(1/2).
(5) Penalti L2 bawaan memberi bobot berhingga.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from bab06_turunan import log_loss, rancang
from bab07_data import jam_kuasi, jam_terpisah


def statsmodels_logit(x, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hasil = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    jenis = sorted({type(x.message).__name__ for x in w})
    return hasil, jenis


if __name__ == "__main__":
    x, y = jam_terpisah()
    X = rancang(x)
    arah = np.array([-5.5, 1.0])
    print("(1) sinar t * (-5.5, 1) pada data terpisah:")
    for t in (1, 2, 5, 10, 20, 40):
        print(f"    t = {t:2d}: log-loss = {log_loss(t * arah, X, y):.3e}")
    hasil, jenis = statsmodels_logit(x, y)
    b, w = hasil.params
    print("(2) statsmodels:", ", ".join(jenis))
    print(f"    converged = {hasil.mle_retvals['converged']}, "
          f"w sekitar {w:.0f}, galat baku w {hasil.bse[1]:.0e}")
    print("(3) scikit-learn, penalty=None:")
    for tol in (1e-4, 1e-8, 1e-12):
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always")
            m = LogisticRegression(penalty=None, tol=tol,
                                   max_iter=100_000).fit(x[:, None], y)
        print(f"    tol {tol:.0e}: w = {m.coef_[0, 0]:5.1f}, "
              f"-b/w = {-m.intercept_[0] / m.coef_[0, 0]:.3f}, "
              f"iter {m.n_iter_[0]}, peringatan {len(cw)}")
    xk, yk = jam_kuasi()
    hasil, jenis = statsmodels_logit(xk, yk)
    p = hasil.predict()
    print("(4) pemisahan kuasi, statsmodels:", ", ".join(jenis))
    print(f"    log-kemungkinan {hasil.llf:.4f}, 2 log(1/2) = "
          f"{2 * np.log(0.5):.4f}")
    print(f"    peluang dua mahasiswa 5 jam: {p[6]:.4f}, {p[7]:.4f}")
    m = LogisticRegression().fit(x[:, None], y)
    print(f"(5) penalti L2, C = 1: w = {m.coef_[0, 0]:.4f}, "
          f"b = {m.intercept_[0]:.4f}, -b/w = "
          f"{-m.intercept_[0] / m.coef_[0, 0]:.3f}")
