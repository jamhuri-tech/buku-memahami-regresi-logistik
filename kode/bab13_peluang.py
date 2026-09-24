"""Bab 13: dari koefisien ke perubahan peluang.

(1) Efek marginal rata-rata (AME) dari nol, dicocokkan dengan
    get_margeff statsmodels, dan dibandingkan dengan batas w/4.
(2) Odds ratio lawan rasio risiko untuk perokok.
(3) Ketidakruntuhan odds ratio: pada percobaan acak, menambahkan
    peubah yang tidak berkorelasi dengan perlakuan mengubah OR-nya.
"""
import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

from bab04_data import BENIH
from bab13_data import data_pasien

warnings.simplefilter("ignore")


def ame(hasil, data, peubah):
    """Rata-rata turunan peluang terhadap satu peubah numerik."""
    p = hasil.predict(data)
    return np.mean(hasil.params[peubah] * p * (1 - p))


def beda_rata(hasil, data, peubah):
    """Rata-rata selisih peluang bila peubah biner 1 lawan 0."""
    p1 = hasil.predict(data.assign(**{peubah: 1}))
    p0 = hasil.predict(data.assign(**{peubah: 0}))
    return np.mean(p1 - p0), np.mean(p1), np.mean(p0)


if __name__ == "__main__":
    d = data_pasien()
    h = smf.logit("penyakit ~ usia + perokok + "
                  "C(wilayah, Treatment('kota')) + imt", d).fit(disp=0)
    mf = h.get_margeff(at="overall", method="dydx", dummy=True)
    sm_ame = dict(zip(mf.summary_frame().index,
                      mf.summary_frame()["dy/dx"]))
    print("(1) efek marginal rata-rata:")
    for v in ("usia", "imt"):
        print(f"    {v:4s}: kita {ame(h, d, v):.5f}, sm {sm_ame[v]:.5f}"
              f", w/4 = {h.params[v] / 4:.5f}")
    selisih, p1, p0 = beda_rata(h, d, "perokok")
    print(f"    perokok: kita {selisih:.5f}, sm {sm_ame['perokok']:.5f}")
    print("(2) perokok:")
    print(f"    odds ratio          {np.exp(h.params['perokok']):.3f}")
    print(f"    rasio risiko rata2  {p1 / p0:.3f} "
          f"(p1 = {p1:.3f}, p0 = {p0:.3f})")
    rng = np.random.default_rng(BENIH)
    n = 200_000
    t = rng.integers(0, 2, n)                 # perlakuan acak
    z = rng.normal(size=n)                    # tak berkorelasi dengan t
    y = (rng.random(n) < expit(-1 + 1.0 * t + 2.0 * z)).astype(int)
    tanpa = sm.Logit(y, sm.add_constant(t)).fit(disp=0).params[1]
    dengan = sm.Logit(y, sm.add_constant(np.column_stack([t, z]))) \
        .fit(disp=0).params[1]
    print("(3) percobaan acak, n = 200000, log-OR sebenarnya 1.0:")
    print(f"    korelasi t dan z      {np.corrcoef(t, z)[0, 1]:+.4f}")
    print(f"    log-OR tanpa z        {tanpa:.4f}")
    print(f"    log-OR dengan z       {dengan:.4f}")
