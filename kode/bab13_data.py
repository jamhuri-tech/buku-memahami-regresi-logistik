"""Bab 13: data pasien rekaan dengan parameter yang diketahui.

Peubah: usia (tahun), imt (indeks massa tubuh), perokok (0/1), dan
wilayah (kota, pinggiran, desa). Label: penyakit (0/1) dari model
    logit = -2 + 0.05 (usia - 55) + 0.8 perokok
            + 0.03 (usia - 55) perokok + 0.4 [pinggiran] + 0.7 [desa]
            + 0.08 (imt - 26).
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from bab04_data import BENIH


def data_pasien(n=2000, benih=BENIH):
    rng = np.random.default_rng(benih)
    usia = rng.uniform(30, 80, n)
    perokok = (rng.random(n) < 0.3).astype(int)
    wilayah = rng.choice(["kota", "pinggiran", "desa"], size=n,
                         p=[0.5, 0.3, 0.2])
    imt = rng.normal(26, 4, n)
    u = usia - 55
    z = (-2 + 0.05 * u + 0.8 * perokok + 0.03 * u * perokok
         + 0.4 * (wilayah == "pinggiran") + 0.7 * (wilayah == "desa")
         + 0.08 * (imt - 26))
    penyakit = (rng.random(n) < expit(z)).astype(int)
    return pd.DataFrame({"usia": usia, "imt": imt, "perokok": perokok,
                         "wilayah": wilayah, "penyakit": penyakit})


if __name__ == "__main__":
    d = data_pasien()
    print(f"n = {len(d)}, penyakit = {d.penyakit.sum()} "
          f"({d.penyakit.mean():.3f})")
    print(d.head(3).round(2).to_string())
