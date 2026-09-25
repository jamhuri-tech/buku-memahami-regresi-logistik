"""Bab 15-17: model pasien untuk menilai peluang.

Data pasien Bab 13 (6000 titik) dibagi menjadi 3000 latih dan 3000
uji. Model: regresi logistik tanpa penalti dengan usia, imt, perokok,
dan wilayah (one-hot, kota sebagai rujukan), tanpa interaksi.
"""
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bab13_data import data_pasien

PEUBAH = ["usia", "imt", "perokok", "wilayah"]


def bagi_pasien(n=6000):
    d = data_pasien(n)
    return d.iloc[: n // 2], d.iloc[n // 2:]


def model_pasien(peubah=PEUBAH, **opsi):
    numerik = [p for p in peubah if p != "wilayah"]
    ubah = make_column_transformer(
        (StandardScaler(), numerik),
        (OneHotEncoder(categories=[["kota", "pinggiran", "desa"]],
                       drop="first"), ["wilayah"] if "wilayah" in peubah
         else []))
    opsi = {"penalty": None, "max_iter": 10_000, **opsi}
    return make_pipeline(ubah, LogisticRegression(**opsi))


def latih_pasien(peubah=PEUBAH, **opsi):
    latih, uji = bagi_pasien()
    m = model_pasien(peubah, **opsi).fit(latih[peubah], latih.penyakit)
    p = m.predict_proba(uji[peubah])[:, 1]
    return m, uji.penyakit.to_numpy(), p


if __name__ == "__main__":
    latih, uji = bagi_pasien()
    print(f"latih {len(latih)} (penyakit {latih.penyakit.mean():.3f}), "
          f"uji {len(uji)} (penyakit {uji.penyakit.mean():.3f})")
