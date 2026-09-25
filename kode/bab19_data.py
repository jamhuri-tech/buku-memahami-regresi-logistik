"""Bab 19: data Heart Disease (Cleveland) dari UCI.

Berkas data/processed.cleveland.data (lihat data/SUMBER.md). Enam
baris dengan nilai hilang ('?') pada pembuluh atau thal dibuang. Label
'sakit' bernilai 1 bila num > 0 (penyempitan pembuluh koroner).
"""
from pathlib import Path

import pandas as pd

BERKAS = Path(__file__).resolve().parent.parent / "data" / \
    "processed.cleveland.data"
KOLOM = ["usia", "pria", "nyeri", "tensi", "kolesterol", "gula", "ekg",
         "nadi_maks", "angina_latih", "depresi_st", "lereng_st",
         "pembuluh", "thal", "num"]
KATEGORI = {
    "nyeri": {1: "angina khas", 2: "angina tak khas",
              3: "bukan angina", 4: "tanpa gejala"},
    "ekg": {0: "normal", 1: "kelainan ST-T", 2: "hipertrofi"},
    "lereng_st": {1: "naik", 2: "datar", 3: "turun"},
    "thal": {3: "normal", 6: "cacat tetap", 7: "cacat pulih"},
}
NUMERIK = ["usia", "tensi", "kolesterol", "nadi_maks", "depresi_st",
           "pembuluh"]
BINER = ["pria", "gula", "angina_latih"]


def baca_jantung():
    d = pd.read_csv(BERKAS, header=None, names=KOLOM, na_values="?")
    d = d.dropna().reset_index(drop=True)
    for k, peta in KATEGORI.items():
        d[k] = d[k].astype(int).map(peta)
    for k in BINER + ["pembuluh"]:
        d[k] = d[k].astype(int)
    d["sakit"] = (d.num > 0).astype(int)
    return d.drop(columns="num")


if __name__ == "__main__":
    mentah = pd.read_csv(BERKAS, header=None, names=KOLOM, na_values="?")
    d = baca_jantung()
    print(f"baris mentah {len(mentah)}, nilai hilang "
          f"{int(mentah.isna().sum().sum())}, tersisa {len(d)}")
    print(f"sakit {d.sakit.sum()} ({d.sakit.mean():.3f})")
    for k in ("nyeri", "thal"):
        t = d.groupby(k).sakit.agg(["count", "mean"]).round(3)
        print(f"{k}:")
        for nama, baris in t.iterrows():
            print(f"    {nama:16s} n = {int(baris['count']):3d}, "
                  f"proporsi sakit {baris['mean']:.3f}")
