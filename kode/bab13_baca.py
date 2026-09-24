"""Bab 13: skala peubah, peubah kategori, dan interaksi.

(1) Odds ratio usia per tahun, per sepuluh tahun, dan per simpangan
    baku.
(2) Kategori rujukan wilayah: mengganti rujukan mengubah koefisien,
    tidak mengubah peluang. One-hot penuh dengan intersep: singular.
(3) Interaksi usia x perokok, tanpa dan dengan pemusatan usia.
"""
import warnings

import numpy as np
import statsmodels.formula.api as smf

from bab13_data import data_pasien

warnings.simplefilter("ignore")


def or_selang(hasil, nama, k=1.0):
    b, se = hasil.params[nama], hasil.bse[nama]
    return (np.exp(k * b), np.exp(k * (b - 1.96 * se)),
            np.exp(k * (b + 1.96 * se)))


if __name__ == "__main__":
    d = data_pasien()
    rumus = "penyakit ~ usia + perokok + C(wilayah, Treatment('{}')) + imt"
    h = smf.logit(rumus.format("kota"), d).fit(disp=0)
    sb = d.usia.std()
    print("(1) odds ratio usia:")
    for nama, k in (("per tahun", 1), ("per 10 tahun", 10),
                    (f"per SB ({sb:.1f} th)", sb)):
        o, lo, hi = or_selang(h, "usia", k)
        print(f"    {nama:17s}: {o:.3f} [{lo:.3f}, {hi:.3f}]")
    print("(2) rujukan wilayah:")
    for ruj in ("kota", "desa"):
        g = smf.logit(rumus.format(ruj), d).fit(disp=0)
        kof = {k.split("T.")[1].rstrip("]"): v
               for k, v in g.params.items() if "wilayah" in k}
        teks = ", ".join(f"{k} {v:+.4f}" for k, v in kof.items())
        print(f"    rujukan {ruj:4s}: {teks}")
        if ruj == "kota":
            p_kota = g.predict(d)
        else:
            beda = np.abs(g.predict(d) - p_kota).max()
            print(f"    beda peluang terbesar kedua model < 1e-10: "
                  f"{beda < 1e-10}")
    d1 = d.assign(**{f"w_{v}": (d.wilayah == v).astype(int)
                     for v in ("kota", "pinggiran", "desa")})
    g = smf.logit("penyakit ~ usia + perokok + imt + w_kota"
                  " + w_pinggiran + w_desa", d1).fit(disp=0)
    print(f"    one-hot penuh + intersep: galat baku intersep > 1e6: "
          f"{g.bse['Intercept'] > 1e6}")
    jumlah = g.params["Intercept"] + g.params["w_kota"]
    print(f"    intersep + w_kota = {jumlah:.4f} "
          f"(rujukan kota: {h.params['Intercept']:.4f})")
    print("(3) interaksi usia x perokok:")
    for nama, u in (("usia mentah   ", "usia"),
                    ("usia - 55     ", "I(usia - 55)")):
        g = smf.logit(f"penyakit ~ {u} * perokok + "
                      "C(wilayah, Treatment('kota')) + imt", d).fit(disp=0)
        print(f"    {nama}: perokok {g.params['perokok']:+.4f}, "
              f"interaksi {g.params[f'{u}:perokok']:+.4f}")
    for umur in (40, 55, 70):
        b = g.params["perokok"] + g.params["I(usia - 55):perokok"] * \
            (umur - 55)
        print(f"    OR perokok pada usia {umur}: {np.exp(b):.3f}")
