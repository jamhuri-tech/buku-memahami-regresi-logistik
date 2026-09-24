"""Bab 6: bentuk lanskap di sekitar MLE dan pemusatan peubah.

Hessian di MLE untuk data jam belajar dengan x mentah, x yang
dipusatkan (x - rata-rata), dan x yang dibakukan. Ketiga
parameterisasi memberi model dan log-loss yang sama, tetapi bilangan
kondisi Hessian-nya sangat berbeda.
"""
import numpy as np
import statsmodels.api as sm

from bab04_data import jam_belajar
from bab06_turunan import hessian, log_loss, rancang

if __name__ == "__main__":
    x, y = jam_belajar()
    xb, s = x.mean(), x.std()
    print(f"rata-rata x = {xb:.4f}, simpangan baku x = {s:.4f}")
    print("peubah       b        w     log-loss  kondisi  korelasi")
    for nama, u in (("mentah   ", x), ("dipusat  ", x - xb),
                    ("dibakukan", (x - xb) / s)):
        X = rancang(u)
        t = sm.Logit(y, X).fit(disp=0).params
        H = hessian(t, X, y)
        e = np.linalg.eigvalsh(H)
        kor = -H[0, 1] / np.sqrt(H[0, 0] * H[1, 1])
        print(f"{nama} {t[0]:7.4f}  {t[1]:7.4f}  {log_loss(t, X, y):.6f}"
              f"  {e[1] / e[0]:7.1f}  {kor:+.3f}")
