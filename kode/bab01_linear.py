"""Bab 1: regresi linear lawan regresi logistik pada label 0/1.

Data jam belajar Bab 4, lalu data yang sama ditambah tiga mahasiswa
yang belajar sangat lama (25, 30, 35 jam) dan lulus. Titik setengah
adalah jam belajar tempat model memberi peluang 0,5.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from bab04_data import jam_belajar

x, y = jam_belajar()
x2 = np.r_[x, 25.0, 30.0, 35.0]
y2 = np.r_[y, 1, 1, 1]


def dua_model(x, y):
    X = x[:, None]
    lin = LinearRegression().fit(X, y)
    log = LogisticRegression(penalty=None, tol=1e-10).fit(X, y)
    return lin, log


print("(1) garis kuadrat terkecil pada label 0/1:")
lin, log = dua_model(x, y)
a, c = lin.intercept_, lin.coef_[0]
print(f"    y = {a:.4f} + {c:.4f} x")
for j in [0.0, 1.0, 11.0, 15.0, 20.0]:
    print(f"    x = {j:4.1f}:  garis {a + c * j:7.4f}"
          f"   logistik {log.predict_proba([[j]])[0, 1]:.4f}")

print("(2) titik setengah dan tebakan untuk 6,5 jam:")
print("    data               model      titik setengah  p(6,5 jam)")
for nama, (xx, yy) in [("16 mahasiswa", (x, y)),
                       ("+ 3 belajar lama", (x2, y2))]:
    lin, log = dua_model(xx, yy)
    t_lin = (0.5 - lin.intercept_) / lin.coef_[0]
    t_log = -log.intercept_[0] / log.coef_[0, 0]
    p_lin = lin.predict([[6.5]])[0]
    p_log = log.predict_proba([[6.5]])[0, 1]
    print(f"    {nama:<18} linear     {t_lin:8.3f}        {p_lin:.4f}")
    print(f"    {'':<18} logistik   {t_log:8.3f}        {p_log:.4f}")
