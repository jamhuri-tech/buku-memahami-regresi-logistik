"""Bab 3: matriks rancangan, luapan, dan sistem linear dengan NumPy."""
import numpy as np
from scipy.linalg import hilbert
from scipy.special import expit, log_expit

from bab04_data import jam_belajar

x, y = jam_belajar()
X = np.column_stack([np.ones(len(x)), x])
theta = np.array([-3.5903, 0.6393])
z = X @ theta
p = expit(z)
print("(1) matriks rancangan jam belajar:")
print(f"    X.shape = {X.shape}, theta.shape = {theta.shape},"
      f" z.shape = {z.shape}")
print("    tiga baris pertama X:")
for baris in X[:3]:
    print("       ", baris)
print("    p tiga mahasiswa pertama:", np.round(p[:3], 4))

print("(2) luapan dan cara menghindarinya:")
with np.errstate(over="ignore", divide="ignore"):
    print(f"    np.exp(709) = {np.exp(709):.4e}")
    print(f"    np.exp(710) = {np.exp(710)}")
    print(f"    log(1 + exp(800))      = {np.log(1 + np.exp(800))}")
    print(f"    np.logaddexp(0, 800)   = {np.logaddexp(0, 800)}")
    print(f"    log(expit(-800))       = {np.log(expit(-800))}")
    print(f"    log_expit(-800)        = {log_expit(-800)}")
    print(f"    log(1 - expit(40))     = {np.log(1 - expit(40))}")
    print(f"    log_expit(-40)         = {log_expit(-40)}")

print("(3) solve lawan inv, matriks Hilbert 10 x 10:")
A = hilbert(10)
b = A @ np.ones(10)
r_solve = np.abs(A @ np.linalg.solve(A, b) - b).max()
r_inv = np.abs(A @ (np.linalg.inv(A) @ b) - b).max()
print(f"    bilangan kondisi {np.linalg.cond(A):.1e}")
print(f"    residu solve < 1e-12: {r_solve < 1e-12}")
print(f"    residu inv   > 1e-8 : {r_inv > 1e-8}")
